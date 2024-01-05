#!/usr/bin/env python3

from argparse import ArgumentParser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import HTTPServer, BaseHTTPRequestHandler
from lib.img_to_text import TextRenderer, generate_glyph_cache
from lib.img_sampler import (
    compute_glyph_diff_with_brightness_scores_for_samples,
    NestedDirImageLoader,
    get_labels_for_samples,
    extract_w_h_samples,
    compute_brightness_scores_for_samples,
    compute_glyph_diff_scores_for_samples,
    num_samples_for_img,
)
from pathlib import Path
from PIL import Image

import torchvision.transforms.functional as transforms
import tempfile

import json
import subprocess
import torch
import numpy
import urllib.parse

from lib.load_network import (
    load_model_params,
    load_network_from_model_params_and_weights,
)

NETWORK_OUTPUT_LABEL = "network_output"


def serve_numpy_as_png(img, wfile):
    img = Image.fromarray(img).convert(mode="L")
    img.save(wfile, format="png")


def get_scores_from_network(samples, net):
    n, h, w = samples.shape
    samples = samples.reshape(n, h * w) / 255.0

    scores = net.forward(samples)
    return scores


def get_labels_from_network(samples, net):
    scores = get_scores_from_network(samples, net)
    labels = scores.max(dim=1)[1]

    return labels


def get_labels_from_scorer(samples, blurred_glyph_cache, scoring_fn):
    samples_for_comparison = transforms.resize(
        samples,
        [blurred_glyph_cache.shape[1], blurred_glyph_cache.shape[2]],
    )
    labels = get_labels_for_samples(
        samples_for_comparison,
        blurred_glyph_cache,
        scoring_fn=scoring_fn,
    )
    return labels


def path_to_mimetype(path):
    suffixes = {
        ".js": "text/javascript",
        ".html": "text/html",
        ".css": "text/css",
    }
    return suffixes[Path(path).suffix]


def open_image_filtered(path):
    with tempfile.TemporaryDirectory() as d:
        output = f"{d}/output.png"
        # Local contrast enhancement. Overall we care more that details are
        # preserved than brightness is perfect. Boost differences while still
        # attempting to preserve some semblance of a reasonable image
        subprocess.run(
            ["magick", str(path), "-clahe", "20x20%+128+2", output], check=True
        )

        img = Image.open(output).convert(mode="L")
        img = torch.tensor(numpy.array(img)).to(torch.float32)

        # Looking at other image network normalization params... normalization
        # happens with ~115 mean and ~55 std
        #
        # If we look at our glyph brightness distribution it skews very dark.
        # Around a 50 mean and 25 std
        #
        # Darken the whole image to improve the chances that shapes will match
        # cleanly. Brighten above the mean to preserve very bright areas.
        standard_mean = 114
        new_mean = 75
        new_img = img - standard_mean + new_mean

        idx = new_img > new_mean
        new_img[idx] = (new_img[idx] - new_mean) * 1.5 + new_mean
        new_img.clamp_(0.0, 255.0)

        new_img = new_img.to(torch.uint8).squeeze()
        return Image.fromarray(new_img.cpu().numpy())


def query_params_to_sample(url, img_sampler, device):
    parsed_path = urllib.parse.urlparse(url)
    query_params = urllib.parse.parse_qs(parsed_path.query)
    image_id = int(float(query_params["image_id"][0]))
    x = int(float(query_params["x"][0]))
    y = int(float(query_params["y"][0]))
    metric = query_params.get("metric", None)
    if metric is not None:
        metric = metric[0]
    sample_height = img_sampler.sample_height
    sample_width = img_sampler.sample_width
    img = open_image_filtered(img_sampler.images[image_id]).convert(mode="L")
    img = torch.tensor(numpy.array(img), device=device)
    img = img[y : y + sample_height, x : x + sample_width]
    return img, x, y, metric


class VisualizerServer(HTTPServer):
    def __init__(
        self,
        img_sampler: NestedDirImageLoader,
        glyph_cache,
        score_metrics,
        net,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.img_sampler = img_sampler
        self.glyph_cache = glyph_cache
        self.score_metrics = score_metrics
        self.net = net


class VisualizerRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Some type hints for the indexer
        assert isinstance(self.server, VisualizerServer)
        self.server: VisualizerServer

    def _set_response_headers(self, content_type):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", content_type)
        self.end_headers()

    def _get_glyphs(self):
        self._set_response_headers("application/json")
        output = json.dumps({"num_glyphs": self.server.glyph_cache.shape[0]})
        self.wfile.write(output.encode())

    def _get_glyph(self):
        self._set_response_headers("image/png")
        glyph_num = self.path[8:]
        img = self.server.glyph_cache[int(glyph_num)]
        serve_numpy_as_png(img.cpu().numpy(), self.wfile)

    def _get_label_metrics(self):
        self._set_response_headers("application/json")
        options = []
        if self.server.net is not None:
            options.append(NETWORK_OUTPUT_LABEL)
        options += list(self.server.score_metrics.keys())
        options = json.dumps(options)
        self.wfile.write(options.encode())

    def _get_image_input(self):
        parsed_path = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_path.query)
        id = int(query_params["id"][0])
        img_path = self.server.img_sampler.images[id]
        img = Image.open(img_path).convert(mode="L")

        self._set_response_headers("image/png")
        img.save(self.wfile, format="png")

    def _get_image_output(self):
        parsed_path = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_path.query)
        id = int(query_params["id"][0])
        metric = query_params["metric"][0]

        img_path = self.server.img_sampler.images[id]
        img = torch.tensor(
            numpy.array(open_image_filtered(img_path).convert(mode="L")),
            device=self.server.glyph_cache.device,
        )
        _, num_x_samples = num_samples_for_img(
            img,
            self.server.img_sampler.sample_width,
            self.server.img_sampler.sample_height,
        )
        samples = extract_w_h_samples(
            int(self.server.img_sampler.sample_width),
            int(self.server.img_sampler.sample_height),
            img,
        )

        labels = self._get_labels_for_samples(samples, metric)
        renderer = TextRenderer(self.server.glyph_cache)
        img = renderer.render(labels, num_x_samples)
        img = Image.fromarray(img.cpu().numpy())
        self._set_response_headers("image/png")
        img.save(self.wfile, format="png")

    def _get_sample_size(self):
        self._set_response_headers("application/json")
        response = {
            "width": self.server.img_sampler.sample_width,
            "height": self.server.img_sampler.sample_height,
        }
        self.wfile.write(json.dumps(response).encode())

    def _get_sample_input(self):
        img, _, _, _ = query_params_to_sample(
            self.path, self.server.img_sampler, self.server.glyph_cache.device
        )
        self._set_response_headers("image/png")
        img = serve_numpy_as_png(img.cpu().numpy(), self.wfile)

    def _get_sample_output(self):
        sample, _, _, metric = query_params_to_sample(
            self.path, self.server.img_sampler, self.server.glyph_cache.device
        )

        sample = sample.unsqueeze(0)
        label = self._get_labels_for_samples(sample, metric)
        img = self.server.glyph_cache[label[0]]

        self._set_response_headers("image/png")
        img = serve_numpy_as_png(img.cpu().numpy(), self.wfile)

    def _get_sample_metadata(self):
        sample, _, _, metric = query_params_to_sample(
            self.path, self.server.img_sampler, self.server.glyph_cache.device
        )
        sample = sample.unsqueeze(0)

        if metric == NETWORK_OUTPUT_LABEL:
            scores = get_scores_from_network(sample, self.server.net)
            scores *= -1
            scores = scores.cpu()[0].tolist()
        else:
            sample = transforms.resize(
                sample,
                [self.server.glyph_cache.shape[1], self.server.glyph_cache.shape[2]],
            )
            scores = self.server.score_metrics[metric](
                sample,
                self.server.img_sampler.blurred_glyph_cache,
            )

            scores = scores.cpu()[0].tolist()

        self._set_response_headers("application/json")
        self.wfile.write(json.dumps(scores).encode())

    def _get_labels_for_samples(self, samples, metric):
        if metric == NETWORK_OUTPUT_LABEL:
            labels = get_labels_from_network(samples, self.server.net)
        else:
            labels = get_labels_from_scorer(
                samples,
                self.server.img_sampler.blurred_glyph_cache,
                self.server.score_metrics[metric],
            )
        return labels

    def _serve_file(self):
        file_path = "res" + self.path
        if not Path(file_path).exists():
            self.send_response(404)
            return

        self._set_response_headers(path_to_mimetype(self.path))
        with open(file_path) as f:
            self.wfile.write(f.read().encode())

    def do_GET(self):
        assert isinstance(self.server, VisualizerServer)

        @dataclass
        class StartsWith:
            start: str

        path_mapping = [
            ("/glyphs", self._get_glyphs),
            (StartsWith("/glyphs/"), self._get_glyph),
            ("/label_metrics", self._get_label_metrics),
            ("/num_images", self._get_label_metrics),
            (StartsWith("/input?"), self._get_image_input),
            (StartsWith("/output?"), self._get_image_output),
            ("/sample_size", self._get_sample_size),
            (StartsWith("/sample_input?"), self._get_sample_input),
            (StartsWith("/sample_output?"), self._get_sample_output),
            (StartsWith("/sample_metadata?"), self._get_sample_metadata),
        ]

        for matcher, f in path_mapping:
            if isinstance(matcher, str) and self.path != matcher:
                continue
            elif isinstance(matcher, StartsWith) and not self.path.startswith(
                matcher.start
            ):
                continue

            f()
            return

        # On disk files aren't handled in the path mapping. If we haven't
        # matched yet, check the disk
        self._serve_file()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--network-params", dest="network_params_path", default=None)
    parser.add_argument("--network-weights", default=None)

    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sample-width", type=int, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    parsed_args = parser.parse_args()

    network_params_provided = parsed_args.network_params_path is None
    network_weights_provided = parsed_args.network_weights is None
    if network_weights_provided != network_params_provided:
        raise RuntimeError(
            "--network-params and --network-weights are both required if one is provided"
        )

    return parsed_args


def main(data_path, sample_width, device, network_params_path, network_weights):
    _, glyph_cache = generate_glyph_cache(device=device)
    sample_height = int(sample_width * glyph_cache.shape[1] / glyph_cache.shape[2])
    sampler = NestedDirImageLoader(
        data_path, sample_width, sample_height, glyph_cache, device
    )

    net = None
    if network_params_path is not None and network_weights is not None:
        network_params = load_model_params(network_params_path)
        net = load_network_from_model_params_and_weights(
            network_params, network_weights, device
        )
        pass

    score_metrics = {
        "training_labels": compute_glyph_diff_with_brightness_scores_for_samples,
        "blurred_diff": compute_glyph_diff_scores_for_samples,
        "brightness": compute_brightness_scores_for_samples,
    }

    server = VisualizerServer(
        sampler,
        glyph_cache,
        score_metrics,
        net,
        ("localhost", 9999),
        RequestHandlerClass=VisualizerRequestHandler,
    )
    server.serve_forever()


if __name__ == "__main__":
    main(**vars(parse_args()))
