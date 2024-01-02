#!/usr/bin/env python3

from argparse import ArgumentParser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import HTTPServer, BaseHTTPRequestHandler
from lib.img_to_text import GlyphRenderer
from lib.img_sampler import (
    compute_glyph_diff_with_brightness_scores_for_samples,
    ImgSampler,
    get_labels_for_samples,
    extract_w_h_samples,
    compute_brightness_scores_for_samples,
    compute_glyph_diff_scores_for_samples,
    num_samples_for_img,
    sample_to_sample_for_comparison,
)
from pathlib import Path
from PIL import Image
from typing import Dict

import json
import torch
import urllib.parse


def serve_numpy_as_png(img, wfile):
    img = Image.fromarray(img).convert(mode="L")
    img.save(wfile, format="png")


def path_to_mimetype(path):
    suffixes = {
        ".js": "text/javascript",
        ".html": "text/html",
        ".css": "text/css",
    }
    return suffixes[Path(path).suffix]


def query_params_to_sample(url, img_sampler):
    parsed_path = urllib.parse.urlparse(url)
    query_params = urllib.parse.parse_qs(parsed_path.query)
    x = int(float(query_params["x"][0]))
    y = int(float(query_params["y"][0]))
    metric = query_params.get("metric", None)
    if metric is not None:
        metric = metric[0]
    sample_height = img_sampler.sample_height
    sample_width = img_sampler.sample_width
    img = img_sampler.img[y : y + sample_height, x : x + sample_width]
    return img, x, y, metric


class VisualizerServer(HTTPServer):
    def __init__(
        self,
        img_sampler: ImgSampler,
        input_img,
        output_imgs,
        score_metrics,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.img_sampler = img_sampler
        self.output_imgs: Dict[str, torch.Tensor] = output_imgs
        self.input_img = input_img
        self.score_metrics = score_metrics


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
        output = json.dumps(
            {"num_glyphs": self.server.img_sampler.glyph_cache.shape[0]}
        )
        self.wfile.write(output.encode())

    def _get_glyph(self):
        self._set_response_headers("image/png")
        glyph_num = self.path[8:]
        img = self.server.img_sampler.glyph_cache[int(glyph_num)]
        serve_numpy_as_png(img.cpu().numpy() * 255.0, self.wfile)

    def _get_label_metrics(self):
        self._set_response_headers("application/json")
        options = json.dumps(list(self.server.output_imgs.keys()))
        self.wfile.write(options.encode())

    def _get_image_input(self):
        self._set_response_headers("image/png")
        serve_numpy_as_png(self.server.input_img.cpu().numpy(), self.wfile)

    def _get_image_output(self):
        parsed_path = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_path.query)
        metric = query_params["metric"][0]

        self._set_response_headers("image/png")
        serve_numpy_as_png(
            self.server.output_imgs[metric].cpu().cpu().numpy(), self.wfile
        )

    def _get_sample_size(self):
        self._set_response_headers("application/json")
        response = {
            "width": self.server.img_sampler.sample_width,
            "height": self.server.img_sampler.sample_height,
        }
        self.wfile.write(json.dumps(response).encode())

    def _get_sample_input(self):
        img, _, _, _ = query_params_to_sample(self.path, self.server.img_sampler)
        self._set_response_headers("image/png")
        img = serve_numpy_as_png(img.cpu().numpy() * 255.0, self.wfile)

    def _get_sample_output(self):
        sample, x, y, metric = query_params_to_sample(
            self.path, self.server.img_sampler
        )
        sample_for_comparison = sample_to_sample_for_comparison(
            x,
            y,
            sample,
            self.server.img_sampler.img,
            self.server.img_sampler.img_for_comparison,
        )
        label = get_labels_for_samples(
            sample_for_comparison.unsqueeze(0),
            self.server.img_sampler.blurred_glyph_cache,
            self.server.score_metrics[metric],
        )
        img = self.server.img_sampler.glyph_cache[label[0]]

        self._set_response_headers("image/png")
        img = serve_numpy_as_png(img.cpu().numpy() * 255.0, self.wfile)

    def _get_sample_metadata(self):
        sample, x, y, metric = query_params_to_sample(
            self.path, self.server.img_sampler
        )
        sample_for_comparison = sample_to_sample_for_comparison(
            x,
            y,
            sample,
            self.server.img_sampler.img,
            self.server.img_sampler.img_for_comparison,
        )
        scores = self.server.score_metrics[metric](
            sample_for_comparison.unsqueeze(0),
            self.server.img_sampler.blurred_glyph_cache,
        )
        scores = scores.cpu()[0].tolist()
        self._set_response_headers("application/json")
        self.wfile.write(json.dumps(scores).encode())

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
            ("/input", self._get_image_input),
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
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--sample-width", type=int, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main(image_path, sample_width, device):
    renderer = GlyphRenderer()
    sample_height = int(sample_width / renderer.char_aspect())
    sampler = ImgSampler(sample_width, sample_height, renderer, image_path, device)

    num_y_samples, num_x_samples = num_samples_for_img(
        sampler.img, sample_width, sample_height
    )

    sample_height_comparison = sampler.img_for_comparison.shape[0] / num_y_samples
    sample_width_comparison = sampler.img_for_comparison.shape[1] / num_x_samples

    score_metrics = {
        "training_labels": compute_glyph_diff_with_brightness_scores_for_samples,
        "blurred_diff": compute_glyph_diff_scores_for_samples,
        "brightness": compute_brightness_scores_for_samples,
    }

    output_imgs = {}
    for label_metric, score_fn in score_metrics.items():
        print(label_metric)
        data_for_comparison = extract_w_h_samples(
            int(sample_width_comparison),
            int(sample_height_comparison),
            sampler.img_for_comparison,
        )
        labels = get_labels_for_samples(
            data_for_comparison, sampler.blurred_glyph_cache, score_fn
        )
        output_img = torch.zeros_like(sampler.img_for_comparison, dtype=torch.float32)

        char_lookup = [
            chr(x) for x in renderer.glyph_cache.keys() if isinstance(x, int)
        ]
        for y in range(num_y_samples):
            for x in range(num_x_samples):
                label = labels[y * num_x_samples + x]
                output_y_start = int(y * sample_height_comparison)
                output_y_end = int(output_y_start + sample_height_comparison)
                output_x_start = int(x * sample_width_comparison)
                output_x_end = int(output_x_start + sample_width_comparison)
                output_img[
                    output_y_start:output_y_end, output_x_start:output_x_end
                ] = sampler.glyph_cache[label]
                print(char_lookup[label], end="")
            print()

        output_imgs[label_metric] = output_img * 255.0

    server = VisualizerServer(
        sampler,
        sampler.img * 255.0,
        output_imgs,
        score_metrics,
        ("localhost", 9999),
        RequestHandlerClass=VisualizerRequestHandler,
    )
    server.serve_forever()


if __name__ == "__main__":
    main(**vars(parse_args()))
