#!/usr/bin/env python3

from lib.img_to_text import GlyphRenderer
from lib.img_sampler import (
    ImgSampler,
    num_samples_for_img,
    extract_w_h_samples,
)

from argparse import ArgumentParser
from PIL import Image

import numpy
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


class Network(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        return self.linear1(x)


def render_full_image(img, glyph_cache, sample_width, sample_height, iter, net):
    data = extract_w_h_samples(sample_width, sample_height, img)
    predictions = net(
        data.reshape((data.shape[0], sample_width * sample_height)).to(torch.float32)
    )
    num_y_samples, num_x_samples = num_samples_for_img(img, sample_width, sample_height)
    output_img = numpy.zeros(
        (
            num_y_samples * glyph_cache.shape[1],
            num_x_samples * glyph_cache.shape[2],
        ),
        dtype=numpy.float32,
    )

    char_idxs = torch.max(predictions, 1)[1]
    for y in range(num_y_samples):
        output_y_start = y * glyph_cache.shape[1]
        output_y_end = output_y_start + glyph_cache.shape[1]
        for x in range(num_x_samples):
            char_idx = char_idxs[y * num_x_samples + x]

            rendered = glyph_cache[char_idx]
            output_x_start = x * glyph_cache.shape[2]
            output_x_end = output_x_start + glyph_cache.shape[2]
            output_img[
                output_y_start:output_y_end, output_x_start:output_x_end
            ] = rendered.cpu().numpy()

    Image.fromarray(output_img * 255.0).convert(mode="L").save("{}.png".format(iter))


def main(image_path, device):
    renderer = GlyphRenderer()
    sample_width = 12
    sample_height = int(sample_width / renderer.char_aspect())
    sampler = ImgSampler(sample_width, sample_height, renderer, image_path, device)
    net = Network(sample_width * sample_height, sampler.glyph_cache.shape[0]).to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    i = 0

    while True:
        optimizer.zero_grad()
        data, labels = sampler.get_samples()
        data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
        data = data.to(torch.float32)
        output = net(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        i += 1
        if i % 1000 == 1:
            print("loss", loss)
            render_full_image(
                sampler.img, sampler.glyph_cache, sample_width, sample_height, i, net
            )


if __name__ == "__main__":
    main(**vars(parse_args()))
