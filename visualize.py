#!/usr/bin/env python3

from argparse import ArgumentParser
from lib.img_to_text import GlyphRenderer, CHAR_LOOKUP
from lib.img_sampler import (
    ImgSampler,
    get_samples_and_labels_for_img,
    num_samples_for_img,
)
from PIL import Image

import torch


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--sample-width", type=int, required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main(image_path, sample_width, output_path):
    renderer = GlyphRenderer()
    sample_height = int(sample_width / renderer.char_aspect())
    sampler = ImgSampler(sample_width, sample_height, renderer, image_path)

    num_y_samples, num_x_samples = num_samples_for_img(
        sampler.img, sample_width, sample_height
    )

    data, labels = get_samples_and_labels_for_img(
        sample_width,
        sample_height,
        sampler.img,
        sampler.img_for_comparison,
        sampler.glyph_cache,
    )

    output_img = torch.zeros_like(sampler.img)

    for y in range(num_y_samples):
        for x in range(num_x_samples):
            label = labels[y * num_x_samples + x]
            output_y_start = y * sample_height
            output_y_end = output_y_start + sample_height
            output_x_start = x * sample_width
            output_x_end = output_x_start + sample_width
            output_img[output_y_start:output_y_end, output_x_start:output_x_end] = data[
                y * num_x_samples + x
            ]
            print(CHAR_LOOKUP[label], end="")
        print()

    Image.fromarray(output_img.numpy()).save(output_path)


if __name__ == "__main__":
    main(**vars(parse_args()))
