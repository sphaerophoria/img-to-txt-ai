#!/usr/bin/env python3

from argparse import ArgumentParser
from lib.img_to_text import GlyphRenderer
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
    sample_height_comparison = sampler.img_for_comparison.shape[0] / num_y_samples
    sample_width_comparison = sampler.img_for_comparison.shape[1] / num_x_samples

    output_img = torch.zeros_like(sampler.img_for_comparison)

    char_lookup = [chr(x) for x in renderer.glyph_cache.keys() if isinstance(x, int)]
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

    Image.fromarray(output_img.numpy() * 255.0).convert(mode="L").save(output_path)


if __name__ == "__main__":
    main(**vars(parse_args()))
