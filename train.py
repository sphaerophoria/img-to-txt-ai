#!/usr/bin/env python3

from lib.img_to_text import GlyphRenderer, CHAR_LOOKUP
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


def render_full_image(img, renderer, sample_width, sample_height, iter, net):
    output_img = numpy.zeros(
        (
            int(img.shape[0] / sample_height * renderer.row_height()),
            int(
                img.shape[1]
                / sample_width
                * renderer.row_height()
                * renderer.char_aspect()
            ),
        ),
        dtype=numpy.uint8,
    )

    cursor_y = 0
    cursor_x = 0

    data = extract_w_h_samples(sample_width, sample_height, img)
    predictions = net(
        data.reshape((data.shape[0], sample_width * sample_height)).to(torch.float32)
    )
    num_y_samples, num_x_samples = num_samples_for_img(img, sample_width, sample_height)
    char_idxs = torch.max(predictions, 1)[1]
    for y in range(num_y_samples):
        for x in range(num_x_samples):
            char_idx = char_idxs[y * num_x_samples + x]

            rendered = renderer.render_char(CHAR_LOOKUP[char_idx])
            output_y_end = cursor_y + rendered.bitmap.shape[0]
            output_x_end = cursor_x + rendered.bitmap.shape[1]
            output_img[cursor_y:output_y_end, cursor_x:output_x_end] = rendered.bitmap
            cursor_x += int(rendered.advance)

        cursor_x = 0
        cursor_y += int(renderer.row_height())

    Image.fromarray(output_img).save("{}.png".format(iter))


def main(image_path, device):
    renderer = GlyphRenderer()
    sample_width = 12
    sample_height = int(sample_width / renderer.char_aspect())
    net = Network(sample_width * sample_height, len(CHAR_LOOKUP)).to(device)
    sampler = ImgSampler(sample_width, sample_height, renderer, image_path, device)

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
                sampler.img, renderer, sample_width, sample_height, i, net
            )


if __name__ == "__main__":
    main(**vars(parse_args()))
