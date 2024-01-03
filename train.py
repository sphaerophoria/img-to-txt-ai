#!/usr/bin/env python3

from lib.img_sampler import (
    NestedDirImageLoader,
)
from lib.img_to_text import generate_glyph_cache
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--training-data", dest="training_data_path", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


class Network(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        return self.linear1(x)


def main(training_data_path, device):
    _, glyph_cache = generate_glyph_cache(device)
    sample_width = 12
    sample_height = int(sample_width * glyph_cache.shape[1] / glyph_cache.shape[2])
    sampler = NestedDirImageLoader(
        training_data_path, sample_width, sample_height, glyph_cache, device=device
    )
    net = Network(sample_width * sample_height, glyph_cache.shape[0]).to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    i = 0

    while True:
        optimizer.zero_grad()

        data, labels = sampler.get_samples(4)
        data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
        data = data.to(torch.float32)
        output = net(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        i += 1
        if i % 50 == 1:
            print("loss", loss)


if __name__ == "__main__":
    main(**vars(parse_args()))
