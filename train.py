#!/usr/bin/env python3

from lib.img_sampler import (
    NestedDirImageLoader,
)
from lib.network import Network
from lib.img_to_text import generate_glyph_cache
from argparse import ArgumentParser
from pathlib import Path

import json
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--training-data", dest="training_data_path", required=True)
    parser.add_argument("--output", dest="output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main(training_data_path, output_dir, device):
    output_dir = Path(output_dir)
    if output_dir.exists() and output_dir.is_dir():
        if len(list(output_dir.iterdir())) != 0:
            raise RuntimeError("output directory is not empty")

    try:
        output_dir.mkdir(parents=True)
    except RuntimeError as e:
        raise RuntimeError("Failed to create output dir") from e

    char_codes, glyph_cache = generate_glyph_cache(device=device)
    sample_width = 12
    sample_height = int(sample_width * glyph_cache.shape[1] / glyph_cache.shape[2])
    sampler = NestedDirImageLoader(
        training_data_path, sample_width, sample_height, glyph_cache, device=device
    )
    net = Network(sample_width * sample_height, glyph_cache.shape[0]).to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.005)

    model_params = {
        "sample_width": sample_width,
        "sample_height": sample_height,
        "char_map": char_codes,
    }

    with open(output_dir / "model_params.json", "w") as f:
        json.dump(model_params, f)

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
        print("step {}, loss: {}".format(i, loss), end="\r")
        if i % 50 == 1:
            print()
            torch.save(net.state_dict(), output_dir / "{}.state_dict".format(i))


if __name__ == "__main__":
    main(**vars(parse_args()))
