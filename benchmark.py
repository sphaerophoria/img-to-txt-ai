#!/usr/bin/env python3

from lib.img_to_text import generate_glyph_cache
from lib.img_sampler import (
    NestedDirImageLoader,
)
from lib.load_network import (
    load_model_params,
    load_network_from_model_params_and_weights,
)

from argparse import ArgumentParser

import torch


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--validation-data", dest="validation_data_path", required=True)
    parser.add_argument("--model-params", dest="model_params_path", required=True)
    parser.add_argument("--weights", dest="weights_path", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main(validation_data_path, model_params_path, weights_path, device):
    model_params = load_model_params(model_params_path)

    sample_width = model_params["sample_width"]
    sample_height = model_params["sample_height"]
    char_codes = model_params["char_map"]
    _, glyph_cache = generate_glyph_cache(char_codes, device)

    sampler = NestedDirImageLoader(
        validation_data_path, sample_width, sample_height, glyph_cache, device=device
    )
    net = load_network_from_model_params_and_weights(model_params, weights_path, device)

    matched = 0
    total = 0
    with torch.no_grad():
        for i in range(sampler.num_images()):
            data, labels = sampler.get_sample(i)
            data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
            output = net(data)
            predicted = torch.max(output, 1)[1]
            matched += (predicted == labels).sum()
            total += predicted.shape[0]
            print("percentage correct: {}%".format(matched / total * 100.0))


if __name__ == "__main__":
    main(**vars(parse_args()))
