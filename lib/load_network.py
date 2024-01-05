from .network import Network

import json
import torch

from os import PathLike
from typing import Dict


def load_model_params(model_params_path: PathLike) -> Dict:
    with open(model_params_path) as f:
        model_params = json.load(f)

    return model_params


def load_network_from_model_params_and_weights(
    model_params: Dict, weights_path: PathLike, device="cpu"
) -> Network:
    sample_width = model_params["sample_width"]
    sample_height = model_params["sample_height"]
    net = Network(sample_width * sample_height, len(model_params["char_map"])).to(
        device
    )
    net.load_state_dict(torch.load(weights_path))
    return net
