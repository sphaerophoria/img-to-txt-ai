import torch.nn as nn


class Network(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        return self.linear1(x)
