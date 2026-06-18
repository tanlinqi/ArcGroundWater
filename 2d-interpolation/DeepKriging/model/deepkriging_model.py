"""Neural network components for DeepKriging."""

import torch.nn as nn


class DeepKrigingMLP(nn.Module):
    def __init__(self, input_dim, hidden_units, hidden_layers):
        super(DeepKrigingMLP, self).__init__()
        layers = []
        current = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current, hidden_units))
            layers.append(nn.ReLU())
            current = hidden_units
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features)

