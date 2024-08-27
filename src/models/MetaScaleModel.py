import torch 
import torch.nn as nn 
import numpy as np


class MetaScaleModel(nn.Module):
    def __init__(self, input_dim: int):
        super(MetaScaleModel, self).__init__()
        self.architecture = [512, 512, 64, 1]
        self.layers = nn.ModuleList()
        self.input_dim = input_dim

        current_dim = self.input_dim
        for i, layer in enumerate(self.architecture):
            next_dim = layer
            if i == len(self.architecture) - 1:
                self.layers.append(
                    nn.Sequential(nn.Linear(current_dim, next_dim), nn.Softplus())
                )
            else:
                self.layers.append(
                    nn.Sequential(nn.Linear(current_dim, next_dim), nn.Softplus())
                )
                current_dim = next_dim

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x