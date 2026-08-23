import torch
from torch import nn

from gelu import GELU

class FeedForwardNetwork(nn.Module):
    def __init__(self, din, expanding_factor = 4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(din, expanding_factor * din),
            GELU(),
            nn.Linear(expanding_factor * din, din)
        )

    def forward(self, X):
        return self.layers(X)