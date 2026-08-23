import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, din):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(din))
        self.beta = nn.Parameter(torch.zeros(din))

    def forward(self, X):
        mean = X.mean(dim = -1, keepdim = True)
        var = X.var(dim = -1, keepdim = True, unbiased = False)  # this prevents the bias towards the /n or /n-1 (bessels correction)
        std = torch.sqrt(var + 1e-5)  # add a small value to prevent division by zero

        norm_x = (X - mean) / std

        return self.gamma * norm_x + self.beta