import torch
from torch import nn

class GELU(nn.Module):    
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return 0.5 * X * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (X + 0.044715 * torch.pow(X, 3))))