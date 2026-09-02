import torch
from torch import nn

from src.multihead import MultiHeadAttention
from src.layernorm import LayerNorm
from src.feedforward import FeedForwardNetwork

class TransformerBlock(nn.Module):
    def __init__(self, din, context_length, n_heads, expanding_factor, multihead_dropout, dropout):
        super().__init__()

        self.norm1 = LayerNorm(din)
        self.multiheadattention = MultiHeadAttention(din, din, n_heads, multihead_dropout, context_length)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = LayerNorm(din)
        self.fnn = FeedForwardNetwork(din, expanding_factor)


    def forward(self, X):

        shortcut = X
        X = self.norm1(X)
        X = self.multiheadattention(X)
        X = self.dropout(X)

        X = X + shortcut

        shortcut = X

        X = self.norm2(X)
        X = self.fnn(X)
        X = self.dropout(X)
        X = X + shortcut

        return X
