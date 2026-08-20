import sys
from pathlib import Path

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, din, dout, n_heads, dropout, context_length, qkv_bias = False):
        super().__init__()

        assert (dout % n_heads == 0), 'Number of heads has to be divisible by the output dimensions'

        self.dropout = nn.Dropout(dropout)
        self.head_dims = dout // n_heads

        # we create the 3 weight matrices the Wq, Wk, Wv
        self.Wq = nn.Linear(din, dout, bias = qkv_bias)
        self.Wk = nn.Linear(din, dout, bias = qkv_bias)
        self.Wv = nn.Linear(din, dout, bias = qkv_bias)

        # will be used for reshaping
        self.n_heads = n_heads
        self.dout = dout

        # we create the mask
        self.register_buffer('mask',
                            torch.triu(torch.ones(context_length, context_length),
                                        diagonal = 1))   # 1 above the main diagonal

        self.output_proj = nn.Linear(dout, dout)

    def forward(self, X):
        n_batches, n_tokens, _ = X.shape

        # now we create the k q v 
        key = self.Wk(X)
        query = self.Wq(X)
        value = self.Wv(X)

        # now each of these tensors have a shape of [n_batches, n_tokens, dout]
        # we want that dout to be instead n_heads, head_dims

        key = key.view(n_batches, n_tokens, self.n_heads, self.head_dims)
        query = query.view(n_batches, n_tokens, self.n_heads, self.head_dims)
        value = value.view(n_batches, n_tokens, self.n_heads, self.head_dims)

        # now we want the n_heads to be before the n_tokens
        # [n_batches, n_heads, n_tokens, head_dims]     # because that would make each head represent a single independant matrix
        # full of tokens as rows where each column represent a dout dimension of that head

        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)

        # now we start the full process
        # attention scores W
        attention_scores = query @ key.transpose(-1, -2)

        # we apply the causal mask to the attention_scores
        attention_scores.masked_fill_(self.mask.bool()[:n_tokens, :n_tokens], -torch.inf)   # type: ignore

        # now attention weight A
        attention_weights = torch.softmax(attention_scores / key.shape[-1] ** 0.5, dim = -1)

        # next we apply dropout to the attention weights
        attention_weights = self.dropout(attention_weights)
        
        # now we finally calculate the context vector Z
        context_vectors = attention_weights @ value

        # now we turn them back to [n_batches, n_tokens, dout]
        context_vectors = context_vectors.transpose(1, 2)
        context_vectors = context_vectors.contiguous().view(n_batches, n_tokens, self.dout)

        # finally we pass them onto a final linear layer to capture richer combinations between the inputs
        context_vectors = self.output_proj(context_vectors)

        return context_vectors