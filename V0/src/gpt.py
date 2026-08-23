import torch
from torch import nn

from transformer import TransformerBlock
from layernorm import LayerNorm


class GPT8TModel(nn.Module):
    def __init__(self, emb_dim, vocab_size, context_length, n_layers, n_heads, emb_dropout, mha_dropout, dropout, expanding_factor = 4,
                 ):
        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)

        self.embeds_dropout = nn.Dropout(emb_dropout)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(emb_dim, context_length, n_heads, expanding_factor, mha_dropout, dropout) for _ in range(n_layers)]
        )

        self.final_layernorm = LayerNorm(emb_dim)

        self.output_projection = nn.Linear(emb_dim, vocab_size, bias = False)

        # we use weight tying where input weights == output weights to act as a sort of regularization + will offer 
        # a great computational efficiencty due to less number of weights
        self.output_projection.weight = self.tok_emb.weight

    def forward(self, X):
        n_batches, sequences = X.shape

        tok_embeds = self.tok_emb(X)
        pos_embeds = self.pos_emb(torch.arange(sequences, device = X.device))

        input_embeds = tok_embeds + pos_embeds

        # next up is applying the dropout
        input_embeds = self.embeds_dropout(input_embeds)

        # now we feed it to the trf blocks

        context_vectors = self.trf_blocks(input_embeds)

        context_vectors = self.final_layernorm(context_vectors)

        return self.output_projection(context_vectors)