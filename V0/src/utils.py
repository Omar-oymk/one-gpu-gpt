from pathlib import Path

import torch
from torch import nn
def print_model_size(model):
    total_params = sum(param.numel() for param in model.parameters())

    model_size_MB = (total_params * 32 / 4) / (1024 * 1024) 
    model_size_GB = model_size_MB / 1024

    print(f"The model has {total_params:,} total parameters")
    print(f'The model is {model_size_MB:.2f}MB')
    print(f'Approximately {model_size_GB:.2f}GB')


def load_text_data(path = Path().cwd().parents[1] / 'data' / 'train_data.txt'):
    with open(path, 'r', encoding = 'utf-8') as f:
        raw_text = f.read()

    return raw_text

def generate_text_tokIDs(model, tokenIDs, max_new_tokens, context_size, tokenizer):
    for _ in range(max_new_tokens):
        context_idx = tokenIDs[:, -context_size:]

        with torch.no_grad():
            logits = model(context_idx)

        logits = logits[:, -1, :]   # take only the last token
        probas = torch.softmax(logits, dim = -1)
        next_tokID = probas.argmax(dim = -1, keepdim=True)
        if (next_tokID == tokenizer.eot_token).any():
            break
        tokenIDs = torch.cat((tokenIDs, next_tokID), dim = 1)

    return tokenIDs

def text_to_tokenIDs(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
    encoded = torch.tensor(encoded).unsqueeze(0)

    return encoded

def tokenIDs_to_text(tokIDs, tokenizer):
    decoded = tokenizer.decode(list(tokIDs.squeeze(0)))

    return decoded