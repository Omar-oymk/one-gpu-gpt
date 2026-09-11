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


# the greedy decoding approach where u just take the tokenid with highest prob 
# the problem with it is it is determministic and overall non humane
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

# now the idea the updated version with better decoding is
# firstly we use the multinomial instead of argmax
# what it does is it generates a sample using that probabilitiy
# it is like instead of picking the one with highest probability directly
# u throw a die knowing that for example there is a 60% chance of getting a 6 for example 
# another thing to do is to use temperature scaling
# what it does is it kinda modifies the outputs of the softmax by dividing the logits by a constant "T"
# this + multinomial will help make the more common sense words or similar meanings aka for example the top predictions
# to have a near accuracy so that the multinomial can kind of find different and generate different words
# lastly we can use the top - k sampling to choose which top k values to keep and remove the rest with -inf
# the idea here is to reduce the risk of generating an incoherent irrelivent word due to multinomial + temperature scaling
def generate_text_tokIDs_advanced(model, idx, device, eos_id, max_new_tokens, context_size, temperature = 0.0, topk = None):
    idx = idx.to(device)

    for i in range(max_new_tokens):
        idx = idx[:, -context_size:]

        next_token_id = model(idx)[:, -1, :]

        if topk is not None:
            top_logits, top_indices = torch.topk(next_token_id, topk, dim = -1)
            # then we need to know the threshold to do so we get the least logit in the top_logits
            min_logit = top_logits[:, -1]       # all batches, last logit

            next_token_id = torch.where(
                # condition here 
                next_token_id < min_logit,
                # what to replace them with
                torch.tensor(float('-inf')).to(device),
                next_token_id
            )

        # if there is a temperature
        if temperature > 0.0:
            next_token_id /= temperature
            probs = torch.softmax(next_token_id, dim = -1)
            tok_id = torch.multinomial(probs, num_samples= 1)

        else:
            tok_id = torch.argmax(probs, dim = -1, keepdim= True)


        if tok_id == eos_id:
            break

        idx = torch.cat((idx, tok_id), dim = -1)

    return idx


def text_to_tokenIDs(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
    encoded = torch.tensor(encoded).unsqueeze(0)

    return encoded

def tokenIDs_to_text(tokIDs, tokenizer):
    decoded = tokenizer.decode(list(tokIDs.squeeze(0)))

    return decoded

def count_tokens(text_data, tokenizer):
    tokens = tokenizer.encode(text_data, allowed_special = {"<|endoftext|>"})
    return len(tokens)