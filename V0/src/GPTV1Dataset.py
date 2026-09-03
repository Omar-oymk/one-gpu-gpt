import sys
from pathlib import Path
from typing import Any

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import Dataset

# the approach here instead is to make the init of the dataset less expensive
# and handle it through another way which is to make it 
# tokenize the whole dataset and save it as a list
# then instead of creating a huge tensor containing all inputs and all targets
# we would instead be lazy about it we would make an access on demand only
# that would calculate based on the index and then return the tokenized list when u need it
# instead of initializing it all
# this creates the kind of "access on the go" kind of thing where it only operates on the lists that u need now
class GPTV1Dataset(Dataset):
    def __init__(self, text, max_window_length, tokenizer, stride):
        self.tokenized_text = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
        self.max_window_length = max_window_length
        self.stride = stride

        # and as u can see we got rid of the huge initialization and extra lists

    def __getitem__(self, index):
        # the idea is here
        # u need to calculate the subset that u would need based on the index
        # since u know for example the first sequence would have a 0 --> max_window_length
        # and for the next iteration like the second sequence would move a stride
        # so if we say stride = 256
        # then the next start would be 256 --> max_window_length + 256
        # and so on
        start = index * self.stride

        input_tokens = self.tokenized_text[start:start + self.max_window_length]
        target_tokens = self.tokenized_text[start + 1: start + self.max_window_length + 1]

        return torch.tensor(input_tokens), torch.tensor(target_tokens)

    def __len__(self):
        # now the confusing part u would ask urself
        # then if i dont have the full list of tokens how would i get its length
        # the length as u can remember should give u how many sequences is in that list and should be the same
        # for input and target lists
        # so instead of counting it fr
        # we will instead return a list of range() that would "mimic" the behaviour of a real length of an existing list
        return range(0,
                     len(self.tokenized_text) - self.max_window_length,
                     self.stride)