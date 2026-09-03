import sys
from pathlib import Path

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import Dataset
class GPTV0Dataset(Dataset):
    def __init__(self, text, max_window_length, tokenizer, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_window_length, stride):
            input_window = token_ids[i: i + max_window_length]
            target_window = token_ids[i + 1: i + max_window_length + 1]

            self.input_ids.append(torch.tensor(input_window))
            self.target_ids.append(torch.tensor(target_window))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]