import sys
from pathlib import Path

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from src.GPTV1Dataset import GPTV1Dataset

# now we define a function that creates the dataloader for the dataset
def create_dataloaderV1(text, max_window_length, tokenizer, stride, batch_size, shuffle, drop_last, num_workers, pin_memory):
    dataset = GPTV1Dataset(text, max_window_length, tokenizer, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, pin_memory=pin_memory)
    return dataloader