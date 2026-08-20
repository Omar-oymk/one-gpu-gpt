import random
from pathlib import Path

from tiktoken import get_encoding   # will be used to count the number of tokens as we load the data
from tqdm import tqdm   # to create a loading animation to see
from datasets import load_dataset   # hugging face datasets

#### configurations
path_to_data = Path().cwd() / 'data'        # ./data
output_file = path_to_data / 'train_data.txt'

# MODIFY THIS DICTIONARY TO DECIDE ON WHAT N TOKENS U WANT TO HAVE FROM EACH SOURCE
# IN MY CASE ILL START WITH 
targets = {
    'fineweb': 60_000_000,
    'wikipedia': 30_000_000,
    'tinystories': 10_000_000
}

random.seed(42)

tokenizer = get_encoding('gpt2')

