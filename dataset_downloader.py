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

def collect_dataset(dataset, target_tok_count, dataset_name, text_field = 'text'):
    """
    Takes in the dataset from huggingface, loads the field based on text_field
    then uses tokenizer to count how many tokens and stops loading when tokens >= target_tok_count
    dataset_name: used to name the dataset can be anything but it will be the name written in the textfile
    """

    documents = []
    curr_tok_count = 0

    progress = tqdm(        # create a loading bar 
        total = target_tok_count,
        desc = f'{dataset_name}',
        unit = 'tokens'
    )

    for example in dataset:         # for each row in the dataset
        text = example.get(text_field).strip()  # fetch the text there

        if not text:
            continue

        ntokens = len(tokenizer.encode_ordinary(text))       # this counts ntokens if it were to be used with gpt2 tokenizer

        if ntokens == 0:     # check if no tokens were fetched then skip
            continue

        remaining_ntokens = target_tok_count - curr_tok_count

        if ntokens >= remaining_ntokens:
            # We need only part of this document.
            token_ids = tokenizer.encode_ordinary(text)     # encode it to know n tokens 
            token_ids = token_ids[:remaining_ntokens]   # take only the first n_tokens till the remaining

            text = tokenizer.decode(token_ids)  # decode it back to text

            documents.append(text)      # append it to the documents list
            curr_tok_count += remaining_ntokens     

            progress.update(remaining_ntokens)
            break
        else:
            documents.append(text)      # just append doesnt matter because it is not more than remaining tokens

            curr_tok_count += ntokens

            progress.update(ntokens)        # add to the loading 

        progress.close()
        print(f'{dataset_name}: Current Token Count: {curr_tok_count} tokens')

        return documents

def main():

    path_to_data.mkdir(exist_ok=True)

    #region load datasets
    fineweb = load_dataset(     # load the 10b token dataset
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    # then only load the 60m tokens
    fineweb_docs = collect_dataset(
        fineweb,
        target_tok_count=targets['fineweb'],
        dataset_name="FineWeb",
    )

    wikipedia = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    wikipedia_docs = collect_dataset(
        wikipedia,
        target_tok_count=targets["wikipedia"],
        dataset_name="Wikipedia",
    )

    tinystories = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True,
    )

    tinystories_docs = collect_dataset(
        tinystories,
        target_tok_count=targets["tinystories"],
        dataset_name="TinyStories",
    )
    #endregion
    
    documents = fineweb_docs + wikipedia_docs + tinystories_docs \
    if fineweb_docs and wikipedia_docs and tinystories_docs is not None else print("Nothing got fetched")

    # now shuffle them 
    random.shuffle(documents)   # type:ignore

    with open(output_file, 'w') as f:

        for document in tqdm(documents, desc = 'Writing in the text file'):
            f.write(document)
            f.write('\n\n')


if __name__ == '__main__':
    main()