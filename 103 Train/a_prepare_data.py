import datetime
import os

import argparse
from datasets import load_dataset
import numpy as np
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Tokenization script arguments')
    
    parser.add_argument('--dataset-name', type=str, default='TucanoBR/wikipedia-PT')
    
    parser.add_argument('--cache-dir', type=str, default='../data')

    parser.add_argument('--output-dir', type=str, default='../data/tokenized')

    parser.add_argument('--tokenizer-path', type=str, default='../models/30k/')

    parser.add_argument('--num-proc', type=int, default=8)

    return parser.parse_args()

args = parse_arguments()

def now() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# copy from Andrej Karpathy's nanoGPT:
# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = args.num_proc

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

# load tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
print(f'{now()}: tokenizer loaded from {args.tokenizer_path}')


if __name__ == "__main__":
    # load dataset
    dataset = load_dataset(args.dataset_name,
                           cache_dir=args.cache_dir,
                           num_proc=num_proc_load_dataset)
    print(f'{now()}: dataset {args.dataset_name} loaded')

    # split train and validation
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val
    print(f'{now()}: dataset split into train and val')

    def process(example):
        ids = tokenizer.encode(example['text'], add_special_tokens=False) # encode_ordinary ignores any special tokens
        ids.append(tokenizer.eos_token_id) # add the end of text token
        out = {'ids': ids, 'len': len(ids)}
        return out

        # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    print(f'{now()}: dataset tokenized')

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        os.makedirs(args.output_dir, exist_ok=True)
        filename = os.path.join(args.output_dir, f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 30000 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 64

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
