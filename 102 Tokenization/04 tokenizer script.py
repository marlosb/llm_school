import datetime
from multiprocess import Process, Queue
import os 

from datasets import load_dataset
import h5py
import numpy as np
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader
import torch

from _utils import parse_arguments

def now() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def import_data(dataset_name:str='TucanoBR/GigaVerbo', 
                dataset_split:str='train[:10]',
                text_column:str='text',
                cache_dir:str='../data',
                batch_size:int=100000,
                ):
    
    dataset = load_dataset(dataset_name, 
                           split=dataset_split, 
                           cache_dir=cache_dir)
    print(f"{now()}: Dataset {dataset_name} loaded with {len(dataset)} samples")
    
    # Create collate function
    def collate_fn(batch):
        return [item[text_column] for item in batch]
    
    return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=collate_fn)

def parallel_tokenize(worker_id: int,
                      input_queue:Queue, 
                      output_path: str,
                      total_batches: int,
                      tokenizer_path: str = '../models/30k.json', 
                      max_length: int = 512
                  ) -> None:
    
    # Each worker loads its own tokenizer to avoid sharing issues
    worker_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    if worker_tokenizer.pad_token is None:
        worker_tokenizer.pad_token = worker_tokenizer.eos_token
    
    print(f"{now()}: Worker {worker_id} started")

    def tokenize_batch(text_batch):
        tokenized = worker_tokenizer(
            text_batch,
            padding="max_length",  # Enable padding to max_length
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,  # Need attention masks when padding
            return_tensors="np"
        )
        return tokenized['input_ids'], tokenized['attention_mask']
    
    def save_batch(token_sequences,
                   attention_masks, 
                   output_path: str, 
                   batch_id: int, 
                   total_batches: int):
        """
        Save BatchEncoding as NPZ file - efficient for NumPy arrays
        """
        # Ensure output directory exists
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Convert to int16 to save space (30K vocab fits in 16 bits)
        tokens_int16 = np.clip(token_sequences, 0, 65535).astype(np.int16)
        attention_bool = attention_masks.astype(bool)

        # Save as .npz file
        filename = f"tokenized_{batch_id + 1}_of_{total_batches}.npz"
        filepath = os.path.join(output_path, filename)

        np.savez_compressed(filepath, 
                           tokens=tokens_int16, 
                           attention_mask=attention_bool)
    
        file_size = os.path.getsize(filepath) / (1024**2)
        print(f"{now()}: worker {worker_id} saved batch {batch_id} to {filepath} ({file_size:.2f} MB)")

    while True:
        queue_item = input_queue.get()
    
        if queue_item is None:  # Sentinel value to signal termination
            print(f"{now()}: Worker {worker_id} terminating")
            break
        batch_id, text_batch = queue_item
        print(f"{now()}: Worker {worker_id} processing batch id {batch_id}")
        tokenized_sequences, attention_masks = tokenize_batch(text_batch)
        save_batch(tokenized_sequences, 
                   attention_masks,
                   output_path, 
                   batch_id=batch_id, 
                   total_batches=total_batches)


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    input_queue = Queue()
    print(f'{now()}: Input queue created')

    dataLoader = import_data(dataset_name=args.dataset_name,
                             dataset_split=args.dataset_split,
                             text_column=args.text_column,
                             cache_dir=args.cache_dir,
                             batch_size=args.batch_size
                             )
    total_batches = len(dataLoader)

    for batch_id, batch_text in enumerate(dataLoader):
        print(f"{now()}: Enqueuing batch {batch_id} of size {len(batch_text)}")
        input_queue.put((batch_id, batch_text))
    
    workers = []
    for worker_id in range(args.num_workers):
        # Add sentinel values (one per consumer) to signal exit
        input_queue.put(None)
        # Start worker processes
        p = Process(target=parallel_tokenize, args=(worker_id, 
                                                    input_queue, 
                                                    args.output_path,
                                                    total_batches,
                                                    args.tokenizer_path,
                                                    args.max_length))
        p.start()
        workers.append(p)

    # Wait for all workers to complete
    for worker in workers:
        worker.join()
    print(f"{now()}: All workers completed successfully!")

# how to run the script:
# python "04 tokenizer script.py" --dataset-name "TucanoBR/tucano-sft" 
#        --dataset-split "train[:100]" --tokenizer-path "../models/30k/" 
#        --max-length 1024 --batch-size 25000 --num-workers 8 
#        --output-path "../data/custom_tokenized/"