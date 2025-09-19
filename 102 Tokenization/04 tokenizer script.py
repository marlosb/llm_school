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

    def tokenize_batch(text_batch, max_length):
        all_input_ids = []
        all_attention_masks = []

        # Process each text individually
        for text in text_batch:
            # First, tokenize WITHOUT the post-processor to avoid automatic EOS
            # We'll add EOS manually only to the last chunk
            tokenized = worker_tokenizer(
                text,
                padding=False,  # Don't pad yet
                truncation=True,
                max_length=max_length,
                stride=60,
                return_attention_mask=True,
                return_overflowing_tokens=True,
                return_tensors=None,
                add_special_tokens=False
            )

            # Extract the lists - these are guaranteed to be Python lists now
            input_ids_list = tokenized['input_ids']
            attention_mask_list = tokenized['attention_mask']
            
            # Handle single vs multiple chunks
            if isinstance(input_ids_list[0], int):
                # Single chunk case
                input_ids_list = [input_ids_list]
                attention_mask_list = [attention_mask_list]
                
            # Process chunks and add EOS only to the last one (INDENTED CORRECTLY!)
            for chunk_idx, (chunk_ids, chunk_mask) in enumerate(zip(input_ids_list, attention_mask_list)):
                is_last_chunk = (chunk_idx == len(input_ids_list) - 1)
                
                # Add EOS only to the last chunk
                if is_last_chunk:
                    chunk_ids = chunk_ids + [worker_tokenizer.eos_token_id]
                    chunk_mask = chunk_mask + [1]  # EOS is a valid token
                
                # Now pad to max_length
                padding_length = max_length - len(chunk_ids)
                if padding_length > 0:
                    chunk_ids = chunk_ids + [worker_tokenizer.pad_token_id] * padding_length
                    chunk_mask = chunk_mask + [0] * padding_length
                elif padding_length < 0:
                    # Truncate if somehow too long
                    chunk_ids = chunk_ids[:max_length]
                    chunk_mask = chunk_mask[:max_length]
                
                all_input_ids.append(chunk_ids)
                all_attention_masks.append(chunk_mask)

        # Convert to numpy arrays with efficient dtypes
        all_input_ids = np.array(all_input_ids, dtype=np.uint16)
        all_attention_masks = np.array(all_attention_masks, dtype=np.bool_)
            
        print_text = (f"{now()}: Worker {worker_id}: created ",
                    f"{len(all_input_ids)} chunks from {len(text_batch)}",
                    f" original texts")
        print(''.join(print_text))

        return all_input_ids, all_attention_masks
    
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

        # Save as .npz file
        filename = f"tokenized_{batch_id + 1}_of_{total_batches}.npz"
        filepath = os.path.join(output_path, filename)

        np.savez_compressed(filepath, 
                           tokens=token_sequences, 
                           attention_mask=attention_masks)
    
        file_size = os.path.getsize(filepath) / (1024**2)
        print_text = (f"{now()}: Worker {worker_id}: saved batch {batch_id} ",
                      f"to {filepath} ({file_size:.2f} MB)")
        print(''.join(print_text))

    while True:
        queue_item = input_queue.get()
    
        if queue_item is None:  # Sentinel value to signal termination
            print(f"{now()}: Worker {worker_id} terminating")
            break
        batch_id, text_batch = queue_item
        print(f"{now()}: Worker {worker_id} processing batch id {batch_id}")
        tokenized_sequences, attention_masks = tokenize_batch(text_batch, 
                                                              max_length)
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