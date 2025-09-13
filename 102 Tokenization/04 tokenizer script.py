import datetime
from multiprocess import Process, Queue
import os 

import argparse
from datasets import load_dataset
from pathlib import Path
from transformers import PreTrainedTokenizerFast
import torch
from torch.utils.data import DataLoader

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parallel tokenization script \
                                                   for datasets')
    
    # Dataset arguments
    parser.add_argument('--dataset-name', 
                        type=str, default='TucanoBR/wikipedia-PT',
                        help='Dataset name to load (default: \
                              TucanoBR/wikipedia-PT)')
    
    parser.add_argument('--dataset-split', 
                        type=str, default='train[:10]',
                        help='Dataset split to use (default: train[:10])')
    
    parser.add_argument('--text-column', type=str, default='text',
                        help='Column name containing text data \
                             (default: text)')
    
    parser.add_argument('--cache-dir', type=str, default='../data',
                        help='Directory to cache dataset \
                              (default: ../data)')
    
    # Tokenizer arguments
    parser.add_argument('--tokenizer-path', type=str, 
                        default='../models/30k/',
                        help='Path to tokenizer model \
                            (default: ../models/30k/)')
    
    parser.add_argument('--max-length', type=int, 
                        default=512,
                        help='Maximum sequence length for tokenization \
                              (default: 512)')
    
    # Processing arguments
    parser.add_argument('--batch-size', type=int, 
                        default=100000,
                        help='Batch size for processing (default: 100000)')
    
    parser.add_argument('--num-workers', type=int, 
                        default=4,
                        help='Number of worker processes (default: 4)')
    
    # Output arguments
    parser.add_argument('--output-path', type=str, 
                        default='../data/tokenized/',
                        help='Output directory for tokenized data \
                             (default: ../data/tokenized/)')
    
    return parser.parse_args()

def import_data(dataset_name:str='TucanoBR/GigaVerbo', 
                dataset_split:str='train[:10]',
                text_column:str='text',
                cache_dir:str='../data',
                batch_size:int=100000,
                ):
    
    dataset = load_dataset(dataset_name, 
                           split=dataset_split, 
                           cache_dir=cache_dir)
    print(f"{datetime.datetime.now()}: Dataset {dataset_name} loaded with {len(dataset)} samples")
    
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
                      tokenizer_path: str = '../models/30k/', 
                      max_length: int = 512
                  ) -> None:
    
    # Each worker loads its own tokenizer to avoid sharing issues
    worker_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    if worker_tokenizer.pad_token is None:
        worker_tokenizer.pad_token = worker_tokenizer.eos_token
    
    print(f"{datetime.datetime.now()}: Worker {worker_id} started")

    def tokenize_batch(text_batch):
        tokenized = worker_tokenizer(
            text_batch,
            padding='max_length',  # Always pad to max_length
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return tokenized
    
    def save_batch(batch_encoding, 
                   output_path: str, 
                   batch_id: int, 
                   total_batches: int):
        """
        Save BatchEncoding as PyTorch tensors - most efficient for PyTorch usage
        """
        # Ensure output directory exists
        Path(output_path).mkdir(parents=True, exist_ok=True)
    
        # Save as .pt file
        filename = f"tokenized_{batch_id + 1}_of_{total_batches}.pt"
        filepath = os.path.join(output_path, filename)
    
        # Convert to CPU tensors and save
        cpu_batch_encoding = {key: tensor.cpu() for key, tensor in batch_encoding.items()}
        torch.save(cpu_batch_encoding, filepath)
    
        print(f"{datetime.datetime.now()}: Saved batch {batch_id} from worker {worker_id} to {filepath}")
    
    while True:
        queue_item = input_queue.get()
    
        if queue_item is None:  # Sentinel value to signal termination
            print(f"{datetime.datetime.now()}: Worker {worker_id} terminating")
            break
        batch_id, text_batch = queue_item
        print(f"{datetime.datetime.now()}: Worker {worker_id} processing batch of size {len(text_batch)}")
        tokenized_output = tokenize_batch(text_batch)
        save_batch(tokenized_output, 
                   output_path, 
                   batch_id=batch_id, 
                   total_batches=total_batches)


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    input_queue = Queue()
    print(f'{datetime.datetime.now()}: Input queue created')

    dataLoader = import_data(dataset_name=args.dataset_name,
                             dataset_split=args.dataset_split,
                             text_column=args.text_column,
                             cache_dir=args.cache_dir,
                             batch_size=args.batch_size
                             )
    total_batches = len(dataLoader)

    for batch_id, batch_text in enumerate(dataLoader):
        print(f"{datetime.datetime.now()}: Enqueuing batch {batch_id} of size {len(batch_text)}")
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
    print(f"{datetime.datetime.now()}: All workers completed successfully!")

# how to run the script:
# python "04 tokenizer script.py" --dataset-name "TucanoBR/tucano-sft" 
#        --dataset-split "train[:100]" --tokenizer-path "../models/30k/" 
#        --max-length 1024 --batch-size 25000 --num-workers 8 
#        --output-path "../data/custom_tokenized/"