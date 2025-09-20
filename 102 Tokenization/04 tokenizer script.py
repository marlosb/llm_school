import datetime
from multiprocess import Process, Queue
import os
import time
from typing import Iterator, List

from datasets import load_dataset
import numpy as np
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader

from _utils import parse_arguments

def now() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def stream_dataset(dataset_name: str = 'TucanoBR/GigaVerbo', 
                   dataset_split: str = 'train',
                   text_column: str = 'text',
                   cache_dir: str = '../data',
                   batch_size: int = 100000,
                   label_column: str = None,
                   label_filter_value: int = None,
                   max_samples: int = None,
                   ) -> Iterator[List[str]]:
    """
    Stream dataset in batches without loading everything into memory
    """
    
    # Load dataset in streaming mode
    dataset = load_dataset(dataset_name, 
                          split=dataset_split, 
                          cache_dir=cache_dir,
                          streaming=True)
    
    print(f"{now()}: Dataset {dataset_name} loaded in streaming mode")
    
    # Filter dataset by label if specified
    if label_column and label_filter_value is not None:
        dataset = dataset.filter(lambda example: example[label_column] == label_filter_value)
        print(f"{now()}: Applied label filter: {label_column}={label_filter_value}")

    batch = []
    total_processed = 0

    for sample in dataset:
        # Check if we've reached the maximum number of samples
        if max_samples and total_processed >= max_samples:
            print(f"{now()}: Reached maximum samples limit: {max_samples}")
            break
            
        batch.append(sample[text_column])
        total_processed += 1
        
        # Yield batch when it reaches the specified size
        if len(batch) >= batch_size:
            print(f"{now()}: Yielding batch of {len(batch)} samples (total processed: {total_processed})")
            yield batch
            batch = []
    
    # Yield remaining samples if any
    if batch:
        print(f"{now()}: Yielding final batch of {len(batch)} samples (total processed: {total_processed})")
        yield batch    

def parallel_tokenize(worker_id: int,
                      input_queue:Queue, 
                      output_path: str,
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
                
            # Process chunks and add EOS only to the last one
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
                   batch_id: int):
        """
        Save BatchEncoding as NPZ file - efficient for NumPy arrays
        """
        # Ensure output directory exists
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Save as .npz file
        filename = f"tokenized_batch_{batch_id}.npz"
        filepath = os.path.join(output_path, filename)

        np.savez_compressed(filepath, 
                           tokens=token_sequences, 
                           attention_mask=attention_masks)
    
        file_size = os.path.getsize(filepath) / (1024**2)
        print_text = (f"{now()}: Worker {worker_id}: saved batch {batch_id} ",
                      f"to {filepath} ({file_size:.2f} MB)")
        print(''.join(print_text))

    batch_count = 0
    while True:
        queue_item = input_queue.get()
    
        if queue_item is None:  # Sentinel value to signal termination
            print(f"{now()}: Worker {worker_id} terminating after processing {batch_count} batches")
            break

        batch_id, text_batch = queue_item
        print(f"{now()}: Worker {worker_id} processing batch id {batch_id}")
        tokenized_sequences, attention_masks = tokenize_batch(text_batch, 
                                                              max_length)
        save_batch(tokenized_sequences, 
                   attention_masks,
                   output_path, 
                   batch_id=batch_id)
        batch_count += 1

def producer_process(dataset_name: str,
                    dataset_split: str,
                    text_column: str,
                    cache_dir: str,
                    batch_size: int,
                    label_column: str,
                    label_filter_value: int,
                    max_samples: int,
                    input_queue: Queue, 
                    max_queue_size: int,
                    num_workers: int) -> None:
    """
    Producer process that creates dataset stream and feeds batches to the queue
    """
    print(f"{now()}: Producer process started")
    
    # Create dataset stream within the producer process
    dataset_stream = stream_dataset(
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        text_column=text_column,
        cache_dir=cache_dir,
        batch_size=batch_size,
        label_column=label_column,
        label_filter_value=label_filter_value,
        max_samples=max_samples
    )
    
    batch_id = 0
    
    try:
        for batch in dataset_stream:
            # Wait if queue is full (non-blocking approach)
            while input_queue.qsize() >= max_queue_size:
                print(f"{now()}: Queue full ({input_queue.qsize()}/{max_queue_size}), waiting...")
                time.sleep(0.1)
            
            print(f"{now()}: Adding batch {batch_id} to queue (queue size: {input_queue.qsize()})")
            input_queue.put((batch_id, batch))
            batch_id += 1
    
    except Exception as e:
        print(f"{now()}: Error in producer process: {e}")
        raise
    
    finally:
        # Add sentinel values to signal workers to terminate
        print(f"{now()}: Finished streaming data. Adding sentinel values for {num_workers} workers")
        for _ in range(num_workers):
            input_queue.put(None)
        
        print(f"{now()}: Producer finished. Total batches: {batch_id}")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Create bounded queue - size should be manageable but allow some buffering
    max_queue_size = args.num_workers * 2  # Allow 2 batches per worker in queue
    input_queue = Queue(maxsize=max_queue_size)
    print(f'{now()}: Input queue created with max size: {max_queue_size}')

    label_column = getattr(args, 'label_column', None)
    label_filter_value = getattr(args, 'label_filter_value', None)
    max_samples = getattr(args, 'max_samples', None)

    # Start worker processes
    workers = []
    for worker_id in range(args.num_workers):
        p = Process(target=parallel_tokenize, args=(worker_id, 
                                                    input_queue, 
                                                    args.output_path,
                                                    args.tokenizer_path,
                                                    args.max_length))
        p.start()
        workers.append(p)
    
    print(f"{now()}: Started {args.num_workers} worker processes")

    # Start producer process with individual arguments instead of generator
    producer = Process(target=producer_process, args=(
        args.dataset_name,
        args.dataset_split,
        args.text_column,
        args.cache_dir,
        args.batch_size,
        label_column,
        label_filter_value,
        max_samples,
        input_queue,
        max_queue_size,
        args.num_workers
    ))
    producer.start()
    
    print(f"{now()}: Started producer process")

    # Wait for producer to complete
    producer.join()
    print(f"{now()}: Producer process completed")

    # Wait for all workers to complete
    for i, worker in enumerate(workers):
        worker.join()
        print(f"{now()}: Worker {i} completed")
    
    print(f"{now()}: All processes completed successfully!")

# how to run the script:
# python "04 tokenizer script.py" --dataset-name "TucanoBR/tucano-sft" 
#        --dataset-split "train" --tokenizer-path "../models/30k/" 
#        --max-length 512 --batch-size 25000 --num-workers 8 
#        --output-path "../data/custom_tokenized/" --label-column "label" 
#        --label-filter-value 1 --max-samples 1000000