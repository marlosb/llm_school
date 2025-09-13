from multiprocess import Process, Queue
import os
import time
import pickle
from typing import List, Dict, Any
import json

def parallel_embedding_worker(input_queue: Queue, 
                            output_queue: Queue,
                            worker_id: str,
                            tokenizer_path: str = r"../model/30k.json"):
    """
    Worker function that processes dataset chunks and generates embeddings.
    """
    from datasets import load_dataset
    from transformers import PreTrainedTokenizerFast
    from sentence_transformers import SentenceTransformer
    import torch
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    
    print(f"Worker {worker_id} started.")
    
    def generate_embeddings_for_chunk(chunk_data: List[str]) -> torch.Tensor:
        """Generate embeddings for a chunk of text data."""
        try:
            # Generate embeddings
            embeddings = tokenizer.encode(chunk_data, 
                                    convert_to_tensor=True,
                                    show_progress_bar=False,
                                    batch_size=32)
            return embeddings
        except Exception as e:
            print(f"Error in worker {worker_id}: {e}")
            return torch.tensor([])
    
    while True:
        item = input_queue.get()
        if item is None:  # Sentinel value to exit
            print(f"Worker {worker_id} exiting")
            output_queue.put(None)
            break
            
        chunk_id, chunk_data = item
        print(f"Worker {worker_id} processing chunk {chunk_id} with {len(chunk_data)} samples")
        
        # Generate embeddings for this chunk
        embeddings = generate_embeddings_for_chunk(chunk_data)
        
        result = {
            'chunk_id': chunk_id,
            'worker_id': worker_id,
            'embeddings': embeddings,
            'chunk_size': len(chunk_data),
            'embedding_shape': embeddings.shape if len(embeddings) > 0 else None
        }
        
        output_queue.put(result)
        print(f"Worker {worker_id} completed chunk {chunk_id}")

def load_and_chunk_dataset(dataset_name: str = 'TucanoBR/GigaVerbo',
                          dataset_split: str = 'train',
                          cache_dir: str = '../data',
                          text_column: str = 'text',
                          chunk_size: int = 1000,
                          max_samples: int = None) -> List[List[str]]:
    """
    Load dataset and split into chunks for parallel processing.
    """
    from datasets import load_dataset
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=dataset_split, cache_dir=cache_dir)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Extract text and create chunks
    chunks = []
    current_chunk = []
    
    for i, item in enumerate(dataset):
        text = item[text_column]
        if isinstance(text, str) and text.strip():  # Only process valid text
            current_chunk.append(text.strip())
            
            if len(current_chunk) >= chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
    
    # Add remaining items if any
    if current_chunk:
        chunks.append(current_chunk)
    
    print(f"Created {len(chunks)} chunks with chunk size {chunk_size}")
    return chunks

def save_embeddings_to_disk(embeddings_list: List[Dict], 
                           output_dir: str = './embeddings_output'):
    """
    Combine all embeddings and save to disk in multiple formats.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by chunk_id to maintain order
    embeddings_list.sort(key=lambda x: x['chunk_id'])
    
    # Combine all embeddings
    all_embeddings = []
    metadata = []
    
    for chunk_result in embeddings_list:
        if len(chunk_result['embeddings']) > 0:
            all_embeddings.extend(chunk_result['embeddings'])
            metadata.append({
                'chunk_id': chunk_result['chunk_id'],
                'worker_id': chunk_result['worker_id'],
                'chunk_size': chunk_result['chunk_size'],
                'start_idx': len(all_embeddings) - chunk_result['chunk_size'],
                'end_idx': len(all_embeddings)
            })
    
    # Convert to numpy array
    embeddings_array = np.array(all_embeddings)
    
    # Save in different formats
    print(f"Saving {embeddings_array.shape[0]} embeddings of dimension {embeddings_array.shape[1]}")
    
    # 1. Save as numpy array
    np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings_array)
    
    # 2. Save as pickle (includes metadata)
    with open(os.path.join(output_dir, 'embeddings_with_metadata.pkl'), 'wb') as f:
        pickle.dump({
            'embeddings': embeddings_array,
            'metadata': metadata,
            'total_samples': embeddings_array.shape[0],
            'embedding_dimension': embeddings_array.shape[1]
        }, f)
    
    # 3. Save metadata as JSON
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump({
            'metadata': metadata,
            'total_samples': int(embeddings_array.shape[0]),
            'embedding_dimension': int(embeddings_array.shape[1]),
            'total_chunks': len(metadata)
        }, f, indent=2)
    
    print(f"Embeddings saved to {output_dir}")
    return embeddings_array, metadata

def run_parallel_embedding_generation(dataset_name: str = 'TucanoBR/wikipedia-PT',
                                     dataset_split: str = 'train',
                                     cache_dir: str = '../data',
                                     text_column: str = 'text',
                                     chunk_size: int = 1000,
                                     max_samples: int = 10000,
                                     num_workers: int = 4,
                                     model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                                     output_dir: str = './embeddings_output'):
    """
    Main function to orchestrate the parallel embedding generation process.
    """
    start_time = time.time()
    
    print("=" * 60)
    print("PARALLEL EMBEDDING GENERATION")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Max samples: {max_samples}")
    print(f"Chunk size: {chunk_size}")
    print(f"Number of workers: {num_workers}")
    print("=" * 60)
    
    # Step 1: Load and chunk the dataset
    print("\nStep 1: Loading and chunking dataset...")
    chunks = load_and_chunk_dataset(
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        cache_dir=cache_dir,
        text_column=text_column,
        chunk_size=chunk_size,
        max_samples=max_samples
    )
    
    # Step 2: Set up parallel processing
    print(f"\nStep 2: Setting up {num_workers} workers...")
    input_queue = Queue()
    output_queue = Queue()
    
    # Add chunks to input queue
    for i, chunk in enumerate(chunks):
        input_queue.put((i, chunk))
        print(f"Added chunk {i} with {len(chunk)} samples to queue")
    
    # Start worker processes
    workers = []
    for i in range(num_workers):
        print(f"Starting worker {i}")
        p = Process(target=parallel_embedding_worker, 
                   args=(input_queue, output_queue, i, model_name))
        workers.append(p)
        p.start()
    
    # Add sentinel values to signal workers to exit
    for _ in range(num_workers):
        input_queue.put(None)
    
    # Step 3: Collect results
    print("\nStep 3: Collecting results...")
    results = []
    sentinel_count = 0
    
    while sentinel_count < num_workers:
        result = output_queue.get()
        if result is None:
            sentinel_count += 1
        else:
            print(f"Received result from worker {result['worker_id']} for chunk {result['chunk_id']}")
            results.append(result)
    
    # Wait for all workers to finish
    for p in workers:
        p.join()
    
    # Step 4: Save embeddings to disk
    print("\nStep 4: Saving embeddings to disk...")
    embeddings_array, metadata = save_embeddings_to_disk(results, output_dir)
    
    total_time = time.time() - start_time
    
    print("=" * 60)
    print("PROCESS COMPLETED")
    print("=" * 60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total embeddings generated: {embeddings_array.shape[0]}")
    print(f"Embedding dimension: {embeddings_array.shape[1]}")
    print(f"Output directory: {output_dir}")
    print(f"Average time per sample: {total_time/embeddings_array.shape[0]:.4f} seconds")
    print("=" * 60)
    
    return embeddings_array, metadata

# Example usage and testing
if __name__ == "__main__":
    # Test with a small dataset first
    embeddings, metadata = run_parallel_embedding_generation(
        dataset_name='TucanoBR/wikipedia-PT',
        max_samples=5000,  # Start small for testing
        chunk_size=500,
        num_workers=3,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        output_dir='./test_embeddings'
    )