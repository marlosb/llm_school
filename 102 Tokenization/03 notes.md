Here is your conversation formatted in Markdown:

---

I'm learning about LLMs. To get started I create 3 tokenizers that are saved to local disk in json files. Now I want to test these tokenizers. My plan is to tokenize a large dataset from hugging face (TucanoBR/wikipedia-PT) using the 3 tokenizers and see which one consume less tokens.  
Help me create the code to load the dataset, all tokenizers, encode the dataset, count the tokens without keep encoded result to save resources.

---

Let's create a Python script to load the Wikipedia-PT dataset from Hugging Face, load your three tokenizers, tokenize the dataset, and count the tokens for each tokenizer while being memory-efficient. Since you're working with a large dataset, we'll use streaming to process it incrementally and avoid storing the encoded results.

Here's the code:

```python
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
import os
from tqdm import tqdm

def load_tokenizers(tokenizer_paths):
    """Load multiple tokenizers from JSON files."""
    tokenizers = []
    for path in tokenizer_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file {path} not found.")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
        tokenizers.append((os.path.basename(path), tokenizer))
    return tokenizers

def count_tokens(dataset, tokenizers, text_column='text', max_samples=None):
    """Count tokens for each tokenizer on the dataset."""
    token_counts = {name: 0 for name, _ in tokenizers}
    sample_count = 0

    # Iterate over the dataset with a progress bar
    for sample in tqdm(dataset, desc="Processing dataset", total=max_samples):
        text = sample[text_column]
        if not isinstance(text, str):
            continue  # Skip non-string entries

        # Count tokens for each tokenizer
        for name, tokenizer in tokenizers:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_counts[name] += len(tokens)

        sample_count += 1
        if max_samples and sample_count >= max_samples:
            break

    return token_counts, sample_count

def main():
    # Paths to your tokenizer JSON files
    tokenizer_paths = [
        "tokenizer1.json",
        "tokenizer2.json",
        "tokenizer3.json"
    ]

    # Load tokenizers
    print("Loading tokenizers...")
    try:
        tokenizers = load_tokenizers(tokenizer_paths)
    except Exception as e:
        print(f"Error loading tokenizers: {e}")
        return

    # Load the dataset in streaming mode
    print("Loading dataset...")
    try:
        dataset = load_dataset("TucanoBR/wikipedia-PT", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Count tokens (set max_samples to limit processing for testing, e.g., 10000)
    max_samples = None  # Set to None to process the entire dataset
    print("Counting tokens...")
    token_counts, sample_count = count_tokens(dataset, tokenizers, max_samples=max_samples)

    # Print results
    print(f"\nProcessed {sample_count} samples.")
    print("\nToken counts per tokenizer:")
    for name, count in token_counts.items():
        print(f"{name}: {count} tokens")

    # Determine which tokenizer used the fewest tokens
    if token_counts:
        min_tokenizer = min(token_counts.items(), key=lambda x: x[1])
        print(f"\nTokenizer with fewest tokens: {min_tokenizer[0]} ({min_tokenizer[1]} tokens)")

if __name__ == "__main__":
    main()
```

---

#### How the Code Works

1. **Dependencies**:
   - `datasets`: To load the Hugging Face dataset.
   - `transformers`: To use `PreTrainedTokenizerFast` for loading tokenizers.
   - `tqdm`: To show a progress bar for processing the dataset.

2. **Loading Tokenizers**:
   - The `load_tokenizers` function loads each tokenizer from the specified JSON files using `PreTrainedTokenizerFast`.
   - It checks if the files exist to avoid errors.

3. **Processing the Dataset**:
   - The code streams the dataset so you don’t use much memory.
   - For each record, it uses each tokenizer to encode the text and accumulates the token count (does NOT keep encoded output).

4. **Result**:
   - Prints the number of tokens each tokenizer produced and identifies the one with the fewest tokens.

---

Let me know if you need further customization!

Here's your content converted to Markdown, ignoring HTML/CSS specifics and focusing purely on structure and clarity.

---

## Token Counting

- The `count_tokens` function processes the dataset in streaming mode, reading one sample at a time to minimize memory usage.
- For each sample, it extracts the text from the `text` column (default for Wikipedia datasets).
- It encodes the text using each tokenizer and counts the tokens without storing the encoded results.
- A `max_samples` parameter allows you to limit the number of samples processed for testing (set to `None` for the full dataset).
- A progress bar (`tqdm`) tracks processing.

### Main Function

- Specifies the paths to your tokenizer JSON files (update these to match your file names).
- Loads the `TucanoBR/wikipedia-PT` dataset in streaming mode.
- Calls `count_tokens` to process the dataset and count tokens.
- Prints the total number of samples processed and the token count for each tokenizer.
- Identifies the tokenizer with the fewest tokens.

## Setup Instructions

1. ### Install Dependencies

   ```bash
   pip install datasets transformers tqdm
   ```

2. ### Update Tokenizer Paths

    - Modify the `tokenizer_paths` list in the `main` function to point to your JSON tokenizer files (e.g., `["path/to/tokenizer1.json", ...]`).

3. ### Dataset

    - The `TucanoBR/wikipedia-PT` dataset is loaded from Hugging Face. Ensure you have an internet connection.
    - The dataset is streamed to avoid loading it entirely into memory.

4. ### Running the Script

    - Save the script as, e.g., `test_tokenizers.py`.
    - Run it with:
      ```bash
      python test_tokenizers.py
      ```

5. ### Optional: Limit Samples

    - For testing, set `max_samples` in the `main` function to a small number (e.g., `10000`) to process only a subset of the dataset.

## Notes

- **Memory Efficiency:** The script uses streaming and does not store encoded tokens, making it suitable for large datasets.
- **Text Column:** The script assumes the dataset has a `text` column, which is standard for Wikipedia datasets. If the column name differs, update the `text_column` parameter in `count_tokens`.
- **Error Handling:** The script includes basic error handling for file loading and dataset access.
- **Tokenizer Format:** Ensure your JSON tokenizer files are compatible with `PreTrainedTokenizerFast`. They should follow the format generated by Hugging Face's tokenizer training (e.g., containing `vocab`, `merges`, etc.).

## Example Output

```
Loading tokenizers...
Loading dataset...
Processing dataset: 100%|██████████| 10000/10000 [00:45<00:00, 220.00it/s]

Processed 10000 samples.
Token counts per tokenizer:
tokenizer1.json: 1250000 tokens
tokenizer2.json: 1150000 tokens
tokenizer3.json: 1200000 tokens

Tokenizer with fewest tokens: tokenizer2.json (1150000 tokens)
```

This output shows the token counts for each tokenizer and identifies the one with the fewest tokens.

If you have specific requirements (e.g., a different dataset column, additional metrics, or specific tokenizer configurations), let me know, and I can adjust the code!

---

## Your follow-up question

> the code is working fine and I'm expanding it to 9 tokenizer. I see a single logical core under load on my system. How could we run it in multi thread to speed it up? 9 threads, one to each tokenizer would be nice - i have 12 logical cores available

---

## Assistant Response (Markdown)

To speed up the tokenization process for your 9 tokenizers by utilizing your 12 logical cores, you can parallelize the tokenization task using Python's `multiprocessing` module. Since each tokenizer can process the dataset independently, we can assign one process per tokenizer. This is preferable to `threading`, since tokenization is CPU-bound and Python's Global Interpreter Lock (GIL) restricts threading for CPU workloads.

Below is how you can modify your code to use the `multiprocessing` module, with one process per tokenizer. The dataset will still be streamed, and each process will work independently to keep memory usage efficient.

### Modified Code Example (multiprocessing)

```python
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
import multiprocessing as mp
from tqdm import tqdm

def count_tokens_for_single_tokenizer(tokenizer_path, text_column, max_samples, dataset_name):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    dataset = load_dataset(dataset_name, split='train', streaming=True)
    sample_count = 0
    token_count = 0
    for sample in tqdm(dataset, desc=f"Tokenizer: {tokenizer_path}", disable=True):
        text = sample[text_column]
        token_count += len(tokenizer.encode(text))
        sample_count += 1
        if max_samples and sample_count >= max_samples:
            break
    return tokenizer_path, token_count, sample_count

def main():
    tokenizer_paths = [
        "path/to/tokenizer1.json",
        "path/to/tokenizer2.json",
        # ... add up to 9
    ]
    dataset_name = "TucanoBR/wikipedia-PT"
    text_column = "text"
    max_samples = 10000  # or None for all

    with mp.Pool(processes=len(tokenizer_paths)) as pool:
        results = [
            pool.apply_async(
                count_tokens_for_single_tokenizer, 
                (path, text_column, max_samples, dataset_name)
            ) for path in tokenizer_paths
        ]
        outputs = [res.get() for res in results]

    # Print results
    for (tokenizer_path, token_count, sample_count) in outputs:
        print(f"{tokenizer_path}: {token_count} tokens ({sample_count} samples)")
    min_item = min(outputs, key=lambda x: x[1])
    print(f"\nTokenizer with fewest tokens: {min_item[0]} ({min_item[1]} tokens)")

if __name__ == '__main__':
    main()
```

### Summary

- This code will fully utilize your 12 logical cores (with 9 processes, one per tokenizer).
- Each process loads the dataset independently and counts tokens for its respective tokenizer.
- You get a significant speed-up compared to running all tokenizers serially!

---

Let me know if you want to parallelize in a different way (e.g., splitting the dataset among tokenizers) or if you have further questions!

```python
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
import os
from multiprocessing import Pool
from tqdm import tqdm
import itertools

def load_tokenizers(tokenizer_paths):
    """Load multiple tokenizers from JSON files."""
    tokenizers = []
    for path in tokenizer_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file {path} not found.")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
        tokenizers.append((os.path.basename(path), tokenizer))
    return tokenizers

def process_batch(args):
    """Process a batch of texts with a single tokenizer and return token count."""
    tokenizer_name, tokenizer, texts = args
    token_count = 0
    for text in texts:
        if isinstance(text, str):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_count += len(tokens)
    return tokenizer_name, token_count

def count_tokens_parallel(dataset, tokenizers, text_column='text', max_samples=None, batch_size=1000):
    """Count tokens for each tokenizer in parallel."""
    token_counts = {name: 0 for name, _ in tokenizers}
    sample_count = 0
    batch = []

    # Prepare iterator for dataset
    dataset_iter = iter(dataset)
    
    # Use a process pool with one process per tokenizer
    with Pool(processes=len(tokenizers)) as pool:
        for sample in tqdm(dataset_iter, desc="Processing dataset", total=max_samples):
            text = sample.get(text_column)
            if not isinstance(text, str):
                continue  # Skip non-string entries
            batch.append(text)
            sample_count += 1

            # Process batch when it reaches batch_size or at the end
            if len(batch) >= batch_size or (max_samples and sample_count >= max_samples):
                tasks = [(name, tokenizer, batch) for name, tokenizer in tokenizers]
                results = pool.map(process_batch, tasks)
                for name, count in results:
                    token_counts[name] += count
                batch = []  # Clear batch after processing

            if max_samples and sample_count >= max_samples:
                break

        # Process any remaining texts in the batch
        if batch:
            tasks = [(name, tokenizer, batch) for name, tokenizer in tokenizers]
            results = pool.map(process_batch, tasks)
            for name, count in results:
                token_counts[name] += count

    return token_counts, sample_count

def main():
    # Paths to your 9 tokenizer JSON files
    tokenizer_paths = [
        "tokenizer1.json",
        "tokenizer2.json",
        "tokenizer3.json",
        "tokenizer4.json",
        "tokenizer5.json",
        "tokenizer6.json",
        "tokenizer7.json",
        "tokenizer8.json",
        "tokenizer9.json"
    ]

    # Load tokenizers
    print("Loading tokenizers...")
    try:
        tokenizers = load_tokenizers(tokenizer_paths)
    except Exception as e:
        print(f"Error loading tokenizers: {e}")
        return

    # Load the dataset in streaming mode
    print("Loading dataset...")
    try:
        dataset = load_dataset("TucanoBR/wikipedia-PT", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Count tokens in parallel (set max_samples to limit processing for testing, e.g., 10000)
    max_samples = None  # Set to None to process the entire dataset
    batch_size = 1000   # Adjust based on memory and performance

    print(f"Counting tokens with {len(tokenizers)} processes...")
    token_counts, sample_count = count_tokens_parallel(
        dataset,
        tokenizers,
        max_samples=max_samples,
        batch_size=batch_size,
    )

    # Print results
    print(f"\nProcessed {sample_count} samples.")
    print("\nToken counts per tokenizer:")
    for name, count in token_counts.items():
        print(f"{name}: {count} tokens")

    # Determine which tokenizer used the fewest tokens
    if token_counts:
        min_tokenizer = min(token_counts.items(), key=lambda x: x[1])
        print(f"\nTokenizer with fewest tokens: {min_tokenizer[0]} ({min_tokenizer[1]} tokens)")

if __name__ == "__main__":
    main()
```

---

## Key Changes for Multiprocessing

1. **Multiprocessing Pool:**
    - The `Pool` from `multiprocessing` is used to create 9 processes, one for each tokenizer, matching your request to utilize 9 of your 12 logical cores.
    - Each process handles tokenization for one tokenizer, running in parallel.

2. **Batch Processing:**
    - To reduce overhead from process communication, the script processes texts in batches (default `batch_size=1000`).
    - Each batch of texts is sent to all tokenizers simultaneously, and each tokenizer processes the batch in its own process.
    - The `process_batch` function handles tokenization for a single tokenizer on a batch of texts and returns the total token count.

3. **Memory Efficiency:**
    - The dataset is still streamed to avoid loading it entirely into memory.
    - Batches are cleared after processing to prevent memory buildup.
    - Encoded tokens are not stored; only their counts are kept.

4. **Progress Tracking:**
    - The `tqdm` progress bar tracks the number of samples processed, though it reflects the main process's progress (not individual tokenizer progress).

---

## Setup Instructions

1. **Install Dependencies** (if not already done):

    ```bash
    pip install datasets transformers tqdm
    ```

2. **Update Tokenizer Paths:**
    - Update the `tokenizer_paths` list in the `main` function to include the paths to all 9 tokenizer JSON files.

---

You can now run this script, and it will efficiently process the dataset with 9 parallel processes—one for each tokenizer.

Here is your content in clean, structured Markdown format:

---

## Adjust Batch Size

- The `batch_size` parameter (default 1000) controls how many samples are processed before sending them to the process pool. Adjust this based on your system's memory and performance:
    - Smaller batches reduce memory usage but increase process communication overhead.
    - Larger batches use more memory but reduce overhead.
    - Experiment with values (e.g., 500, 1000, 2000) to find the optimal balance.

## Run the Script

- Save the script as, e.g., `test_tokenizers_parallel.py`.
- Run it with:

    ```bash
    python test_tokenizers_parallel.py
    ```

## Performance Considerations

- **CPU Utilization**: With 9 processes, you should see up to 9 logical cores utilized, as each tokenizer runs in its own process. You can monitor this using tools like `htop` or Task Manager.
- **Memory Usage**: Each process loads a copy of the tokenizer and processes a batch of texts. With 9 tokenizers and a batch size of 1000, memory usage should remain manageable, but monitor your system. If memory becomes an issue, reduce `batch_size`.
- **Dataset Streaming**: Streaming ensures the dataset doesn't load entirely into memory, making this approach scalable for large datasets like `TucanoBR/wikipedia-PT`.
- **Overhead**: Multiprocessing has some overhead due to process creation and data transfer. The batch processing approach minimizes this, but very small batches may reduce the speedup.

## Expected Output

```
Loading tokenizers...
Loading dataset...
Counting tokens with 9 processes...
Processing dataset: 100%|██████████| 10000/10000 [00:30<00:00, 333.33it/s]

Processed 10000 samples.
Token counts per tokenizer:
tokenizer1.json: 1250000 tokens
tokenizer2.json: 1150000 tokens
...
tokenizer9.json: 1200000 tokens

Tokenizer with fewest tokens: tokenizer2.json (1150000 tokens)
```
The runtime should be significantly reduced compared to the single-threaded version, as tokenization is parallelized across 9 processes. The actual speedup depends on your system's CPU and memory bandwidth.

## Additional Notes

- **Why Multiprocessing Instead of Threading?**: Tokenization with `PreTrainedTokenizerFast` is CPU-bound, and Python's GIL limits threading performance. `multiprocessing` bypasses the GIL by using separate processes.
- **Tuning**: If you notice uneven core utilization or high memory usage, try adjusting `batch_size` or reducing the number of processes (e.g., to 8) to leave some cores free for system tasks.
- **Error Handling**: The script includes basic error handling, but ensure your tokenizer JSON files are correctly formatted for `PreTrainedTokenizerFast`.
- **Max Samples**: For testing, set `max_samples` to a small number (e.g., 10000) to avoid processing the entire dataset.

If you encounter issues (e.g., memory errors, slow performance) or want to add features (e.g., detailed performance metrics, logging), let me know, and I can refine the code further!

---

## Trouble: Script Frozen, No CPU Load, No Results

If the script appears frozen with no CPU load and no results, it’s likely stuck due to one of these common causes: dataset streaming issues, multiprocessing deadlocks, empty/malformed datasets, or resource contention.

### Potential Issues

1. **Dataset Streaming Hang:** The Hugging Face `datasets` library in streaming mode might be waiting for data from the server or encountering an issue with the `TucanoBR/wikipedia-PT` dataset.
2. **Multiprocessing Deadlock:** The `multiprocessing.Pool` might be stuck if the batch processing or data transfer is mishandled.
3. **Empty or Malformed Dataset:** If the dataset is empty or the `text` column is missing/inconsistent, the script might wait indefinitely.
4. **Resource Contention:** Even with 12 logical cores, other system processes or memory constraints might interfere with the pool.

---

## Revised Code with Diagnostics

Below is an updated version of the code with added logging, a timeout mechanism, and smaller batch sizes to reduce the chance of freezing. It also includes a fallback to single-threaded processing if multiprocessing fails, and checks for dataset issues.

```python
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
import os
from multiprocessing import Pool
from tqdm import tqdm
import logging
import time
from functools import partial
import signal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_tokenizers(tokenizer_paths):
    """Load multiple tokenizers from JSON files."""
    tokenizers = []
    for path in tokenizer_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file {path} not found.")
        logger.info(f"Loading tokenizer: {path}")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
        tokenizers.append((os.path.basename(path), tokenizer))
    return tokenizers
```

*(And so on—the rest of your Python code would continue here as in your original message, indenting and annotating as necessary with triple backticks and no HTML/CSS formatting.)*

---

Let me know if you want to convert the rest of your code or have more content to format!

```python
)
    return tokenizers

def process_batch(args):
    """Process a batch of texts with a single tokenizer and return token count."""
    tokenizer_name, tokenizer, texts = args
    token_count = 0
    for text in texts:
        if isinstance(text, str):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_count += len(tokens)
    return tokenizer_name, token_count

def count_tokens_parallel(dataset, tokenizers, text_column='text', max_samples=None, batch_size=100):
    """Count tokens for each tokenizer in parallel with timeout."""
    token_counts = {name: 0 for name, _ in tokenizers}
    sample_count = 0
    batch = []

    # Initialize dataset iterator
    logger.info("Initializing dataset iterator...")
    try:
        dataset_iter = iter(dataset)
    except Exception as e:
        logger.error(f"Failed to initialize dataset iterator: {e}")
        raise

    # Use a process pool with one process per tokenizer
    try:
        with Pool(processes=len(tokenizers), initializer=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)) as pool:
            logger.info(f"Started {len(tokenizers)} processes for tokenization")
            for sample in tqdm(dataset_iter, desc="Processing dataset", total=max_samples):
                text = sample.get(text_column)
                if not isinstance(text, str):
                    logger.warning(f"Skipping non-string sample at index {sample_count}")
                    continue
                batch.append(text)
                sample_count += 1

                # Process batch when it reaches batch_size or at the end
                if len(batch) >= batch_size or (max_samples and sample_count >= max_samples):
                    logger.info(f"Processing batch of {len(batch)} samples")
                    tasks = [(name, tokenizer, batch) for name, tokenizer in tokenizers]
                    try:
                        # Set a timeout for the batch processing (e.g., 60 seconds)
                        results = pool.map(partial(process_batch), tasks, timeout=60)
                        for name, count in results:
                            token_counts[name] += count
                    except TimeoutError:
                        logger.error("Timeout in batch processing, switching to single-threaded mode")
                        pool.terminate()
                        pool.join()
                        return count_tokens_single(dataset, tokenizers, text_column, max_samples, sample_count, batch)
                    batch = []  # Clear batch after processing

                if max_samples and sample_count >= max_samples:
                    break

            # Process any remaining texts in the batch
            if batch:
                logger.info(f"Processing final batch of {len(batch)} samples")
                tasks = [(name, tokenizer, batch) for name, tokenizer in tokenizers]
                try:
                    results = pool.map(partial(process_batch), tasks, timeout=60)
                    for name, count in results:
                        token_counts[name] += count
                except TimeoutError:
                    logger.error("Timeout in final batch, switching to single-threaded mode")
                    pool.terminate()
                    pool.join()
                    return count_tokens_single(dataset, tokenizers, text_column, max_samples, sample_count, batch)

    except Exception as e:
        logger.error(f"Multiprocessing failed: {e}")
        logger.info("Falling back to single-threaded processing")
        return count_tokens_single(dataset, tokenizers, text_column, max_samples, sample_count, batch)

    return token_counts, sample_count

def count_tokens_single(dataset, tokenizers, text_column='text', max_samples=None, sample_count=0, remaining_batch=None):
    """Fallback single-threaded token counting."""
    logger.info("Running in single-threaded mode")
    token_counts = {name: 0 for name, _ in tokenizers}
    dataset_iter = iter(dataset)
    
    # Process any remaining batch from parallel mode
    if remaining_batch:
        logger.info(f"Processing remaining batch of {len(remaining_batch)} samples")
        for text in remaining_batch:
            for name, tokenizer in tokenizers:
                if isinstance(text, str):
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    token_counts[name] += len(tokens)

    # Continue processing the dataset
    for sample in tqdm(dataset_iter, desc="Processing dataset (single-threaded)", total=max_samples):
        text = sample.get(text_column)
        if not isinstance(text, str):
            logger.warning(f"Skipping non-string sample at index {sample_count}")
            continue
        for name, tokenizer in tokenizers:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_counts[name] += len(tokens)
        sample_count += 1

        if max_samples and sample_count >= max_samples:
            break

    return token_counts, sample_count

def main():
    # Paths to your 9 tokenizer JSON files
    tokenizer_paths = [
        "tokenizer1.json",
        "tokenizer2.json",
        "tokenizer3.json",
        "tokenizer4.json",
        "tokenizer5.json",
        "tokenizer6.json",
        "tokenizer7.json",
        "tokenizer8.json",
        "tokenizer9.json"
    ]

    # Load tokenizers
    logger.info("Loading tokenizers...")
    try:
        tokenizers = load_tokenizers(tokenizer_paths)
    except Exception as e:
        logger.error(f"Error loading tokenizers: {e}")
        return

    # Load the dataset in streaming mode
    logger.info("Loading dataset...")
    try:
        dataset = load_dataset("TucanoBR/wikipedia-PT", split="train", streaming=True)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    # Count tokens in parallel (set max_samples to limit processing for testing)
    max_samples = 10000  # Set to None for full dataset, 10000 for testing
    batch_size = 100  # Reduced for lower memory usage
    logger.info(f"Counting tokens with {len(tokenizers)} processes...")
    start_time = time.time()
    token_counts, sample_count = count_tokens_parallel(dataset, tokenizers, max_samples=max_samples, batch_size=batch_size)
    elapsed_time = time.time() - start_time

    # Print results
    logger.info(f"Processed ")
```
(Note: The code snippet is incomplete at the end as in the original HTML, but all that was present is converted.)

## Key Changes and Diagnostics

1. **Logging:**
   - Added `logging` to track progress, errors, and warnings (e.g., non-string samples, batch processing, or timeouts).
   - Logs are printed with timestamps to help identify where the script might be freezing.

2. **Timeout Mechanism:**
   - Added a `timeout=60` (seconds) to `pool.map` to prevent the script from hanging indefinitely if a process stalls.
   - If a timeout occurs, the script terminates the pool and falls back to single-threaded processing (`count_tokens_single`).

3. **Fallback to Single-Threaded:**
   - If multiprocessing fails (e.g., due to timeout or other errors), the script switches to single-threaded mode to ensure you get results, albeit slower.

4. **Reduced Batch Size:**
   - Lowered `batch_size` from 1000 to 100 to reduce memory usage and process communication overhead, which might help prevent freezing.

5. **Dataset Iterator Check:**
   - Explicitly checks for dataset iterator initialization errors.
   - Logs non-string samples to identify if the dataset has unexpected data.

6. **Signal Handling:**
   - Added `signal.SIGINT` handling in the process pool to prevent issues with interrupting the script (e.g., Ctrl+C).

7. **Limited Samples for Testing:**
   - Set `max_samples=10000` by default to process a smaller subset of the dataset for testing. Change to `None` for the full dataset once you confirm it’s working.

---

## Troubleshooting Steps

1. **Run the Script:**
   - Save the script as, e.g., `test_tokenizers_parallel_debug.py`.
   - Run it:

     ```bash
     python test_tokenizers_parallel_debug.py
     ```

   - Check the console for log messages to see where it might be stuck (e.g., "Loading dataset...", "Processing batch...").

2. **Check Log Output:**
   - Look at the log file or console for messages like:
     - `INFO: Loading dataset...`
     - `WARNING: Skipping non-string sample at index X`
     - `ERROR: Timeout in batch processing`
   - If it stops at "Loading dataset...", the issue is likely with dataset access.

3. **Test Dataset Access:**
   - To isolate dataset issues, run this snippet to check if the dataset loads:

     ```python
     from datasets import load_dataset
     dataset = load_dataset("TucanoBR/wikipedia-PT", split="train", streaming=True)

     for i, sample in enumerate(dataset):
         print(sample)
         if i >= 5:
             break
     ```

   - If this hangs, the dataset server might be slow or unreachable. Try again or check your internet connection.

4. **Monitor CPU/Memory:**
   - Use `htop` (Linux/Mac) or Task Manager (Windows) to confirm if Python processes are running. You should see 9 worker processes plus the main process.
   - If no CPU activity, the script is likely stuck on dataset loading or iterator initialization.

5. **Reduce Processes:**
   - If the issue persists, try reducing the number of processes to 4 or 6 (change `processes=len(tokenizers)` to `processes=4` in `count_tokens_parallel`). This leaves more cores free and reduces contention.

6. **Test Single-Threaded Mode:**
   - Temporarily modify the `main` function to call `count_tokens_single` directly:

     ```python
     token_counts, sample_count = count_tokens_single(dataset, tokenizers, max_samples=max_samples)
     ```

   - This bypasses multiprocessing to confirm if the issue is with the pool.

---

## Expected Behavior

- You should see log messages indicating progress (e.g., "Processing batch of 100 samples").
- CPU usage should spike across multiple cores (up to 9) during tokenization.
- If the dataset is loading correctly, the `tqdm` progress bar should advance.
- If a timeout or error occurs, the script will fall back to single-threaded mode and log the issue.

---

## Example Log Output

```
2025-08-21 11:15:23,123 - INFO - Loading tokenizers...
2025-08-21 11:15:23,125 - INFO - Loading tokenizer: tokenizer1.json
...
2025-08-21 11:15:23,150 - INFO - Loading dataset...
2025-08-21 11:15:23,200 - INFO - Initializing dataset iterator...
2025-08-21 11:15:23,210 - INFO - Counting tokens with 9 processes...
2025-08-21 11:15:23,220 - INFO - Started 9 processes for tokenization
Processing dataset:  10%|█         | 1000/10000 [00:05<00:45, 200.00it/s]
2025-08-21 11:15:28,230 - INFO - Processing batch of 100 samples
...
```

If the script still freezes, share the last log message or where the progress bar stops, and I can narrow down the issue further. You can also try running with a smaller `max_samples` (e.g., 1000) or fewer tokenizers to isolate the problem.

Here is the conversation, converted to Markdown and stripped to just its information and code content:

---

```
2025-08-21 11:18:40,007 - INFO - Initializing dataset iterator...
2025-08-21 11:18:40,013 - ERROR - Multiprocessing failed: Can't get local object 'count_tokens_parallel.<locals>.<lambda>'
2025-08-21 11:18:40,013 - INFO - Falling back to single-threaded processing
2025-08-21 11:18:40,014 - INFO - Running in single-threaded mode
```

The log output indicates that the script is failing during the initialization of the multiprocessing pool with the error:

*Can't get local object 'count_tokens_parallel.<locals>.<lambda>'*

This error typically occurs in Python's `multiprocessing` module when a lambda function (used in the `initializer` for signal handling) cannot be pickled, which is required for multiprocessing. The script then correctly falls back to single-threaded mode, but this defeats the purpose of parallelization. Let’s fix the multiprocessing issue to utilize your 9 processes and prevent the script from freezing or falling back.

---

### Cause of the Error

The error stems from the

`initializer=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)`

in the `Pool` setup. Lambda functions are not picklable, which causes `multiprocessing` to fail. We can fix this by replacing the lambda with a proper function. Additionally, we’ll simplify the signal handling and add more diagnostics to ensure the dataset iterator and multiprocessing pool work correctly.

---

### Revised Code

Here’s an updated version of the script that removes the problematic lambda, adds better error handling, and includes a test to ensure dataset access before starting multiprocessing. I’ve also kept the batch size small and added a mechanism to pre-check the dataset.

```python
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

import os
from multiprocessing import Pool
from tqdm import tqdm

import logging
import time
import signal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_pool():
    """Initialize worker processes to ignore SIGINT."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def load_tokenizers(tokenizer_paths):
    """Load multiple tokenizers from JSON files."""
    tokenizers = []
    for path in tokenizer_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file {path} not found.")
        logger.info(f"Loading tokenizer: {path}")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
        tokenizers.append((os.path.basename(path), tokenizer))
    return tokenizers

def process_batch(args):
    """Process a batch of texts with a single tokenizer and return token count."""
    tokenizer_name, tokenizer, texts = args
    token_count = 0
    for text in texts:
        if isinstance(text, str):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_count += len(tokens)
        else:
            logger.warning(f"Non-string text encountered in batch for {tokenizer_name}")
    return tokenizer_name, token_count

def count_tokens_parallel(
    dataset, tokenizers, text_column='text',
    max_samples=None, batch_size=100
):
    """Count tokens for each tokenizer in parallel with timeout."""
    token_counts = {name: 0 for name, _ in tokenizers}
    sample_count = 0
    batch = []

    # Test dataset access
    logger.info("Testing dataset access...")
    try:
        test_iter = iter(dataset)
        test_sample = next(test_iter)
        if text_column not in test_sample:
            logger.error(
                f"Column '{text_column}' not found in dataset. "
                f"Available columns: {list(test_sample.keys())}"
            )
            raise KeyError(f"Column '{text_column}' not found in dataset")
        logger.info("Dataset access test passed")
    except Exception as e:
        logger.error(f"Failed to access dataset: {e}")
        raise

    # Initialize dataset iterator
    logger.info("Initializing dataset iterator...")
    try:
        dataset_iter = iter(dataset)
    except Exception as e:
        logger.error(f"Failed to initialize dataset iterator: {e}")
        raise

    # Use a process pool with one process per tokenizer
    logger.info(f"Starting {len(tokenizers)} processes for tokenization")
    try:
        with Pool(processes=len(tokenizers), initializer=init_pool) as pool:
            for sample in tqdm(dataset_iter, desc="Processing dataset", total=max_samples):
                text = sample.get(text_column)
                if not isinstance(text, str):
                    logger.warning(f"Skipping non-string sample at index {sample_count}")
                    continue
                batch.append(text)
                sample_count += 1

                # Process batch when it reaches batch_size or at the end
                if len(batch) >= batch_size or (max_samples and sample_count >= max_samples):
                    logger.info(f"Processing batch of {len(batch)} samples")
                    tasks = [(name, tokenizer, batch) for name, tokenizer in tokenizers]
                    try:
                        results = pool.map_async(process_batch, tasks).get(timeout=60)
                        for name, count in results:
                            token_counts[name] += count
                    except Exception as e:
                        logger.error(f"Multiprocessing failed during batch: {e}")
                        raise
                    batch = []
                if max_samples and sample_count >= max_samples:
                    break
            # Final leftover batch
            if batch:
                logger.info(f"Processing final batch of {len(batch)} samples")
                tasks = [(name, tokenizer, batch) for name, tokenizer in tokenizers]
                try:
                    results = pool.map_async(process_batch, tasks).get(timeout=60)
                    for name, count in results:
                        token_counts[name] += count
                except Exception as e:
                    logger.error(f"Multiprocessing failed during final batch: {e}")
                    raise

        logger.info(f"Token count results: {token_counts}")
        return token_counts

    except Exception as e:
        logger.error(f"Multiprocessing failed: {e}")
        logger.info("Falling back to single-threaded processing")
        logger.info("Running in single-threaded mode")
        # Add a fallback single process mode if desired.
        # (add this code if needed)
        raise

```


```python
(timeout=60)
    for name, count in results:
        token_counts[name] += count
except Exception as e:
    logger.error(f"Error in batch processing: {e}")
    pool.terminate()
    pool.join()
    logger.info("Falling back to single-threaded mode")
    return count_tokens_single(dataset, tokenizers, text_column, max_samples, sample_count, batch)
batch = []  # Clear batch after processing

if max_samples and sample_count >= max_samples:
    break

# Process any remaining texts in the batch
if batch:
    logger.info(f"Processing final batch of {len(batch)} samples")
    tasks = [(name, tokenizer, batch) for name, tokenizer in tokenizers]
    try:
        results = pool.map_async(process_batch, tasks).get(timeout=60)
        for name, count in results:
            token_counts[name] += count
    except Exception as e:
        logger.error(f"Error in final batch processing: {e}")
        pool.terminate()
        pool.join()
        logger.info("Falling back to single-threaded mode")
        return count_tokens_single(dataset, tokenizers, text_column, max_samples, sample_count, batch)

except Exception as e:
    logger.error(f"Multiprocessing failed: {e}")
    logger.info("Falling back to single-threaded mode")
    return count_tokens_single(dataset, tokenizers, text_column, max_samples, sample_count, batch)

return token_counts, sample_count

def count_tokens_single(dataset, tokenizers, text_column='text', max_samples=None, sample_count=0, remaining_batch=None):
    """Fallback single-threaded token counting."""
    logger.info("Running in single-threaded mode")
    token_counts = {name: 0 for name, _ in tokenizers}
    dataset_iter = iter(dataset)
    
    # Process any remaining batch from parallel mode
    if remaining_batch:
        logger.info(f"Processing remaining batch of {len(remaining_batch)} samples")
        for text in remaining_batch:
            for name, tokenizer in tokenizers:
                if isinstance(text, str):
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    token_counts[name] += len(tokens)

    # Continue processing the dataset
    for sample in tqdm(dataset_iter, desc="Processing dataset (single-threaded)", total=max_samples):
        text = sample.get(text_column)
        if not isinstance(text, str):
            logger.warning(f"Skipping non-string sample at index {sample_count}")
            continue
        for name, tokenizer in tokenizers:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_counts[name] += len(tokens)
        sample_count += 1
        if max_samples and sample_count >= max_samples:
            break

    return token_counts, sample_count

def main():
    # Paths to your 9 tokenizer JSON files
    tokenizer_paths = [
        "tokenizer1.json",
        "tokenizer2.json",
        "tokenizer3.json",
        "tokenizer4.json",
        "tokenizer5.json",
        "tokenizer6.json",
        "tokenizer7.json",
        "tokenizer8.json",
        "tokenizer9.json"
    ]

    # Load tokenizers
    logger.info("Loading tokenizers...")
    try:
        tokenizers = load_tokenizers(tokenizer_paths)
    except Exception as e:
        logger.error(f"Error loading tokenizers: {e}")
        return

    # Load the dataset in streaming mode
    logger.info("Loading dataset...")
    try:
        dataset = load_dataset("TucanoBR/wikipedia-PT", split="train", streaming=True)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    # Count tokens in parallel
    max_samples = 10000  # Set to None for full dataset, 10000 for testing
    batch_size = 100     # Small batch size to reduce memory and overhead
    logger.info(f"Counting tokens with {len(tokenizers)} processes...")
    start_time = time.time()
    token_counts, sample_count = count_tokens_parallel(dataset, tokenizers, max_samples=max_samples, batch_size=batch_size)
    elapsed_time = time.time() - start_time

    # Print results
    logger.info(f"Processed {sample_count} samples in {elapsed_time:.2f} seconds")
    print("\nToken counts per tokenizer:")
    for name, count in token_counts.items():
        print(f"{name}: {count} tokens")

    # Determine which tokenizer used the fewest tokens
    if token_counts:
        min_tokenizer = min(token_counts.items(), key=lambda x: x[1])
        print(f"\nTokenizer with fewest tokens: {min_tokenizer[0]} ({min_tokenizer[1]} tokens)")

if __name__ == "__main__":
    main()
```

---

## Key Changes

1. **Fixed Lambda Issue:**
   - Replaced `initializer=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)` with a proper function `init_pool` to avoid pickling errors.

2. **Dataset Access Test:**
   - Added a pre-check to fetch one sample from the dataset and verify the `text` column exists, logging available columns if it fails.

3. **Improved Multiprocessing:**
   - Used `pool.map_async(...).get(timeout=60)` instead of `pool.map` for better timeout handling.
   - Ensures proper process termination on errors.

4. **Logging Enhancements:**
   - Added warnings for non-string texts within batches.
   - Logs dataset access test results and any multiprocessing errors.

5. **Kept Small Batch Size:**
   - Maintained `batch_size=100` to minimize memory usage and reduce the chance of freezing.

# Steps to Run

1. **Save and Run:**
   - Save the script as, e.g., `test_tokenizers_parallel_fixed.py`.
   - Run it:
     ```bash
     python test_tokenizers_parallel_fixed.py
     ```

2. **Monitor Output:**
   - Check the log messages for progress. You should see:
     ```
     2025-08-21 11:19:XX,XXX - INFO - Loading tokenizers...
     2025-08-21 11:19:XX,XXX - INFO - Loading dataset...
     2025-08-21 11:19:XX,XXX - INFO - Testing dataset access...
     2025-08-21 11:19:XX,XXX - INFO - Dataset access test passed
     2025-08-21 11:19:XX,XXX - INFO - Initializing dataset iterator...
     2025-08-21 11:19:XX,XXX - INFO - Starting 9 processes for tokenization
     Processing dataset: ...
     ```
   - If it stops or errors, note the last log message.

3. **Check CPU Usage:**
   - Use `htop` or Task Manager to confirm 9 Python processes are running with CPU activity.

4. **If It Freezes Again:**
   - If it hangs, note the last log message and whether the `tqdm` progress bar appears.
   - Try reducing `max_samples` to 1000 or `processes` to 4 in the `Pool` (change `processes=len(tokenizers)` to `processes=4`).

5. **Verify Dataset Columns:**
   - If the log shows `Column 'text' not found`, the `TucanoBR/wikipedia-PT` dataset might use a different column name. Run this to check:
     ```python
     from datasets import load_dataset
     dataset = load_dataset("TucanoBR/wikipedia-PT", split="train", streaming=True)
     print(next(iter(dataset)).keys())
     ```
   - Update `text_column` in the script to match the correct column (e.g., `content` or `text_content`).

---

## Expected Behavior

- The script should now initialize the multiprocessing pool without the lambda error.
- You should see CPU usage across multiple cores (up to 9) during tokenization.
- The `tqdm` progress bar should advance, and logs should indicate batch processing.
- If multiprocessing fails again, it will fall back to single-threaded mode and complete the task.

---

## If Issues Persist

- **Share Last Log Message:** Provide the last log message or any new errors.
- **Test Single-Threaded:** Temporarily modify `main` to call `count_tokens_single` directly to confirm the dataset and tokenizers work:
  ```python
  token_counts, sample_count = count_tokens_single(dataset, tokenizers, max_samples=max_samples)
  ```
- **Dataset Availability:** If the dataset access test fails, try a different dataset (e.g., `wikipedia` with `lang='pt'`) or check Hugging Face’s status for server issues.

---

This should resolve the multiprocessing error and prevent freezing. Let me know the results or any new log messages!

---

## Example Log Output

```
2025-08-21 11:23:27,261 - INFO - Starting 9 processes for tokenization
Processing dataset:  10%|███████▎   | 968/10000 [00:11<01:04, 140.03it/s]
2025-08-21 11:23:39,178 - INFO - Processing batch of 1000 samples
Processing dataset:  10%|███████▎   | 968/10000 [00:29<01:04, 140.03it/s]
2025-08-21 11:24:39,193 - ERROR - Error in batch processing: 
2025-08-21 11:24:39,300 - INFO - Falling back to single-threaded mode
```

The log output shows that the script successfully started 9 processes for tokenization and processed 968 samples (with a batch of 1000 samples in progress) before encountering an error during batch processing at around 11:24:39. The error message is empty (`Error in batch processing:`), which suggests an unhandled exception in the multiprocessing pool, likely related to data transfer, process communication, or a timeout. The script then fell back to single-threaded mode as designed. The progress bar indicates it was processing at ~140 samples/second before the error, and it froze for about a minute during the batch processing.

The lack of a specific error message makes it tricky to pinpoint the exact cause, but common issues include:

- **Multiprocessing Timeout or Resource Issue:** The batch of 1000 samples might be too large for the system’s memory or process communication, causing a stall.
- **Pickling Issue:** Some objects (e.g., tokenizer or dataset samples) might not be properly picklable, causing the `Pool` to fail.
- **Dataset Streaming Issue:** The Hugging Face dataset streaming might have hiccuped, causing the iterator to stall during batch processing.

Given your system has 12 logical cores and the script was using 9 processes, resource contention is possible but unlikely.

---

**Let’s address this by:**

1. Reducing the batch size to minimize memory and communication overhead.
2. Adding more detailed error logging to capture the specific exception.

Certainly! Here is the provided HTML-conveyed content converted into Markdown:

---

- Simplifying the multiprocessing setup to avoid pickling issues.
- Adding a retry mechanism for batch processing.

## Revised Code

This version reduces the batch size to 50, improves error logging, and adds a retry mechanism for batch processing. It also includes a check to ensure tokenizers are picklable and simplifies the multiprocessing setup.

```python
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
import os

from multiprocessing import Pool
from tqdm import tqdm
import logging
import time
import signal
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_pool():
    """Initialize worker processes to ignore SIGINT."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def load_tokenizers(tokenizer_paths):
    """Load multiple tokenizers from JSON files."""
    tokenizers = []
    for path in tokenizer_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file {path} not found.")
        logger.info(f"Loading tokenizer: {path}")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
        # Test picklability
        try:
            pickle.dumps(tokenizer)
            logger.info(f"Tokenizer {path} is picklable")
        except Exception as e:
            logger.error(f"Tokenizer {path} is not picklable: {e}")
            raise
        tokenizers.append((os.path.basename(path), tokenizer))
    return tokenizers

def process_batch(args):
    """Process a batch of texts with a single tokenizer and return token count."""
    try:
        tokenizer_name, tokenizer, texts = args
        token_count = 0
        for text in texts:
            if isinstance(text, str):
                tokens = tokenizer.encode(text, add_special_tokens=False)
                token_count += len(tokens)
            else:
                logger.warning(f"Non-string text encountered in batch for {tokenizer_name}")
        return tokenizer_name, token_count
    except Exception as e:
        logger.error(f"Error in process_batch for {tokenizer_name}: {e}")
        raise

def count_tokens_parallel(dataset, tokenizers, text_column='text', max_samples=None, batch_size=50):
    """Count tokens for each tokenizer in parallel with retry."""
    token_counts = {name: 0 for name, _ in tokenizers}
    sample_count = 0
    batch = []

    # Test dataset access
    logger.info("Testing dataset access...")
    try:
        test_iter = iter(dataset)
        test_sample = next(test_iter)
        if text_column not in test_sample:
            logger.error(
                f"Column '{text_column}' not found in dataset. Available columns: {list(test_sample.keys())}")
            raise KeyError(f"Column '{text_column}' not found in dataset")
        logger.info("Dataset access test passed")
    except Exception as e:
        logger.error(f"Failed to access dataset: {e}")
        raise

    # Initialize dataset iterator
    logger.info("Initializing dataset iterator...")
    try:
        dataset_iter = iter(dataset)
    except Exception as e:
        logger.error(f"Failed to initialize dataset iterator: {e}")
        raise

    # Use a process pool with one process per tokenizer
    logger.info(f"Starting {len(tokenizers)} processes for tokenization")
    try:
        with Pool(processes=len(tokenizers), initializer=init_pool) as pool:
            for sample in tqdm(dataset_iter, desc="Processing dataset", total=max_samples):
                text = sample.get(text_column)
                if not isinstance(text, str):
                    logger.warning(f"Skipping non-string sample at index {sample_count}")
                    continue
                batch.append(text)
                sample_count += 1

                # Process batch when it reaches batch_size or at the end
                if len(batch) >= batch_size or (max_samples and sample_count >= max_samples):
                    logger.info(f"Processing batch of {len(batch)} samples")
                    tasks = [(name, tokenizer, batch) for name, tokenizer in tokenizers]
                    for attempt in range(3):  # Retry up to 3 times
                        try:
                            results = pool.map_async(process_batch, tasks).get(timeout=30)
                            for name, count in results:
                                token_counts[name] += count
                            break  # Success, exit retry loop
                        except Exception as e:
                            logger.error(f"Batch processing attempt {attempt + 1} failed: {e}")
                            if attempt == 2:
                                logger.info("Max retries reached, falling back to single-threaded mode")
                                pool.terminate()
                                pool.join()
                                return count_tokens_single(dataset, tokenizers, text_column, max_samples, sample_count, batch)
                    batch = []  # Clear batch after processing

                if max_samples and sample_count >= max_samples:
                    break

            # Process any remaining texts in the batch
            if batch:
                logger.info(f"Processing final batch of {len(batch)} samples")
                tasks = [(name, tokenizer, batch) for name, tokenizer in tokenizers]
                for attempt in range(3):  # Retry up to 3 times
                    try:
                        results = pool.map_async(process_batch, tasks).get(timeout=30)
                        for name, count in results:
                            token_counts[name] += count
                        break
                    except Exception as e:
                        logger.error(f"Final batch processing attempt {attempt + 1} failed: {e}")
                        if attempt == 2:
                            logger.info("Max retries reached on final batch, exiting")
                            break

    except Exception as e:
        logger.error(f"Failed in parallel processing: {e}")
        raise

    return token_counts
```

---

Here is the conversation content converted from HTML and styled spans to plain Markdown.

---

## Key Changes

1. **Reduced Batch Size:**
    - Lowered `batch_size` from 100 to 50 to reduce memory usage and communication overhead, which may prevent the stall during batch processing.

2. **Detailed Error Logging:**
    - Added detailed logging in `process_batch` to capture any errors during tokenization.
    - Logs the specific exception during batch processing attempts.

3. **Retry Mechanism:**
    - Added up to 3 retries for each batch to handle transient issues (e.g., network hiccups or temporary resource contention).
    - If all retries fail, it falls back to single-threaded mode.

4. **Picklability Check:**
    - Tests each tokenizer for picklability during loading to catch potential issues early.

5. **Timeout Adjustment:**
    - Reduced the timeout to 30 seconds per batch to catch issues faster.
    - Uses `map_async(...).get(timeout=30)` for more reliable async processing.

---

## Steps to Run

1. **Save and Run:**
    - Save the script as, e.g., `test_tokenizers_parallel_retry.py`.
    - Run it:

      ```bash
      python test_tokenizers_parallel_retry.py
      ```

2. **Monitor Logs:**
    - Check the console for logs like:

      ```
      2025-08-21 11:27:XX,XXX - INFO - Loading tokenizers...
      2025-08-21 11:27:XX,XXX - INFO - Loading dataset...
      2025-08-21 11:27:XX,XXX - INFO - Testing dataset access...
      2025-08-21 11:27:XX,XXX - INFO - Starting 9 processes for tokenization
      Processing dataset: ...
      2025-08-21 11:27:XX,XXX - INFO - Processing batch of 50 samples
      ```
    - Look for any errors and watch for detailed logging/output.

Here is your HTML content converted to Markdown, focusing only on the content and logical structure:

---

### 1. Look for any `ERROR` messages that specify the batch processing failure.

### 2. Check CPU Usage:
- Use `htop` or Task Manager to confirm 9 Python processes are active with CPU usage.

### 3. If It Fails Again:
- Note the exact error message (e.g., in `Batch processing attempt 1 failed: ...`).
- Try reducing `batch_size` to 20 or `processes` to 4 in the `Pool` (change `processes=len(tokenizers)` to `processes=4`).
- Test single-threaded mode by modifying `main` to call `count_tokens_single` directly:

    ```python
    token_counts, sample_count = count_tokens_single(dataset, tokenizers, max_samples=max_samples)
    ```

### 4. Verify Dataset:
- If the issue is dataset-related, run this to inspect samples:

    ```python
    from datasets import load_dataset
    dataset = load_dataset("TucanoBR/wikipedia-PT", split="train", streaming=True)

    for i, sample in enumerate(dataset):
        print(sample)
        if i >= 5:
            break
    ```

- Ensure the `text` column exists. If not, update `text_column` in the script to the correct column name.

---

## Expected Behavior

- The script should process batches of 50 samples at a time, with logs indicating progress every ~50 samples.
- You should see CPU usage across up to 9 cores during tokenization.
- If a batch fails, the retry mechanism should attempt up to 3 times, logging each failure’s error message.
- If all retries fail, it will fall back to single-threaded mode and complete the task.

---

## Troubleshooting Tips

- **Empty Error Message**: The empty error (`Error in batch processing:`) suggests the exception wasn’t properly captured. The new logging in `process_batch` should provide more details.
- **Network Issues**: If the dataset streaming is slow, try increasing the timeout to 60 seconds or test with a local dataset.
- **Memory Usage**: Monitor memory usage in Task Manager or `htop`. If it’s high, further reduce `batch_size` to 20.
- **Tokenizer Issues**: The picklability check should catch any tokenizer issues. If a tokenizer fails, ensure your JSON files are correctly formatted for `PreTrainedTokenizerFast`.

---

Please run the updated script and share the new log output, especially any `ERROR` messages or the last log before it freezes/falls back. This will help pinpoint the exact issue (e.g., specific tokenizer error, dataset issue, or system resource limit).

---