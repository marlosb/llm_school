import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import os

class TokenizedDataset(Dataset):
    """PyTorch Dataset for loading tokenized NPZ files with resume capability.
    
    Supports batch loading of files to manage memory usage for large datasets.
    """
    
    def __init__(self, 
                 data_dir: str, 
                 file_pattern: str = "tokenized_*.npz", 
                 processed_files_log: str = "",
                 max_files_in_memory: int = 1):
        """Initialize dataset with batch file loading capability.
        
        Args:
            data_dir: Directory containing tokenized files
            file_pattern: Glob pattern to match files
            processed_files_log: Path to log file for tracking progress
            max_files_in_memory: Maximum number of files to load at once
        """
        self.data_dir = Path(data_dir)
        self.processed_files_log = processed_files_log
        self.max_files_in_memory = max_files_in_memory
        
        # Find all available files
        all_files = sorted(list(self.data_dir.glob(file_pattern)))
        
        if not all_files:
            raise ValueError(
                f"No files found matching pattern '{file_pattern}' "
                f"in {data_dir}"
            )
        
        print(f"Found {len(all_files)} total tokenized files")
        
        # Load processed files list if log file exists
        processed_files = set()
        if processed_files_log and os.path.exists(processed_files_log):
            with open(processed_files_log, 'r') as f:
                processed_files = set(
                    line.strip() for line in f.readlines() 
                    if line.strip()
                )
            print(f"Found {len(processed_files)} already processed files")
            
            # Filter out processed files
            self.files = [
                f for f in all_files 
                if str(f) not in processed_files
            ]
            skipped = len(all_files) - len(self.files)
            print(f"Skipping {skipped} processed files")
        else:
            # No log file exists or not provided - process all files
            self.files = all_files
            if processed_files_log:
                print(f"Log file '{processed_files_log}' doesn't exist - "
                      "processing all files")
            else:
                print("No log file specified - processing all files")
        
        if not self.files:
            print("All files have been processed! No new data to load.")
            self._initialize_empty_dataset()
            return
        
        print(f"Will process {len(self.files)} files in batches of "
              f"{max_files_in_memory}")
        
        # Initialize batch loading
        self.current_batch_idx = 0
        self.total_batches = (
            len(self.files) + max_files_in_memory - 1
        ) // max_files_in_memory
        
        # Load first batch
        self._load_current_batch()
        
        print(f"Loaded batch 1/{self.total_batches}")
        print(f"Total sequences in current batch: {len(self.input_ids)}")
        memory_mb = (self.input_ids.nbytes + self.attention_masks.nbytes) / (1024**2)
        print(f"Memory usage: {memory_mb:.2f} MB")

    def _initialize_empty_dataset(self):
        """Initialize empty dataset when no files to process."""
        self.input_ids = np.array([], dtype=np.int16).reshape(0, 0)
        self.attention_masks = np.array([], dtype=np.bool_).reshape(0, 0)
        self.file_boundaries = []
        self.current_batch_files = []
        self.current_batch_idx = 0
        self.total_batches = 0
    
    def _load_current_batch(self):
        """Load the current batch of files into memory."""
        start_idx = self.current_batch_idx * self.max_files_in_memory
        end_idx = min(
            start_idx + self.max_files_in_memory, 
            len(self.files)
        )
        
        self.current_batch_files = self.files[start_idx:end_idx]
        
        print(f"Loading batch {self.current_batch_idx + 1}/"
            f"{self.total_batches}:")
        for f in self.current_batch_files:
            print(f"  - {f.name}")
        
        # Load data from current batch files
        self.input_ids = []
        self.attention_masks = []
        self.file_boundaries = []
        
        current_sequence_count = 0
        
        for file_path in self.current_batch_files:
            data = np.load(file_path)
            tokens = data['tokens']
            masks = data['attention_mask']
            
            print(f"Loading {file_path.name}: {tokens.shape[0]} sequences")
            
            # Convert to memory-efficient data types
            tokens_int16 = tokens.astype(np.int16)
            masks_bool = masks.astype(np.bool_)
            
            # Track file boundaries for marking as processed later
            start_seq_idx = current_sequence_count
            end_seq_idx = current_sequence_count + len(tokens_int16)
            self.file_boundaries.append({
                'file_path': str(file_path),
                'start_idx': start_seq_idx,
                'end_idx': end_seq_idx,
                'processed': False
            })
            current_sequence_count = end_seq_idx
            
            self.input_ids.extend(tokens_int16)
            self.attention_masks.extend(masks_bool)
        
        # Convert to numpy arrays
        self.input_ids = np.array(self.input_ids, dtype=np.int16)
        self.attention_masks = np.array(self.attention_masks, dtype=np.bool_)
    
    def load_next_batch(self):
        """Load the next batch of files. Returns True if successful."""
        # Mark current batch files as processed before loading next batch
        self._mark_current_batch_as_processed()
        
        if self.current_batch_idx + 1 >= self.total_batches:
            print("No more batches to load")
            return False
        
        self.current_batch_idx += 1
        self._load_current_batch()
        
        print(f"Loaded batch {self.current_batch_idx + 1}/"
            f"{self.total_batches}")
        print(f"Total sequences in current batch: {len(self.input_ids)}")
        
        return True
    
    def _mark_current_batch_as_processed(self):
        """Mark all files in current batch as processed."""
        if not self.processed_files_log:
            return
        
        # Mark all files in current batch as processed
        with open(self.processed_files_log, 'a') as f:
            for boundary in self.file_boundaries:
                if not boundary['processed']:
                    boundary['processed'] = True
                    f.write(f"{boundary['file_path']}\n")
                    file_name = Path(boundary['file_path']).name
                    print(f"Marked file as processed: {file_name}")

    def finalize_training(self):
        """Call this at the end of training to mark the last batch as processed."""
        self._mark_current_batch_as_processed()
    
    def has_next_batch(self):
        """Check if there are more batches to load."""
        return self.current_batch_idx + 1 < self.total_batches
    
    def get_batch_info(self):
        """Get information about current batch loading state."""
        return {
            'current_batch': self.current_batch_idx + 1,
            'total_batches': self.total_batches,
            'files_in_current_batch': len(self.current_batch_files),
            'max_files_per_batch': self.max_files_in_memory,
            'sequences_in_current_batch': len(self.input_ids)
        }
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(
                self.attention_masks[idx], dtype=torch.long
            )
        }
    
    def get_file_for_sequence(self, sequence_idx: int):
        """Get the file path for a given sequence index."""
        for boundary in self.file_boundaries:
            if boundary['start_idx'] <= sequence_idx < boundary['end_idx']:
                return boundary['file_path']
        return None
    
    def get_unprocessed_files(self):
        """Get list of files that haven't been marked as processed yet."""
        return [
            b['file_path'] for b in self.file_boundaries 
            if not b['processed']
        ]
    
    def get_current_batch_progress(self):
        """Get progress information for current batch."""
        total_files = len(self.file_boundaries)
        processed_files = sum(
            1 for b in self.file_boundaries if b['processed']
        )
        return {
            'total_files_in_batch': total_files,
            'processed_files_in_batch': processed_files,
            'remaining_files_in_batch': total_files - processed_files,
            'batch_progress_percent': (
                (processed_files / total_files * 100) 
                if total_files > 0 else 0
            )
        }
    
    def get_overall_progress(self):
        """Get overall training progress across all batches."""
        # Calculate files processed in previous batches
        files_in_prev_batches = (
            self.current_batch_idx * self.max_files_in_memory
        )
        
        # Add processed files in current batch
        current_batch_progress = self.get_current_batch_progress()
        files_processed_current = (
            current_batch_progress['processed_files_in_batch']
        )
        
        total_processed = files_in_prev_batches + files_processed_current
        total_files = len(self.files)
        
        return {
            'total_files': total_files,
            'processed_files': total_processed,
            'remaining_files': total_files - total_processed,
            'overall_progress_percent': (
                (total_processed / total_files * 100) 
                if total_files > 0 else 0
            ),
            'current_batch_info': self.get_batch_info()
        }