import argparse
import os

def parse_training_args():
    """Parse command line arguments for training script."""
    
    parser = argparse.ArgumentParser(
        description='Train DistilGPT2 model with tokenized data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data/tokenized/',
        help='Directory containing tokenized NPZ files'
    )
    
    parser.add_argument(
        '--max-files-in-memory',
        type=int,
        default=1,
        help='Maximum number of files to load in memory at once'
    )
    
    parser.add_argument(
        '--processed-files-log',
        type=str,
        default='processed_files.txt',
        help='Log file to track processed files for resuming'
    )
    
    # Model arguments
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=30000,
        help='Vocabulary size for the model'
    )
    
    parser.add_argument(
        '--max-position',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    
    parser.add_argument(
        '--embed-dim',
        type=int,
        default=768,
        help='Embedding dimension'
    )
    
    parser.add_argument(
        '--num-layers',
        type=int,
        default=6,
        help='Number of transformer layers'
    )
    
    parser.add_argument(
        '--num-heads',
        type=int,
        default=12,
        help='Number of attention heads'
    )
    
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=30,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-5,
        help='Learning rate for optimizer'
    )
    
    # Checkpoint and logging arguments
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='../checkpoints',
        help='Directory to save model checkpoints'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='../logs/training.log',
        help='Path to training log file'
    )
    
    # not in use as checkpoints are now saved after data batch
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=10000,
        help='Save checkpoint every N steps'
    )

    parser.add_argument(
        '--progress-freq',
        type=int,
        default=400,
        help='Log training progress every N steps'
    )
    
    # Resume training arguments
    parser.add_argument(
        '--resume-from-checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file to resume training from'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory does not exist: {args.data_dir}")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Validate resume checkpoint
    if args.resume_from_checkpoint and not os.path.exists(
        args.resume_from_checkpoint
    ):
        raise ValueError(
            f"Checkpoint file does not exist: {args.resume_from_checkpoint}"
        )
    
    return args

def print_args(args):
    """Print all arguments in a formatted way."""
    
    print("=" * 60)
    print("🚀 TRAINING CONFIGURATION")
    print("=" * 60)
    
    print("\n📂 Data Configuration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Max files in memory: {args.max_files_in_memory}")
    print(f"  Processed files log: {args.processed_files_log}")
    
    print("\n🤖 Model Configuration:")
    print(f"  Vocabulary size: {args.vocab_size:,}")
    print(f"  Max position: {args.max_position}")
    print(f"  Embedding dimension: {args.embed_dim}")
    print(f"  Number of layers: {args.num_layers}")
    print(f"  Number of heads: {args.num_heads}")
    print(f"  Dropout: {args.dropout}")
    
    print("\n🏋️ Training Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    
    print("\n💾 Checkpoint & Logging:")
    print(f"  Checkpoint directory: {args.checkpoint_dir}")
    print(f"  Log file: {args.log_file}")
    print(f"  Checkpoint frequency: {args.checkpoint_freq} steps")
    print(f"  Progress frequency: {args.progress_freq} steps")
    
    if args.resume_from_checkpoint:
        print(f"\n🔄 Resume Configuration:")
        print(f"  Resume from: {args.resume_from_checkpoint}")
    
    print("=" * 60)