import argparse
import logging
import os

def parse_training_args():
    """Parse command line arguments for training script."""
    
    parser = argparse.ArgumentParser(
        description='Train DistilGPT2 model with tokenized data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--data-dir', type=str, default='../data/tokenized/')

    parser.add_argument('--out-dir', type=str, default='../models/mymodel/')
    
    parser.add_argument('--vocab-size', type=int, default=30000)

    parser.add_argument('--resume-from-checkpoint', type=str, default=None)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory does not exist: {args.data_dir}")
    
    # Create directories if they don't exist
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Validate resume checkpoint
    if args.resume_from_checkpoint and not os.path.exists(
        args.resume_from_checkpoint
    ):
        raise ValueError(
            f"Checkpoint file does not exist: {args.resume_from_checkpoint}"
        )
    
    return args