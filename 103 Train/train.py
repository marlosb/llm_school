from datetime import timedelta
import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_utils import TokenizedDataset
from arg_utils import parse_training_args, print_args

# Function to save checkpoint
def save_checkpoint(model, 
                    optimizer, 
                    epoch, 
                    step, 
                    loss, 
                    checkpoint_dir, 
                    filename_prefix):
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir, 
        f"{filename_prefix}_epoch{epoch}_step{step}.pt"
    )
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)
    logging.info(f"Checkpoint saved: {checkpoint_path}")

# Training function with file tracking
def train_model(model, args):
    """Train model using parsed arguments."""
    
    # Setup logging
    logging.basicConfig(
        filename=args.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'  # overwrite/clear the file
    )
    
    # Create dataset with file tracking and batch loading
    dataset = TokenizedDataset(
        data_dir=args.data_dir, 
        processed_files_log=args.processed_files_log,
        max_files_in_memory=args.max_files_in_memory
    )
    
    # Check if there's data to process
    if len(dataset) == 0:
        print("✅ All files have been processed! Training complete.")
        logging.info("All files have been processed! Training complete.")
        return
    
    # Show batch and overall progress info
    batch_info = dataset.get_batch_info()
    progress = dataset.get_overall_progress()
    print(f"📊 Training Progress:")
    print(f"  Total files: {progress['total_files']}")
    print(f"  Already processed: {progress['processed_files']}")
    print(f"  Remaining files: {progress['remaining_files']}")
    print(f"  Progress: {progress['overall_progress_percent']:.1f}%")
    print(f"  Current batch: {batch_info['current_batch']}/"
          f"{batch_info['total_batches']}")

    # Initialize model, optimizer, and loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\tUsing device: {device}")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Training loop with batch loading support
    model.train()
    global_step = 0
    
    # Process all batches
    while True:
        if len(dataset) == 0:
            # Try to load next batch
            if not dataset.load_next_batch():
                print("All batches processed!")
                break
            
        # Show current batch info
        batch_info = dataset.get_batch_info()
        print(f"\n🔄 Processing batch {batch_info['current_batch']}/"
              f"{batch_info['total_batches']}")
        print(f"   Files in batch: {batch_info['files_in_current_batch']}")
        print(f"   Sequences in batch: "
              f"{batch_info['sequences_in_current_batch']}")
        
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True
        )
        
        for epoch in range(args.num_epochs):
            epoch_start_time = time.time()
            total_loss = 0
            
            for batch_idx, batch in enumerate(dataloader):
                global_step += 1
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                # Forward pass
                optimizer.zero_grad()
                logits = model(input_ids, attention_mask=attention_mask)

                # Shift for language modeling (predict next token)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()

                # Compute loss
                loss_val = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                )
                loss_val.backward()
                optimizer.step()

                total_loss += loss_val.item()
                avg_loss = total_loss / global_step

                # Log progress
                if global_step % 400 == 0:
                    progress = dataset.get_overall_progress()
                    duration = time.time() - epoch_start_time
                    # Format time nicely with timedelta
                    time_str = str(timedelta(seconds=int(duration))) 
                    logging.info(
                        f"Epoch {epoch+1}, Step {global_step}, "
                        f"Loss: {loss_val.item():.4f}, "
                        f"Avg Loss: {avg_loss:.4f}, "
                        f"Progress: "
                        f"{progress['overall_progress_percent']:.1f}%, "
                        f"Time: {time_str}"
                    )
                    print(
                        f"Epoch {epoch+1}, Step {global_step}, "
                        f"Loss: {loss_val.item():.4f}, "
                        f"Avg Loss: {avg_loss:.4f}, "
                        f"Progress: "
                        f"{progress['overall_progress_percent']:.1f}%, "
                        f"Time: {time_str}"
                    )

            # Calculate epoch time
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            # Format time nicely with timedelta
            time_str = str(timedelta(seconds=int(epoch_duration)))    

            # Epoch summary
            progress = dataset.get_overall_progress()
            logging.info(
                f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}, "
                f"Time: {time_str}, "
                f"Overall progress: "
                f"{progress['overall_progress_percent']:.1f}%"
            )
            print(
                f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}, "
                f"Time: {time_str}, "
                f"Overall progress: "
                f"{progress['overall_progress_percent']:.1f}%"
            )
        
        # Check if we need to load next batch
        if not dataset.has_next_batch():
            break

        # Save checkpoint after completing current batch, before loading next
        save_checkpoint(
            model, optimizer, args.num_epochs, global_step, avg_loss, 
            args.checkpoint_dir, 'distilgpt2_batch_complete'
        )
        print(f"💾 Checkpoint saved after completing batch "
              f"{batch_info['current_batch']}")

    # Final summary
    final_progress = dataset.get_overall_progress()
    logging.info(
        f"Training completed. "
        f"Final progress: {final_progress['overall_progress_percent']:.1f}%"
    )
    print(
        f"Training completed. "
        f"Final progress: {final_progress['overall_progress_percent']:.1f}%"
    )
    
    if final_progress['remaining_files'] > 0:
        print(f"⚠️  {final_progress['remaining_files']} files still "
              "unprocessed. Run again to continue.")
    else:
        print("🎉 All files have been processed!")

    # Mark the last batch as processed when training completes
    dataset.finalize_training()

# Resume training function
def resume_training(model, args):
    """Resume training from a checkpoint."""
    
    print(f"🔄 Resuming training from checkpoint: "
          f"{args.resume_from_checkpoint}")
    
    # Load checkpoint
    checkpoint = torch.load(args.resume_from_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Resumed from epoch {checkpoint['epoch']}, "
          f"step {checkpoint['step']}")
    print(f"Last recorded loss: {checkpoint['loss']:.4f}")
    
    # Continue training
    train_model(model, args)

# Main function
def main():
    """Main training function."""
    
    # Parse arguments
    args = parse_training_args()
    
    # Print configuration
    print_args(args)
    
    # Import and initialize model
    from model import DistilGPT2
    
    model = DistilGPT2(
        vocab_size=args.vocab_size,
        max_position=args.max_position,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    )
    
    print(f"\n🤖 Model initialized with "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Check if resuming from checkpoint
    if args.resume_from_checkpoint:
        resume_training(model, args)
    else:
        train_model(model, args)

if __name__ == "__main__":
    main()

# Utilization example:
# Full custom configuration
# > python train.py --data-dir "../data/tokenized/" --batch-size 12 
# --num-epochs 5 --learning-rate 3e-5 --max-files-in-memory 3 
# --checkpoint-freq 500 --vocab-size 30000 --num-layers 8