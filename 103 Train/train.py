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
def train_model(model, args, checkpoint=None):
    """Train model using parsed arguments. Can resume from checkpoint if provided."""
    
    # Determine if we're resuming
    is_resuming = checkpoint is not None
    
    # Setup logging
    logging.basicConfig(
        filename=args.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a' if is_resuming else 'w'  # append if resuming, overwrite if fresh
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
    print(f"  Current batch: {batch_info['current_batch']}/"
          f"{batch_info['total_batches']}")

    # Initialize model, optimizer, and loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Only set up model if not resuming (model is already set up in resume_training)
    if not is_resuming:
        print(f"\tUsing device: {device}")
        model = model.to(device)

        # Then wrap with DataParallel if multiple GPUs are available
        device_count = torch.cuda.device_count()
        if torch.cuda.is_available() and device_count > 1:
            print(f"\t🚀 Using {device_count} GPUs with DataParallel")
            model = nn.DataParallel(model, device_ids=list(range(device_count)))

            # Print memory usage for each GPU
            for i in range(device_count):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"\t  GPU {i}: {allocated:.1f}GB / {total:.1f}GB allocated")
        else:
            print(f"\t💻 Using single "
                  f"{'GPU' if torch.cuda.is_available() else 'CPU'}")
    else:
        # For resumed training, model is already set up
        device = next(model.parameters()).device
        print(f"\t🔄 Resuming with model already configured")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Restore optimizer state if resuming
    if is_resuming and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ Optimizer state restored")
        except Exception as e:
            print(f"⚠️ Could not restore optimizer state: {e}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Training loop with batch loading support
    model.train()
    # Resume from saved step or start from 0
    global_step = checkpoint.get('step', 0) if is_resuming else 0  
    
    # Process all batches
    while True:
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
                # Calculate average loss properly for both fresh and resumed training
                steps_in_current_session = (global_step 
                            - (checkpoint.get('step', 0) if is_resuming else 0))
                avg_loss = (total_loss / steps_in_current_session 
                           if steps_in_current_session > 0 else loss_val.item())

                # Log progress
                if global_step % args.progress_freq == 0:
                    progress = dataset.get_overall_progress()
                    duration = time.time() - epoch_start_time
                    # Format time nicely with timedelta
                    time_str = str(timedelta(seconds=int(duration))) 
                    logging.info(
                        f"Epoch {epoch+1}, Step {global_step}, "
                        f"Loss: {loss_val.item():.4f}, "
                        f"Avg Loss: {avg_loss:.4f}, "
                        f"Time: {time_str}"
                    )
                    print(
                        f"Epoch {epoch+1}, Step {global_step}, "
                        f"Loss: {loss_val.item():.4f}, "
                        f"Avg Loss: {avg_loss:.4f}, "
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
        
        # Save checkpoint after completing current batch
        if isinstance(model, nn.DataParallel):
            model_to_save = model.module
        else:
            model_to_save = model
        save_checkpoint(
            model_to_save, optimizer, args.num_epochs, global_step, avg_loss, 
            args.checkpoint_dir, 'distilgpt2_batch_complete'
        )
        print(f"💾 Checkpoint saved after completing batch "
              f"{batch_info['current_batch']}")

        # Mark current batch as processed after saving checkpoint
        dataset._mark_current_batch_as_processed()
        print(f"✅ Marked batch {batch_info['current_batch']} as processed")

        # Check if we need to load next batch
        if not dataset.has_next_batch():
            print("✅ All batches processed!")
            break

        # Load next batch
        print(f"🔄 Loading next batch...")
        if not dataset.load_next_batch():
            print("❌ Failed to load next batch!")
            break
        
        # Show info about newly loaded batch
        new_batch_info = dataset.get_batch_info()
        print(f"✅ Successfully loaded batch {new_batch_info['current_batch']}"
              f"/{new_batch_info['total_batches']}")
        print(f"\tFiles in new batch: "
              f"{new_batch_info['files_in_current_batch']}")
        print(f"\tSequences in new batch: "
              f"{new_batch_info['sequences_in_current_batch']}")

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
    
    # Return the model for further use if needed
    return model
    
# Resume training function - now much simpler
def resume_training(model, args):
    """Resume training from a checkpoint."""
    
    print(f"🔄 Resuming training from checkpoint: "
          f"{args.resume_from_checkpoint}")
    
    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)

    # Move model to device first
    model = model.to(device)

    # Load state dict first (before wrapping with DataParallel)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Handle DataParallel wrapping AFTER loading state dict
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"\t🚀 Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    else:
        print(f"\t💻 Using 1 {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    print(f"✅ Resumed from epoch {checkpoint['epoch']}, "
          f"step {checkpoint['step']}")
    print(f"Last recorded loss: {checkpoint['loss']:.4f}")
    
    # Continue training by passing the checkpoint to train_model
    return train_model(model, args, checkpoint=checkpoint)

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