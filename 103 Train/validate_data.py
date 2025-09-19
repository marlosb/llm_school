import os

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from typing import List, Tuple, Dict, Any

from data_utils import TokenizedDataset


def get_padding_token_id(tokenizer: PreTrainedTokenizerFast) -> int:
    """Get the actual padding token ID from the tokenizer."""
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    elif tokenizer.eos_token_id is not None:
        # Based on the tokenizer script, pad_token is set to eos_token
        return tokenizer.eos_token_id
    else:
        # Default assumption
        return 0

def validate_tokenizer_compatibility(tokenizer: PreTrainedTokenizerFast, token_sample: np.ndarray) -> Dict[str, Any]:
    """Validate that the tokenizer is compatible with the tokenized data."""
    
    print("🔍 Validating Tokenizer Compatibility...")
    print("-" * 50)
    
    results = {
        'valid': True,
        'issues': [],
        'stats': {}
    }
    
    # Get tokenizer info
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocabulary size: {vocab_size}")
    print(f"Tokenizer type: {type(tokenizer).__name__}")
    
    # Check special tokens
    special_tokens = {
        'pad_token': tokenizer.pad_token,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token': tokenizer.eos_token,
        'eos_token_id': tokenizer.eos_token_id,
        'unk_token': tokenizer.unk_token,
        'unk_token_id': tokenizer.unk_token_id,
    }
    
    print(f"Special tokens:")
    for name, value in special_tokens.items():
        print(f"  {name}: {value}")
    
    # Test decoding with a small sample
    sample_tokens = token_sample[:min(50, len(token_sample))]
    
    try:
        # Test basic decoding
        decoded = tokenizer.decode(sample_tokens, skip_special_tokens=False)
        print(f"Sample decoded text (first 100 chars): {decoded[:100]}...")
        
        # Check for suspicious patterns that indicate tokenizer mismatch
        suspicious_patterns = ['�', '▁', '[UNK]', '<unk>']
        has_suspicious = any(pattern in decoded for pattern in suspicious_patterns)
        
        if has_suspicious:
            results['issues'].append("Decoded text contains suspicious patterns indicating tokenizer mismatch")
            print(f"⚠️  Suspicious patterns detected in decoded text")
        
        # Check if most tokens decode to reasonable text
        if len(decoded.strip()) < len(sample_tokens) * 0.1:  # Very short decode suggests issues
            results['issues'].append("Decoded text is suspiciously short")
            print(f"⚠️  Decoded text is suspiciously short")
            
    except Exception as e:
        results['valid'] = False
        results['issues'].append(f"Decoding failed: {str(e)}")
        print(f"❌ Decoding failed: {e}")
    
    # Test encoding-decoding round trip
    try:
        test_text = "Este é um teste de compatibilidade do tokenizer."
        encoded = tokenizer.encode(test_text)
        decoded_back = tokenizer.decode(encoded)
        
        if test_text.lower().strip() not in decoded_back.lower().strip():
            results['issues'].append("Round-trip encoding-decoding test failed")
            print(f"⚠️  Round-trip test failed")
            print(f"  Original: {test_text}")
            print(f"  Decoded:  {decoded_back}")
        else:
            print(f"✅ Round-trip encoding-decoding test passed")
            
    except Exception as e:
        results['issues'].append(f"Round-trip test failed: {str(e)}")
        print(f"⚠️  Round-trip test failed: {e}")
    
    results['stats'] = special_tokens
    results['stats']['vocab_size'] = vocab_size
    
    return results

def validate_input_ids(input_ids: np.ndarray, tokenizer: PreTrainedTokenizerFast, max_length: int = 512) -> Dict[str, Any]:
    """Validate input IDs according to DistilGPT2 requirements."""
    
    print("🔍 Validating Input IDs...")
    print("-" * 50)
    
    results = {
        'valid': True,
        'issues': [],
        'stats': {}
    }
    
    vocab_size = tokenizer.vocab_size
    
    # Check shape
    print(f"Shape: {input_ids.shape}")
    print(f"Data type: {input_ids.dtype}")
    results['stats']['shape'] = input_ids.shape
    results['stats']['dtype'] = str(input_ids.dtype)
    
    if len(input_ids.shape) != 2:
        results['valid'] = False
        results['issues'].append(f"Expected 2D array, got {len(input_ids.shape)}D")
    
    # Check sequence length
    seq_length = input_ids.shape[1] if len(input_ids.shape) > 1 else 0
    print(f"Sequence length: {seq_length}")
    results['stats']['sequence_length'] = seq_length
    
    if seq_length > max_length:
        results['valid'] = False
        results['issues'].append(f"Sequence length {seq_length} exceeds max_length {max_length}")
    
    # Check token range
    min_token = np.min(input_ids)
    max_token = np.max(input_ids)
    print(f"Token range: {min_token} to {max_token}")
    results['stats']['min_token'] = int(min_token)
    results['stats']['max_token'] = int(max_token)
    
    if min_token < 0:
        results['valid'] = False
        results['issues'].append(f"Found negative token IDs: min = {min_token}")
    
    if max_token >= vocab_size:
        results['valid'] = False
        results['issues'].append(f"Found token IDs >= vocab_size: max = {max_token}, vocab_size = {vocab_size}")
    
    # Check for proper integer type
    if not np.issubdtype(input_ids.dtype, np.integer):
        results['valid'] = False
        results['issues'].append(f"Non-integer dtype: {input_ids.dtype}")
    
    # **CRITICAL FIX**: Check for data type compatibility with tokenization script
    if input_ids.dtype == np.int16:
        print("⚠️  Data is stored as int16, but tokenization script uses uint16")
        print("   This mismatch can cause token interpretation errors!")
        
        # Check if values fit in uint16 range (0 to 65535)
        if min_token < 0:
            results['valid'] = False
            results['issues'].append(f"int16 data contains negative values that don't map to uint16: min = {min_token}")
        elif max_token > 65535:
            results['valid'] = False
            results['issues'].append(f"Token values exceed uint16 range: max = {max_token}")
        else:
            print("✅ Token values could fit in uint16 range, but dtype conversion needed")
            
    elif input_ids.dtype == np.uint16:
        print("✅ Data type matches tokenization script (uint16)")
        if max_token > vocab_size - 1:
            results['valid'] = False
            results['issues'].append(f"Token values exceed vocabulary: max = {max_token}, vocab_size = {vocab_size}")
    else:
        print(f"⚠️  Unexpected data type: {input_ids.dtype}")
    
    # Token statistics
    unique_tokens = len(np.unique(input_ids))
    print(f"Unique tokens used: {unique_tokens}")
    results['stats']['unique_tokens'] = unique_tokens
    
    # Most common tokens (convert to regular int for display)
    token_counts = np.bincount(input_ids.flatten())
    most_common_indices = np.argsort(token_counts)[-10:][::-1]
    most_common = [(int(idx), int(token_counts[idx])) for idx in most_common_indices if token_counts[idx] > 0]
    print(f"Most common tokens: {most_common[:5]}")
    results['stats']['most_common_tokens'] = most_common[:10]
    
    return results

def validate_attention_masks(attention_masks: np.ndarray, input_ids: np.ndarray, tokenizer: PreTrainedTokenizerFast) -> Dict[str, Any]:
    """Validate attention masks."""
    
    print("\n🔍 Validating Attention Masks...")
    print("-" * 50)
    
    results = {
        'valid': True,
        'issues': [],
        'stats': {}
    }
    
    # Get the actual padding token ID
    padding_token_id = get_padding_token_id(tokenizer)
    print(f"Expected padding token ID: {padding_token_id}")
    print(f"Expected padding token: '{tokenizer.pad_token}' (set to EOS token)")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    
    # Check shape alignment
    if attention_masks.shape != input_ids.shape:
        results['valid'] = False
        results['issues'].append(f"Shape mismatch: attention_masks {attention_masks.shape} vs input_ids {input_ids.shape}")
        return results
    
    print(f"Shape: {attention_masks.shape} ✅")
    print(f"Data type: {attention_masks.dtype}")
    results['stats']['shape'] = attention_masks.shape
    results['stats']['dtype'] = str(attention_masks.dtype)
    
    # Check binary values
    unique_values = np.unique(attention_masks)
    print(f"Unique values: {unique_values}")
    results['stats']['unique_values'] = unique_values.tolist()
    
    # For boolean arrays, check if values are only True/False
    if attention_masks.dtype == np.bool_:
        if not np.all(np.isin(unique_values, [False, True])):
            results['valid'] = False
            results['issues'].append(f"Non-boolean values found: {unique_values}")
        else:
            print("✅ Boolean attention masks are valid")
    else:
        # For integer arrays, check if values are only 0/1
        if not np.all(np.isin(unique_values, [0, 1])):
            results['valid'] = False
            results['issues'].append(f"Non-binary values found: {unique_values}")
    
    # Count sequences and their properties
    total_sequences = attention_masks.shape[0]
    
    # Handle boolean vs integer masks
    if attention_masks.dtype == np.bool_:
        sequences_with_padding = np.sum(np.any(~attention_masks, axis=1))  # ~mask for False values
        average_length = np.mean(np.sum(attention_masks, axis=1))
    else:
        sequences_with_padding = np.sum(np.any(attention_masks == 0, axis=1))
        average_length = np.mean(np.sum(attention_masks, axis=1))
    
    print(f"Total sequences: {total_sequences}")
    print(f"Sequences with padding: {sequences_with_padding}")
    print(f"Average sequence length: {average_length:.2f}")
    
    results['stats']['total_sequences'] = total_sequences
    results['stats']['sequences_with_padding'] = int(sequences_with_padding)
    results['stats']['average_length'] = float(average_length)
    
    # Validate padding alignment with more detailed analysis
    misaligned_count = 0
    misaligned_examples = []
    
    for i in range(min(1000, total_sequences)):  # Check first 1000 sequences
        mask = attention_masks[i]
        tokens = input_ids[i]
        
        # Handle boolean vs integer masks
        if attention_masks.dtype == np.bool_:
            padding_positions = ~mask  # False positions are padding
        else:
            padding_positions = mask == 0
        
        # Check if corresponding tokens are padding tokens
        if np.any(padding_positions):
            padding_tokens = tokens[padding_positions]
            unique_padding_tokens = np.unique(padding_tokens)
            
            # Check if all padding tokens match the expected padding token
            if not np.all(padding_tokens == padding_token_id):
                misaligned_count += 1
                if len(misaligned_examples) < 5:  # Store first 5 examples
                    misaligned_examples.append({
                        'sequence_idx': i,
                        'expected_padding_token': padding_token_id,
                        'actual_padding_tokens': unique_padding_tokens.tolist(),
                        'padding_positions_count': np.sum(padding_positions)
                    })
    
    if misaligned_count > 0:
        results['valid'] = False
        results['issues'].append(f"Found {misaligned_count} sequences with misaligned padding")
        
        print(f"\n⚠️  Padding Misalignment Details:")
        print(f"Expected padding token: {padding_token_id}")
        for example in misaligned_examples:
            print(f"  Sequence {example['sequence_idx']}: found tokens {example['actual_padding_tokens']} in {example['padding_positions_count']} padding positions")
        
        # Additional analysis: check if the misalignment is systematic
        print(f"\n🔍 Additional Analysis:")
        print(f"  The tokenizer script sets: pad_token = eos_token")
        print(f"  Expected padding token ID should be: {tokenizer.eos_token_id}")
        print(f"  If misalignment persists, check if tokenization used different settings")
    else:
        print("Padding alignment: ✅")
    
    return results

def inspect_samples(dataset: TokenizedDataset, tokenizer: PreTrainedTokenizerFast, num_samples: int = 5):
    """Inspect a few samples visually with enhanced debugging."""
    
    print(f"\n👀 Inspecting {num_samples} Random Samples...")
    print("=" * 80)
    
    padding_token_id = get_padding_token_id(tokenizer)
    print(f"Using padding token ID: {padding_token_id}")
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        input_ids = sample['input_ids'].numpy()
        attention_mask = sample['attention_mask'].numpy()
        
        print(f"\nSample {i+1} (index {idx}):")
        print("-" * 40)
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Input IDs dtype: {input_ids.dtype}")
        print(f"Attention mask shape: {attention_mask.shape}")
        print(f"Attention mask dtype: {attention_mask.dtype}")
        
        # Show first 20 tokens
        print(f"First 20 tokens: {input_ids[:20].tolist()}")
        print(f"First 20 masks:  {attention_mask[:20].tolist()}")
        
        # Show last 20 tokens (where padding usually is)
        print(f"Last 20 tokens:  {input_ids[-20:].tolist()}")
        print(f"Last 20 masks:   {attention_mask[-20:].tolist()}")
        
        # **ENHANCED**: Check token range for this specific sample
        sample_min, sample_max = np.min(input_ids), np.max(input_ids)
        print(f"Sample token range: {sample_min} to {sample_max}")
        
        if sample_max >= tokenizer.vocab_size:
            print(f"⚠️  Sample contains tokens >= vocab_size ({tokenizer.vocab_size})")
        
        # Check for padding alignment in this sample
        padding_positions = attention_mask == 0
        if np.any(padding_positions):
            padding_tokens = input_ids[padding_positions]
            unique_padding_tokens = np.unique(padding_tokens)
            print(f"Padding tokens found: {unique_padding_tokens.tolist()}")
            if not np.all(padding_tokens == padding_token_id):
                print(f"⚠️  Expected padding token {padding_token_id}, but found {unique_padding_tokens.tolist()}")
            else:
                print(f"✅ All padding tokens match expected value {padding_token_id}")
        else:
            print("No padding found in this sequence")
        
        # **ENHANCED**: More robust decoding with error handling
        try:
            # Try decoding the full sequence
            decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
            print(f"Decoded (first 200 chars): {decoded_text[:200]}...")
            
            # Also try decoding just the first 10 tokens for debugging
            sample_decode = tokenizer.decode(input_ids[:10], skip_special_tokens=False)
            print(f"First 10 tokens decoded: '{sample_decode}'")
            
            # Check individual token decoding
            problem_tokens = []
            for j, token_id in enumerate(input_ids[:10]):
                try:
                    single_decode = tokenizer.decode([token_id], skip_special_tokens=False)
                    if not single_decode.strip():
                        problem_tokens.append((j, token_id))
                except:
                    problem_tokens.append((j, token_id))
            
            if problem_tokens:
                print(f"⚠️  Problem tokens (first 10): {problem_tokens}")
                
        except Exception as e:
            print(f"❌ Decoding error: {e}")
            # Try to identify the problematic token
            try:
                for j in range(min(20, len(input_ids))):
                    token_id = input_ids[j]
                    if token_id >= tokenizer.vocab_size:
                        print(f"  Token at position {j}: {token_id} >= vocab_size ({tokenizer.vocab_size})")
            except:
                pass
        
        # Statistics
        valid_tokens = np.sum(attention_mask)  
        padding_tokens_count = len(attention_mask) - valid_tokens
        print(f"Valid tokens: {valid_tokens}, Padding tokens: {padding_tokens_count}")

# ...existing code...
def test_with_model(dataset: TokenizedDataset, model_path: str = None, batch_size: int = 4):
    """Test data with the actual DistilGPT2 model."""
    
    print(f"\n🤖 Testing with DistilGPT2 Model...")
    print("-" * 50)
    
    # Import the model
    try:
        from model import DistilGPT2
        print("✅ Successfully imported DistilGPT2 model")
    except ImportError as e:
        print(f"❌ Error importing model: {e}")
        return
    
    # Create model instance
    model = DistilGPT2(
        vocab_size=30000,
        max_position=512,
        embed_dim=768,
        num_layers=6,
        num_heads=12,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Test forward pass with multiple batches
    model.eval()
    test_results = {
        'total_batches': 0,
        'successful_batches': 0,
        'failed_batches': 0,
        'padding_mask_success': 0,
        'causal_mask_success': 0,
        'output_shapes_correct': 0,
        'min_output': float('inf'),
        'max_output': float('-inf'),
        'errors': []
    }
    
    max_batches = 15
    print(f"Testing {max_batches} batches...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
                
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            test_results['total_batches'] += 1
            
            batch_success = True
            
            try:
                # Test with padding-aware attention mask
                logits = model(input_ids, attention_mask)
                test_results['padding_mask_success'] += 1
                
                # Check output shape
                expected_shape = (input_ids.shape[0], input_ids.shape[1], 30000)
                if logits.shape == expected_shape:
                    test_results['output_shapes_correct'] += 1
                
                # Track output range
                batch_min = logits.min().item()
                batch_max = logits.max().item()
                test_results['min_output'] = min(test_results['min_output'], batch_min)
                test_results['max_output'] = max(test_results['max_output'], batch_max)
                
                # Test without explicit attention mask (causal only)
                logits_causal = model(input_ids)
                test_results['causal_mask_success'] += 1
                
            except Exception as e:
                batch_success = False
                test_results['errors'].append(f"Batch {i+1}: {str(e)}")
            
            if batch_success:
                test_results['successful_batches'] += 1
            else:
                test_results['failed_batches'] += 1
    
    # Print summary
    print(f"\n📊 Model Testing Summary:")
    print(f"  Total batches tested: {test_results['total_batches']}")
    print(f"  Successful batches: {test_results['successful_batches']}")
    print(f"  Failed batches: {test_results['failed_batches']}")
    print(f"  Success rate: {(test_results['successful_batches']/test_results['total_batches']*100):.1f}%")
    
    print(f"\n🔍 Detailed Results:")
    print(f"  Padding mask tests passed: {test_results['padding_mask_success']}/{test_results['total_batches']}")
    print(f"  Causal mask tests passed: {test_results['causal_mask_success']}/{test_results['total_batches']}")
    print(f"  Correct output shapes: {test_results['output_shapes_correct']}/{test_results['total_batches']}")
    
    if test_results['min_output'] != float('inf'):
        print(f"  Output value range: {test_results['min_output']:.3f} to {test_results['max_output']:.3f}")
    
    if test_results['errors']:
        print(f"\n❌ Errors encountered:")
        for error in test_results['errors'][:3]:  # Show first 3 errors
            print(f"  - {error}")
        if len(test_results['errors']) > 3:
            print(f"  ... and {len(test_results['errors']) - 3} more errors")
    
    # Overall assessment
    if test_results['successful_batches'] == test_results['total_batches']:
        print(f"\n✅ All model tests passed successfully!")
    elif test_results['successful_batches'] > test_results['total_batches'] * 0.8:
        print(f"\n⚠️  Most tests passed, but some issues detected")
    else:
        print(f"\n❌ Significant issues detected in model testing")

def print_data_statistics(input_ids: np.ndarray, attention_masks: np.ndarray):
    """Print detailed data statistics."""
    
    print(f"\n📊 Data Statistics...")
    print("-" * 30)
    
    # Basic statistics
    print(f"Total sequences: {input_ids.shape[0]:,}")
    print(f"Sequence length: {input_ids.shape[1]}")
    
    # Memory usage with data type info
    input_ids_mb = input_ids.nbytes / (1024**2)
    attention_masks_mb = attention_masks.nbytes / (1024**2)
    total_mb = input_ids_mb + attention_masks_mb
    
    print(f"Memory usage:")
    print(f"  Input IDs ({input_ids.dtype}): {input_ids_mb:.2f} MB")
    print(f"  Attention masks ({attention_masks.dtype}): {attention_masks_mb:.2f} MB")
    print(f"  Total: {total_mb:.2f} MB")
    
    # Token usage statistics
    total_tokens = input_ids.size
    unique_tokens = len(np.unique(input_ids))
    print(f"\nToken Statistics:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Unique tokens: {unique_tokens:,}")
    print(f"  Vocabulary utilization: {unique_tokens/30000*100:.1f}%")
    
    # Padding statistics
    if attention_masks.dtype == np.bool_:
        padding_tokens = np.sum(~attention_masks)  # False values
        valid_tokens = np.sum(attention_masks)     # True values
    else:
        padding_tokens = np.sum(attention_masks == 0)
        valid_tokens = np.sum(attention_masks == 1)
        
    print(f"\nPadding Statistics:")
    print(f"  Valid tokens: {valid_tokens:,} ({valid_tokens/total_tokens*100:.1f}%)")
    print(f"  Padding tokens: {padding_tokens:,} ({padding_tokens/total_tokens*100:.1f}%)")

def main():
    """Main validation function."""
    
    # Add argument parsing
    parser = argparse.ArgumentParser(
        description='Validate DistilGPT2 tokenized data'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=1,
        help='Number of files to load in memory at once (default: 1)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default="../data/tokenized/",
        help='Directory containing tokenized files (default: ../data/tokenized/)'
    )
    parser.add_argument(
        '--tokenizer-path',
        type=str,
        default="../models/30k/",
        help='Path to tokenizer (default: ../models/30k/)'
    )
    
    args = parser.parse_args()
    
    print("🚀 DistilGPT2 Data Validation (Enhanced with Tokenizer Compatibility)")
    print("=" * 70)
    
    # Configuration from arguments
    data_dir = args.data_dir
    tokenizer_path = args.tokenizer_path
    batch_size = args.batch_size
    vocab_size = 30000
    max_length = 512
    
    print(f"Using batch size: {batch_size} files")
    
    try:
        # Load tokenizer
        print(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        print("✅ Tokenizer loaded successfully")
        
        # Print tokenizer padding info
        print(f"\n📋 Tokenizer Configuration:")
        print(f"  Vocabulary size: {tokenizer.vocab_size}")
        print(f"  Padding token: {tokenizer.pad_token}")
        print(f"  Padding token ID: {tokenizer.pad_token_id}")
        print(f"  EOS token: {tokenizer.eos_token}")
        print(f"  EOS token ID: {tokenizer.eos_token_id}")
        print(f"  UNK token: {tokenizer.unk_token}")
        print(f"  UNK token ID: {tokenizer.unk_token_id}")
        print(f"  Note: Tokenizer script sets pad_token = eos_token")
        
        # Load dataset with batch size
        print(f"\nLoading dataset from: {data_dir}")
        dataset = TokenizedDataset(
            data_dir, 
            max_files_in_memory=batch_size
        )
        print("✅ Dataset loaded successfully")
        
        # Get data arrays for validation
        input_ids = dataset.input_ids
        attention_masks = dataset.attention_masks
        
        print(f"\nDataset Summary:")
        print(f"  Total sequences: {len(dataset)}")
        print(f"  Input IDs shape: {input_ids.shape} (dtype: {input_ids.dtype})")
        print(f"  Attention masks shape: {attention_masks.shape} "
              f"(dtype: {attention_masks.dtype})")
        
        # Show batch info
        batch_info = dataset.get_batch_info()
        print(f"  Batch info: {batch_info['current_batch']}/"
              f"{batch_info['total_batches']} "
              f"({batch_info['files_in_current_batch']} files)")
        
        # **NEW**: First validate tokenizer compatibility
        validation_results = {}
        
        # Get a sample for tokenizer testing
        if len(input_ids) > 0:
            sample_tokens = input_ids[0] if len(input_ids.shape) > 1 else input_ids[:100]
            validation_results['tokenizer'] = validate_tokenizer_compatibility(
                tokenizer, sample_tokens
            )
        
        # 1. Validate input IDs (now with tokenizer parameter)
        validation_results['input_ids'] = validate_input_ids(
            input_ids, tokenizer, max_length
        )
        
        # 2. Validate attention masks
        validation_results['attention_masks'] = validate_attention_masks(
            attention_masks, input_ids, tokenizer
        )
        
        # 3. Print data statistics
        print_data_statistics(input_ids, attention_masks)
        
        # 4. Inspect samples (enhanced version)
        inspect_samples(dataset, tokenizer, num_samples=3)
        
        # 5. Test with model
        test_with_model(dataset, batch_size=2)
        
        # Final report
        print("\n" + "=" * 70)
        print("📋 FINAL VALIDATION REPORT")
        print("=" * 70)
        
        all_valid = True
        for component, results in validation_results.items():
            status = "✅ PASS" if results['valid'] else "❌ FAIL"
            print(f"{component.upper()}: {status}")
            
            if not results['valid']:
                all_valid = False
                for issue in results['issues']:
                    print(f"  - {issue}")
        
        print("\n" + "-" * 70)
        if all_valid:
            print("🎉 ALL VALIDATIONS PASSED!")
            print("Your data is ready for training with DistilGPT2!")
            print("💾 Memory optimization: Using int16 for tokens and "
                  "bool for attention masks")
            print("🔧 Model updated: Now properly handles padding masks "
                  "combined with causal masks")
        else:
            print("⚠️  SOME VALIDATIONS FAILED!")
            print("Please address the issues above before training.")
            
            # Provide specific guidance for common issues
            has_padding_issues = any(
                'misaligned padding' in issue 
                for results in validation_results.values() 
                for issue in results.get('issues', [])
            )
            has_tokenizer_issues = any(
                'tokenizer' in issue.lower() or 'decoding' in issue.lower()
                for results in validation_results.values() 
                for issue in results.get('issues', [])
            )
            
            if has_tokenizer_issues:
                print("\n💡 Tokenizer Compatibility Issues:")
                print("   The tokenizer used for validation doesn't match the one used for tokenization.")
                print("   Potential fixes:")
                print("   1. Ensure you're using the same tokenizer path for both tokenization and validation")
                print("   2. Check if the tokenizer was updated after tokenization")
                print("   3. Verify the tokenizer vocabulary size matches your data")
                print("   4. Consider re-tokenizing with the correct tokenizer")
                
            if has_padding_issues:
                print("\n💡 Padding Alignment Issues:")
                print("   The attention masks don't align with the expected padding token.")
                print("   Potential fixes:")
                print("   1. Check if pad_token was properly set during tokenization")
                print("   2. Verify the tokenizer script used the same pad_token setting")
                print("   3. Consider re-tokenizing with correct padding configuration")
                
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()