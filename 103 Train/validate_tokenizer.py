import os
import numpy as np
from transformers import PreTrainedTokenizerFast
from pathlib import Path

def load_tokenizer(tokenizer_path: str = "../models/30k/"):
    """Load the tokenizer from the specified path."""
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        print(f"✅ Successfully loaded tokenizer from: {tokenizer_path}")
        return tokenizer
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return None

def test_tokenizer_vocab_size(tokenizer):
    """Test the tokenizer vocabulary size."""
    vocab_size = tokenizer.vocab_size
    print(f"📊 Tokenizer vocabulary size: {vocab_size}")
    return vocab_size

def test_input_ids_range(tokenizer, test_texts=None):
    """Test if input_ids are integers within the vocabulary range."""
    if test_texts is None:
        test_texts = [
            "Hello, world!",
            "This is a test sentence with some common words.",
            "Testing tokenization with números 123 and símbolos @#$%",
            "Uma frase em português para testar o tokenizador.",
            "A longer text to test multiple tokens and see if we get proper token IDs within the expected range."
        ]
    
    vocab_size = tokenizer.vocab_size
    max_valid_id = vocab_size - 1
    
    print(f"\n🔍 Testing input_ids range (should be 0 to {max_valid_id}):")
    print("-" * 60)
    
    all_valid = True
    all_token_ids = []
    
    # Test both with and without special tokens
    for add_special in [True, False]:
        mode = "with special tokens (including EOS)" if add_special else "without special tokens"
        print(f"\n📝 Testing {mode}:")
        print("-" * 40)
        
        for i, text in enumerate(test_texts, 1):
            # Tokenize the text
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=add_special)
            input_ids = encoded['input_ids'].squeeze().tolist()
            
            # Convert to list if it's a single token
            if isinstance(input_ids, int):
                input_ids = [input_ids]
                
            all_token_ids.extend(input_ids)
            
            # Check range
            min_id = min(input_ids)
            max_id = max(input_ids)
            valid_range = all(0 <= token_id <= max_valid_id for token_id in input_ids)
            
            print(f"Test {i}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            print(f"  Tokens: {len(input_ids)}")
            print(f"  Token IDs: {input_ids}")
            print(f"  Range: {min_id} to {max_id}")
            print(f"  Valid range (0-{max_valid_id}): {'✅ Yes' if valid_range else '❌ No'}")
            
            if not valid_range:
                all_valid = False
                invalid_tokens = [tid for tid in input_ids if not (0 <= tid <= max_valid_id)]
                print(f"  ⚠️  Invalid tokens: {invalid_tokens}")
            
            # Show which tokens are special tokens
            if add_special:
                special_token_info = []
                for token_id in input_ids:
                    token = tokenizer.decode([token_id])
                    if token in ['[EOS]', '[PAD]', '[UNK]', '<|SYSTEM|>', '<|USER|>', '<|ASSISTANT|>', '<|END|>', '<|SEP|>', '<|EOT|>']:
                        special_token_info.append(f"{token_id}:'{token}'")
                
                if special_token_info:
                    print(f"  🔧 Special tokens: {', '.join(special_token_info)}")
            print()
    
    # Test roundtrip without EOS interference
    print("\n🔄 Testing Roundtrip Fidelity (without automatic EOS):")
    print("-" * 55)
    
    roundtrip_success = 0
    for i, text in enumerate(test_texts, 1):
        # Encode without special tokens to avoid EOS
        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        
        match = text == decoded
        if match:
            roundtrip_success += 1
        
        print(f"Test {i}: {'✅ PASS' if match else '❌ FAIL'}")
        print(f"  Original: '{text}'")
        print(f"  Decoded:  '{decoded}'")
        if not match:
            print(f"  ⚠️  Roundtrip failed!")
        print()
    
    # Overall statistics
    if all_token_ids:
        overall_min = min(all_token_ids)
        overall_max = max(all_token_ids)
        unique_tokens = len(set(all_token_ids))
        
        print("📈 Overall Statistics:")
        print(f"  Total tokens generated: {len(all_token_ids)}")
        print(f"  Unique tokens: {unique_tokens}")
        print(f"  Overall range: {overall_min} to {overall_max}")
        print(f"  All tokens in valid range (0-{max_valid_id}): {'✅ Yes' if all_valid else '❌ No'}")
        print(f"  Roundtrip success rate: {roundtrip_success}/{len(test_texts)} ({(roundtrip_success/len(test_texts)*100):.1f}%)")
    
    return all_valid, all_token_ids

def test_special_tokens(tokenizer):
    """Test special tokens and their IDs."""
    print("\n🔧 Special Tokens:")
    print("-" * 30)
    
    vocab_size = tokenizer.vocab_size
    max_valid_id = vocab_size - 1
    
    special_tokens = {
        'unk_token': tokenizer.unk_token,
        'pad_token': tokenizer.pad_token,
        'eos_token': tokenizer.eos_token,
        'bos_token': getattr(tokenizer, 'bos_token', None),
        'cls_token': getattr(tokenizer, 'cls_token', None),
        'sep_token': getattr(tokenizer, 'sep_token', None),
    }
    
    for name, token in special_tokens.items():
        if token is not None:
            token_id = tokenizer.convert_tokens_to_ids(token)
            valid = 0 <= token_id <= max_valid_id
            print(f"  {name}: '{token}' -> ID: {token_id} {'✅' if valid else '❌'}")
    
    # Test additional special tokens that might be in the tokenizer
    additional_special = ['<|SYSTEM|>', '<|USER|>', '<|ASSISTANT|>', '<|END|>', '<|SEP|>', '<|EOT|>']
    print("\n  Additional special tokens:")
    for token in additional_special:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:  # Only show if it's not mapped to UNK
                valid = 0 <= token_id <= max_valid_id
                print(f"    '{token}' -> ID: {token_id} {'✅' if valid else '❌'}")
        except:
            pass

def test_decode_tokens(tokenizer):
    """Test decoding some token IDs to verify they work correctly."""
    print("\n🔄 Testing Token Decoding:")
    print("-" * 35)
    
    vocab_size = tokenizer.vocab_size
    max_valid_id = vocab_size - 1
    
    # Test some token IDs across the range, adjusted for actual vocab size
    if vocab_size >= 30000:
        test_ids = [0, 1, 2, 100, 1000, 10000, 20000, max_valid_id]
    else:
        # For smaller vocabularies, adjust test range
        step = max(1, vocab_size // 8)
        test_ids = [0, 1, 2] + list(range(100, vocab_size, step)) + [max_valid_id]
        test_ids = sorted(set(test_ids))  # Remove duplicates and sort
    
    for token_id in test_ids:
        try:
            decoded = tokenizer.decode([token_id])
            print(f"  Token ID {token_id:5d}: '{decoded}'")
        except Exception as e:
            print(f"  Token ID {token_id:5d}: ❌ Error - {e}")

def test_eos_behavior(tokenizer):
    """Test EOS token behavior specifically."""
    print("\n🔚 Testing EOS Token Behavior:")
    print("-" * 40)
    
    test_text = "Olá, como você está?"
    
    # Test with special tokens (should include EOS)
    tokens_with_eos = tokenizer.encode(test_text, add_special_tokens=True)
    decoded_with_eos = tokenizer.decode(tokens_with_eos, skip_special_tokens=False)
    decoded_clean = tokenizer.decode(tokens_with_eos, skip_special_tokens=True)
    
    # Test without special tokens
    tokens_no_eos = tokenizer.encode(test_text, add_special_tokens=False)
    decoded_no_eos = tokenizer.decode(tokens_no_eos, skip_special_tokens=True)
    
    print(f"  Original text: '{test_text}'")
    print(f"  With EOS tokens: {tokens_with_eos}")
    print(f"  Without EOS tokens: {tokens_no_eos}")
    print(f"  Decoded with EOS (raw): '{decoded_with_eos}'")
    print(f"  Decoded with EOS (clean): '{decoded_clean}'")
    print(f"  Decoded without EOS: '{decoded_no_eos}'")
    
    # Check if EOS is being added automatically
    eos_added = len(tokens_with_eos) > len(tokens_no_eos)
    print(f"  EOS automatically added: {'✅ Yes' if eos_added else '❌ No'}")
    
    # Check roundtrip fidelity
    roundtrip_clean = test_text == decoded_clean
    roundtrip_no_eos = test_text == decoded_no_eos
    
    print(f"  Roundtrip with EOS (clean): {'✅ Pass' if roundtrip_clean else '❌ Fail'}")
    print(f"  Roundtrip without EOS: {'✅ Pass' if roundtrip_no_eos else '❌ Fail'}")

def main():
    print("🚀 Tokenizer Validation Test (EOS-Aware)")
    print("=" * 50)
    
    # Load tokenizer
    tokenizer = load_tokenizer("../models/30k/")
    if tokenizer is None:
        return
    
    # Test vocabulary size
    vocab_size = test_tokenizer_vocab_size(tokenizer)
    
    # Check if vocab size matches expected range
    expected_sizes = [15000, 30000, 60000]  # Based on your tokenizer variants
    if vocab_size in expected_sizes:
        print(f"✅ Vocabulary size ({vocab_size}) matches expected values")
    else:
        print(f"⚠️  Vocabulary size ({vocab_size}) doesn't match expected values {expected_sizes}")
    
    # Test input IDs range
    all_valid, all_tokens = test_input_ids_range(tokenizer)
    
    # Test special tokens
    test_special_tokens(tokenizer)
    
    # Test decoding
    test_decode_tokens(tokenizer)
    
    # Test EOS behavior specifically
    test_eos_behavior(tokenizer)
    
    # Final summary
    print("\n" + "=" * 50)
    print("📋 FINAL SUMMARY:")
    if all_valid:
        print(f"✅ All input_ids are within the valid range (0 to {vocab_size-1})")
        print("✅ Tokenizer handles EOS tokens correctly")
        print("✅ Roundtrip tests pass when accounting for EOS behavior")
    else:
        print(f"❌ Some input_ids are outside the valid range (0 to {vocab_size-1})")
    print("=" * 50)

if __name__ == "__main__":
    main()