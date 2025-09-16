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
    """Test if input_ids are integers from 0 to 29,999."""
    if test_texts is None:
        test_texts = [
            "Hello, world!",
            "This is a test sentence with some common words.",
            "Testing tokenization with números 123 and símbolos @#$%",
            "Uma frase em português para testar o tokenizador.",
            "A longer text to test multiple tokens and see if we get proper token IDs within the expected range of 0 to 29999."
        ]
    
    print("\n🔍 Testing input_ids range (should be 0 to 29,999):")
    print("-" * 60)
    
    all_valid = True
    all_token_ids = []
    
    for i, text in enumerate(test_texts, 1):
        # Tokenize the text
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded['input_ids'].squeeze().tolist()
        
        # Convert to list if it's a single token
        if isinstance(input_ids, int):
            input_ids = [input_ids]
            
        all_token_ids.extend(input_ids)
        
        # Check range
        min_id = min(input_ids)
        max_id = max(input_ids)
        valid_range = all(0 <= token_id <= 29999 for token_id in input_ids)
        
        print(f"Test {i}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"  Tokens: {len(input_ids)}")
        print(f"  Token IDs: {input_ids}")
        print(f"  Range: {min_id} to {max_id}")
        print(f"  Valid range (0-29999): {'✅ Yes' if valid_range else '❌ No'}")
        
        if not valid_range:
            all_valid = False
            invalid_tokens = [tid for tid in input_ids if not (0 <= tid <= 29999)]
            print(f"  ⚠️  Invalid tokens: {invalid_tokens}")
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
        print(f"  All tokens in valid range (0-29999): {'✅ Yes' if all_valid else '❌ No'}")
    
    return all_valid, all_token_ids

def test_special_tokens(tokenizer):
    """Test special tokens and their IDs."""
    print("\n🔧 Special Tokens:")
    print("-" * 30)
    
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
            valid = 0 <= token_id <= 29999
            print(f"  {name}: '{token}' -> ID: {token_id} {'✅' if valid else '❌'}")

def test_decode_tokens(tokenizer):
    """Test decoding some token IDs to verify they work correctly."""
    print("\n🔄 Testing Token Decoding:")
    print("-" * 35)
    
    # Test some token IDs across the range
    test_ids = [0, 1, 2, 100, 1000, 10000, 20000, 29999]
    
    for token_id in test_ids:
        try:
            decoded = tokenizer.decode([token_id])
            print(f"  Token ID {token_id:5d}: '{decoded}'")
        except Exception as e:
            print(f"  Token ID {token_id:5d}: ❌ Error - {e}")

def main():
    print("🚀 Tokenizer Input IDs Range Test")
    print("=" * 50)
    
    # Load tokenizer
    tokenizer = load_tokenizer("../models/30k/")
    if tokenizer is None:
        return
    
    # Test vocabulary size
    vocab_size = test_tokenizer_vocab_size(tokenizer)
    
    # Check if vocab size matches expected range
    if vocab_size == 30000:
        print("✅ Vocabulary size matches expected 30k tokens")
    else:
        print(f"⚠️  Vocabulary size ({vocab_size}) doesn't match expected 30k tokens")
    
    # Test input IDs range
    all_valid, all_tokens = test_input_ids_range(tokenizer)
    
    # Test special tokens
    test_special_tokens(tokenizer)
    
    # Test decoding
    test_decode_tokens(tokenizer)
    
    # Final summary
    print("\n" + "=" * 50)
    print("📋 FINAL SUMMARY:")
    if all_valid:
        print("✅ All input_ids are within the valid range (0 to 29,999)")
    else:
        print("❌ Some input_ids are outside the valid range (0 to 29,999)")
    print("=" * 50)

if __name__ == "__main__":
    main()