import datetime
import argparse
from typing import List, Tuple, Dict
import pandas as pd
from collections import Counter

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from difflib import SequenceMatcher

def now() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings"""
    return SequenceMatcher(None, a, b).ratio()

def analyze_differences(original: str, decoded: str) -> Dict[str, any]:
    """Analyze the specific differences between original and decoded text"""
    analysis = {
        'length_diff': len(decoded) - len(original),
        'char_diff_count': sum(1 for i, (a, b) in enumerate(zip(original, decoded)) if a != b),
        'similarity_ratio': similarity(original, decoded),
        'has_whitespace_issues': False,
        'whitespace_changes': []
    }
    
    # Check for whitespace issues
    original_spaces = [i for i, char in enumerate(original) if char.isspace()]
    decoded_spaces = [i for i, char in enumerate(decoded) if char.isspace()]
    
    if len(original_spaces) != len(decoded_spaces):
        analysis['has_whitespace_issues'] = True
        analysis['whitespace_changes'].append(f"Space count: {len(original_spaces)} -> {len(decoded_spaces)}")
    
    # Look for common patterns like split words
    original_words = original.split()
    decoded_words = decoded.split()
    
    if original_words != decoded_words:
        analysis['word_changes'] = {
            'original_word_count': len(original_words),
            'decoded_word_count': len(decoded_words),
            'word_diff_examples': []
        }
        
        # Find examples of word differences
        if len(original_words) != len(decoded_words):
            analysis['word_changes']['word_diff_examples'].append(
                f"Word count changed: {len(original_words)} -> {len(decoded_words)}"
            )
    
    return analysis

def test_tokenizer_roundtrip(
    tokenizer_path: str,
    dataset_name: str = 'TucanoBR/wikipedia-pt',
    dataset_split: str = 'train[:1000]',
    text_column: str = 'text',
    cache_dir: str = '../data',
    max_length: int = 512,
    output_file: str = None,
    test_mode: str = 'without_special'  # New parameter to control EOS behavior
) -> Dict[str, any]:
    
    print(f"{now()}: Loading tokenizer from {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"{now()}: Loading dataset {dataset_name} split {dataset_split}")
    dataset = load_dataset(dataset_name, split=dataset_split, cache_dir=cache_dir)
    
    results = {
        'total_texts': len(dataset),
        'perfect_matches': 0,
        'imperfect_matches': 0,
        'failed_roundtrips': 0,
        'similarity_scores': [],
        'problematic_examples': [],
        'whitespace_issues': 0,
        'analysis_details': [],
        'test_mode': test_mode
    }
    
    # Determine tokenization strategy based on test mode
    use_special_tokens = test_mode == 'with_special'
    mode_description = "with special tokens (including EOS)" if use_special_tokens else "without special tokens"
    
    print(f"{now()}: Testing roundtrip fidelity on {len(dataset)} texts...")
    print(f"{now()}: Test mode: {mode_description}")
    
    for i, item in enumerate(dataset):
        if i % 100 == 0:
            print(f"{now()}: Processed {i}/{len(dataset)} texts")
        
        original_text = item[text_column]
        
        # Skip empty or very short texts
        if not original_text or len(original_text.strip()) < 10:
            continue
            
        try:
            if test_mode == 'without_special':
                # Method 1: Encode without special tokens to avoid EOS interference
                tokens = tokenizer.encode(
                    original_text,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=False
                )
                decoded_text = tokenizer.decode(tokens, skip_special_tokens=True)
                
            elif test_mode == 'with_special':
                # Method 2: Encode with special tokens but handle EOS properly
                tokens = tokenizer(
                    original_text,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=True
                )
                # Decode with skip_special_tokens=True to remove EOS for comparison
                decoded_text = tokenizer.decode(tokens['input_ids'], skip_special_tokens=True)
                
            elif test_mode == 'both':
                # Method 3: Test both approaches and report on both
                # This mode will be handled separately below
                pass
            
            # Compare original with decoded
            if original_text == decoded_text:
                results['perfect_matches'] += 1
            else:
                results['imperfect_matches'] += 1
                
                # Analyze the differences
                analysis = analyze_differences(original_text, decoded_text)
                results['analysis_details'].append(analysis)
                results['similarity_scores'].append(analysis['similarity_ratio'])
                
                if analysis['has_whitespace_issues']:
                    results['whitespace_issues'] += 1
                
                # Store problematic examples (limit to first 20)
                if len(results['problematic_examples']) < 20:
                    results['problematic_examples'].append({
                        'index': i,
                        'original': original_text[:200] + "..." if len(original_text) > 200 else original_text,
                        'decoded': decoded_text[:200] + "..." if len(decoded_text) > 200 else decoded_text,
                        'similarity': analysis['similarity_ratio'],
                        'analysis': analysis
                    })
                
        except Exception as e:
            results['failed_roundtrips'] += 1
            print(f"Error processing text {i}: {e}")
    
    # Calculate statistics
    total_processed = results['perfect_matches'] + results['imperfect_matches']
    if total_processed > 0:
        results['perfect_match_percentage'] = (results['perfect_matches'] / total_processed) * 100
        results['imperfect_match_percentage'] = (results['imperfect_matches'] / total_processed) * 100
        
        if results['similarity_scores']:
            results['average_similarity'] = sum(results['similarity_scores']) / len(results['similarity_scores'])
            results['min_similarity'] = min(results['similarity_scores'])
            results['max_similarity'] = max(results['similarity_scores'])
    
    return results

def test_both_modes(
    tokenizer_path: str,
    dataset_name: str = 'TucanoBR/wikipedia-pt',
    dataset_split: str = 'train[:100]',  # Smaller sample for comparison
    text_column: str = 'text',
    cache_dir: str = '../data',
    max_length: int = 512
) -> Dict[str, any]:
    """Test both with and without special tokens and compare results"""
    
    print(f"{now()}: Running comparative test - both modes")
    
    # Test without special tokens
    results_without = test_tokenizer_roundtrip(
        tokenizer_path, dataset_name, dataset_split, text_column, 
        cache_dir, max_length, test_mode='without_special'
    )
    
    # Test with special tokens
    results_with = test_tokenizer_roundtrip(
        tokenizer_path, dataset_name, dataset_split, text_column,
        cache_dir, max_length, test_mode='with_special'
    )
    
    # Compare results
    comparison = {
        'without_special': {
            'perfect_matches': results_without['perfect_matches'],
            'perfect_match_percentage': results_without.get('perfect_match_percentage', 0),
            'average_similarity': results_without.get('average_similarity', 0),
        },
        'with_special': {
            'perfect_matches': results_with['perfect_matches'],
            'perfect_match_percentage': results_with.get('perfect_match_percentage', 0),
            'average_similarity': results_with.get('average_similarity', 0),
        }
    }
    
    return comparison

def print_results(results: Dict[str, any]):
    """Print comprehensive results"""
    print("\n" + "="*60)
    print("TOKENIZER ROUNDTRIP TEST RESULTS")
    print("="*60)
    
    if 'test_mode' in results:
        print(f"Test mode: {results['test_mode']}")
    
    total_processed = results['perfect_matches'] + results['imperfect_matches']
    
    print(f"Total texts processed: {total_processed}")
    print(f"Perfect matches: {results['perfect_matches']} ({results.get('perfect_match_percentage', 0):.2f}%)")
    print(f"Imperfect matches: {results['imperfect_matches']} ({results.get('imperfect_match_percentage', 0):.2f}%)")
    print(f"Failed roundtrips: {results['failed_roundtrips']}")
    print(f"Texts with whitespace issues: {results['whitespace_issues']}")
    
    if results['similarity_scores']:
        print(f"\nSimilarity Statistics:")
        print(f"Average similarity: {results.get('average_similarity', 0):.4f}")
        print(f"Minimum similarity: {results.get('min_similarity', 0):.4f}")
        print(f"Maximum similarity: {results.get('max_similarity', 0):.4f}")
    
    print(f"\nProblematic Examples (showing first {len(results['problematic_examples'])}):")
    print("-" * 60)
    
    for i, example in enumerate(results['problematic_examples']):
        print(f"\nExample {i+1} (Index {example['index']}, Similarity: {example['similarity']:.4f}):")
        print(f"ORIGINAL: {example['original']}")
        print(f"DECODED:  {example['decoded']}")
        
        if 'word_changes' in example['analysis']:
            word_changes = example['analysis']['word_changes']
            print(f"Word count: {word_changes['original_word_count']} -> {word_changes['decoded_word_count']}")
        
        if example['analysis']['has_whitespace_issues']:
            print(f"WHITESPACE ISSUES: {example['analysis']['whitespace_changes']}")
        
        print("-" * 40)

def print_comparison_results(comparison: Dict[str, any]):
    """Print comparison between both test modes"""
    print("\n" + "="*60)
    print("COMPARATIVE TEST RESULTS")
    print("="*60)
    
    without = comparison['without_special']
    with_special = comparison['with_special']
    
    print(f"{'Mode':<20} {'Perfect Matches':<15} {'Success Rate':<12} {'Avg Similarity':<15}")
    print("-" * 65)
    print(f"{'Without EOS':<20} {without['perfect_matches']:<15} {without['perfect_match_percentage']:<11.2f}% {without['average_similarity']:<15.4f}")
    print(f"{'With EOS (clean)':<20} {with_special['perfect_matches']:<15} {with_special['perfect_match_percentage']:<11.2f}% {with_special['average_similarity']:<15.4f}")
    
    print(f"\nRecommendation:")
    if without['perfect_match_percentage'] > with_special['perfect_match_percentage']:
        print("✅ Use 'without_special' mode for better roundtrip fidelity")
    elif with_special['perfect_match_percentage'] > without['perfect_match_percentage']:
        print("✅ Use 'with_special' mode for better roundtrip fidelity")
    else:
        print("ℹ️ Both modes perform similarly")

def save_detailed_report(results: Dict[str, any], output_file: str):
    """Save detailed report to file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("TOKENIZER ROUNDTRIP TEST DETAILED REPORT\n")
        f.write("="*60 + "\n\n")
        
        if 'test_mode' in results:
            f.write(f"Test mode: {results['test_mode']}\n\n")
        
        total_processed = results['perfect_matches'] + results['imperfect_matches']
        f.write(f"Total texts processed: {total_processed}\n")
        f.write(f"Perfect matches: {results['perfect_matches']} ({results.get('perfect_match_percentage', 0):.2f}%)\n")
        f.write(f"Imperfect matches: {results['imperfect_matches']} ({results.get('imperfect_match_percentage', 0):.2f}%)\n")
        f.write(f"Failed roundtrips: {results['failed_roundtrips']}\n")
        f.write(f"Texts with whitespace issues: {results['whitespace_issues']}\n\n")
        
        f.write("ALL PROBLEMATIC EXAMPLES:\n")
        f.write("-" * 60 + "\n")
        
        for example in results['problematic_examples']:
            f.write(f"\nExample {example['index']} (Similarity: {example['similarity']:.4f}):\n")
            f.write(f"ORIGINAL: {example['original']}\n")
            f.write(f"DECODED:  {example['decoded']}\n")
            f.write(f"Analysis: {example['analysis']}\n")
            f.write("-" * 40 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Test tokenizer roundtrip fidelity (EOS-aware)')
    
    parser.add_argument('--tokenizer-path', type=str, 
                        default='../models/30k/',
                        help='Path to tokenizer model')
    
    parser.add_argument('--dataset-name', type=str, 
                        default='TucanoBR/wikipedia-pt',
                        help='Dataset name to test')
    
    parser.add_argument('--dataset-split', type=str, 
                        default='train[:1000]',
                        help='Dataset split to use')
    
    parser.add_argument('--text-column', type=str, 
                        default='text',
                        help='Column name containing text data')
    
    parser.add_argument('--cache-dir', type=str, 
                        default='../data',
                        help='Directory to cache dataset')
    
    parser.add_argument('--max-length', type=int, 
                        default=512,
                        help='Maximum sequence length')
    
    parser.add_argument('--output-file', type=str, 
                        default=None,
                        help='File to save detailed report')
    
    parser.add_argument('--test-mode', type=str, 
                        choices=['without_special', 'with_special', 'compare'],
                        default='without_special',
                        help='Test mode: without_special (no EOS), with_special (EOS cleaned), or compare (both modes)')
    
    args = parser.parse_args()
    
    if args.test_mode == 'compare':
        # Run comparative test
        comparison = test_both_modes(
            tokenizer_path=args.tokenizer_path,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            text_column=args.text_column,
            cache_dir=args.cache_dir,
            max_length=args.max_length
        )
        print_comparison_results(comparison)
        
    else:
        # Run single mode test
        results = test_tokenizer_roundtrip(
            tokenizer_path=args.tokenizer_path,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            text_column=args.text_column,
            cache_dir=args.cache_dir,
            max_length=args.max_length,
            output_file=args.output_file,
            test_mode=args.test_mode
        )
        
        print_results(results)
        
        if args.output_file:
            save_detailed_report(results, args.output_file)
            print(f"\nDetailed report saved to: {args.output_file}")

if __name__ == "__main__":
    main()

# Example usage:
# python "tokenizer_test.py" --tokenizer-path "../models/30k/" --test-mode "without_special"
# python "tokenizer_test.py" --tokenizer-path "../models/30k/" --test-mode "with_special" 
# python "tokenizer_test.py" --tokenizer-path "../models/30k/" --test-mode "compare"
#        --dataset-name "TucanoBR/wikipedia-pt" --dataset-split "train[:1000]" 
#        --output-file "tokenizer_report.txt"