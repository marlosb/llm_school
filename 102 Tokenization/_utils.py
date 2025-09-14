import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parallel tokenization script \
                                                   for datasets')
    
    # Dataset arguments
    parser.add_argument('--dataset-name', 
                        type=str, default='TucanoBR/wikipedia-PT',
                        help='Dataset name to load (default: \
                              TucanoBR/wikipedia-PT)')
    
    parser.add_argument('--dataset-split', 
                        type=str, default='train[:10]',
                        help='Dataset split to use (default: train[:10])')
    
    parser.add_argument('--text-column', type=str, default='text',
                        help='Column name containing text data \
                             (default: text)')
    
    parser.add_argument('--cache-dir', type=str, default='../data',
                        help='Directory to cache dataset \
                              (default: ../data)')
    
    # Tokenizer arguments
    parser.add_argument('--tokenizer-path', type=str, 
                        default='../models/30k.json',
                        help='Path to tokenizer model \
                            (default: ../models/30k.json)')
    
    parser.add_argument('--max-length', type=int, 
                        default=512,
                        help='Maximum sequence length for tokenization \
                              (default: 512)')
    
    # Processing arguments
    parser.add_argument('--batch-size', type=int, 
                        default=100000,
                        help='Batch size for processing (default: 100000)')
    
    parser.add_argument('--num-workers', type=int, 
                        default=4,
                        help='Number of worker processes (default: 4)')
    
    # Output arguments
    parser.add_argument('--output-path', type=str, 
                        default='../data/tokenized/',
                        help='Output directory for tokenized data \
                             (default: ../data/tokenized/)')
    
    return parser.parse_args()