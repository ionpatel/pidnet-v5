"""
Train a small BPE tokenizer on our corpus.

Why small vocab (1024-4096)?
- On 8GB MacBook, 50K vocab means embedding dominates (12.9M of 14.4M params)
- The PID core only gets ~3.4M params — can't learn 50K token combinations
- With 1024 vocab: embedding = 393K params, PID core gets full capacity
- Small BPE still captures subword structure (common prefixes, suffixes, words)
- Best of both: subword tokenization without vocabulary overhead

Usage:
    python train/train_tokenizer.py --input data/owt_sample_50mb.txt --vocab-size 1024
    # Produces: data/tokenizer_1024.model
"""

import os
import argparse


def train_tokenizer(input_path: str, vocab_size: int = 1024, model_prefix: str = None):
    """Train a SentencePiece BPE tokenizer."""
    try:
        import sentencepiece as spm
    except ImportError:
        print("Installing sentencepiece...")
        os.system("pip install sentencepiece")
        import sentencepiece as spm
    
    if model_prefix is None:
        data_dir = os.path.dirname(input_path) or "data"
        model_prefix = os.path.join(data_dir, f"tokenizer_{vocab_size}")
    
    print(f"Training BPE tokenizer: vocab_size={vocab_size}")
    print(f"Input: {input_path}")
    print(f"Output: {model_prefix}.model")
    
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=0.9995,
        num_threads=os.cpu_count() or 4,
        # Special tokens
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # Training params
        input_sentence_size=1000000,  # max sentences to sample
        shuffle_input_sentence=True,
        max_sentence_length=4192,
        byte_fallback=True,  # handle any byte via fallback
    )
    
    # Verify
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    
    test = "The United States government announced new policies."
    encoded = sp.encode(test)
    decoded = sp.decode(encoded)
    pieces = sp.encode(test, out_type=str)
    
    print(f"\nTokenizer trained!")
    print(f"Vocab size: {sp.get_piece_size()}")
    print(f"Test: '{test}'")
    print(f"Tokens: {pieces}")
    print(f"IDs: {encoded}")
    print(f"Decoded: '{decoded}'")
    print(f"Compression: {len(test)} chars → {len(encoded)} tokens ({len(test)/len(encoded):.1f}x)")
    
    return f"{model_prefix}.model"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--input", type=str, required=True, help="Input text file")
    parser.add_argument("--vocab-size", type=int, default=1024, help="Vocabulary size")
    parser.add_argument("--output", type=str, default=None, help="Model prefix")
    args = parser.parse_args()
    
    train_tokenizer(args.input, args.vocab_size, args.output)
