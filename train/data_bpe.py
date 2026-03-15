"""
BPE/Subword data loading for PID-Net v5 — Phase 2.

Uses tiktoken (GPT-2 tokenizer) for subword tokenization.
Supports OpenWebText, FineWeb, or any text corpus.
"""

import mlx.core as mx
import os
import urllib.request
import numpy as np
from typing import Tuple, Optional


def get_tokenizer():
    """Get tiktoken GPT-2 tokenizer."""
    try:
        import tiktoken
        return tiktoken.get_encoding("gpt2")
    except ImportError:
        print("Installing tiktoken...")
        os.system("pip install tiktoken")
        import tiktoken
        return tiktoken.get_encoding("gpt2")


class BPEDataset:
    """BPE-tokenized dataset for language modeling.
    
    Loads text, tokenizes with tiktoken (GPT-2 BPE, 50257 vocab),
    and provides batched sequences.
    """
    
    def __init__(self, token_ids: list, seq_len: int = 256, name: str = "train"):
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.vocab_size = 50257  # GPT-2 vocab
        self.name = name
        
        print(f"[{name}] Tokens: {len(self.token_ids):,} | Sequences: ~{len(self.token_ids) // seq_len:,}")
    
    def get_batch(self, batch_size: int) -> Tuple[mx.array, mx.array]:
        """Get a random batch of sequences."""
        max_start = len(self.token_ids) - self.seq_len - 1
        starts = [int(mx.random.randint(0, max_start).item()) for _ in range(batch_size)]
        
        inputs = []
        targets = []
        for s in starts:
            inputs.append(self.token_ids[s : s + self.seq_len])
            targets.append(self.token_ids[s + 1 : s + self.seq_len + 1])
        
        return mx.array(inputs), mx.array(targets)
    
    def encode(self, text: str) -> list:
        """Encode text to token IDs."""
        enc = get_tokenizer()
        return enc.encode(text)
    
    def decode(self, indices) -> str:
        """Decode token IDs to text."""
        enc = get_tokenizer()
        if hasattr(indices, 'tolist'):
            indices = indices.tolist()
        return enc.decode(indices)


def load_text_file(path: str, seq_len: int = 256, val_split: float = 0.05) -> Tuple[BPEDataset, BPEDataset]:
    """Load any text file with BPE tokenization."""
    enc = get_tokenizer()
    
    print(f"Loading and tokenizing {path}...")
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    print(f"Text length: {len(text):,} chars")
    tokens = enc.encode(text)
    print(f"Tokens: {len(tokens):,} (compression ratio: {len(text)/len(tokens):.1f}x)")
    
    # Split
    split_idx = int(len(tokens) * (1 - val_split))
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    
    return BPEDataset(train_tokens, seq_len, "train"), BPEDataset(val_tokens, seq_len, "val")


def download_openwebtext_sample(data_dir: str = "data", max_mb: int = 100) -> str:
    """Download a sample of OpenWebText for training.
    
    Uses the HuggingFace datasets library to stream a subset.
    """
    os.makedirs(data_dir, exist_ok=True)
    text_path = os.path.join(data_dir, f"owt_sample_{max_mb}mb.txt")
    
    if os.path.exists(text_path):
        return text_path
    
    try:
        from datasets import load_dataset
        print(f"Downloading OpenWebText sample ({max_mb}MB)...")
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        
        total_chars = 0
        target_chars = max_mb * 1_000_000  # rough MB to chars
        
        with open(text_path, 'w', encoding='utf-8') as f:
            for example in ds:
                text = example['text']
                f.write(text + '\n\n')
                total_chars += len(text)
                if total_chars >= target_chars:
                    break
        
        print(f"Downloaded {total_chars:,} chars → {text_path}")
        return text_path
        
    except ImportError:
        print("Install datasets: pip install datasets")
        raise


def load_openwebtext(seq_len: int = 256, max_mb: int = 100, val_split: float = 0.05) -> Tuple[BPEDataset, BPEDataset]:
    """Load OpenWebText sample with BPE tokenization."""
    path = download_openwebtext_sample(max_mb=max_mb)
    return load_text_file(path, seq_len, val_split)


def load_tinystories_bpe(seq_len: int = 256, max_stories: int = 50000, val_split: float = 0.05) -> Tuple[BPEDataset, BPEDataset]:
    """Load TinyStories with BPE tokenization."""
    from data import download_tinystories
    path = download_tinystories(max_stories=max_stories)
    return load_text_file(path, seq_len, val_split)


class SPDataset:
    """SentencePiece-tokenized dataset for language modeling.
    
    Uses a custom-trained small BPE tokenizer (1K-4K vocab).
    """
    
    def __init__(self, token_ids: list, vocab_size: int, sp_model_path: str,
                 seq_len: int = 256, name: str = "train"):
        self.token_ids = token_ids
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.name = name
        self._sp_path = sp_model_path
        self._sp = None  # lazy load
        
        print(f"[{name}] Tokens: {len(self.token_ids):,} | Sequences: ~{len(self.token_ids) // seq_len:,} | Vocab: {vocab_size}")
    
    def _get_sp(self):
        if self._sp is None:
            import sentencepiece as spm
            self._sp = spm.SentencePieceProcessor()
            self._sp.load(self._sp_path)
        return self._sp
    
    def get_batch(self, batch_size: int) -> Tuple[mx.array, mx.array]:
        max_start = len(self.token_ids) - self.seq_len - 1
        starts = [int(mx.random.randint(0, max_start).item()) for _ in range(batch_size)]
        
        inputs = []
        targets = []
        for s in starts:
            inputs.append(self.token_ids[s : s + self.seq_len])
            targets.append(self.token_ids[s + 1 : s + self.seq_len + 1])
        
        return mx.array(inputs), mx.array(targets)
    
    def encode(self, text: str) -> list:
        return self._get_sp().encode(text)
    
    def decode(self, indices) -> str:
        sp = self._get_sp()
        if hasattr(indices, 'tolist'):
            indices = indices.tolist()
        return sp.decode(indices)


def load_with_sp_tokenizer(text_path: str, tokenizer_path: str, seq_len: int = 256, 
                            val_split: float = 0.05) -> Tuple[SPDataset, SPDataset]:
    """Load text file with a custom SentencePiece tokenizer."""
    import sentencepiece as spm
    
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    vocab_size = sp.get_piece_size()
    
    print(f"Loading {text_path} with {tokenizer_path} (vocab={vocab_size})...")
    with open(text_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    print(f"Text: {len(text):,} chars")
    tokens = sp.encode(text)
    print(f"Tokens: {len(tokens):,} (compression: {len(text)/len(tokens):.1f}x)")
    
    split_idx = int(len(tokens) * (1 - val_split))
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    
    return (
        SPDataset(train_tokens, vocab_size, tokenizer_path, seq_len, "train"),
        SPDataset(val_tokens, vocab_size, tokenizer_path, seq_len, "val"),
    )


# Tokenized binary cache for faster loading
def save_tokens(tokens: list, path: str):
    """Save tokenized data as numpy binary for fast loading."""
    arr = np.array(tokens, dtype=np.uint16)  # GPT-2 vocab fits in uint16
    np.save(path, arr)
    print(f"Saved {len(tokens):,} tokens to {path}")


def load_tokens(path: str) -> list:
    """Load pre-tokenized binary data."""
    arr = np.load(path)
    print(f"Loaded {len(arr):,} tokens from {path}")
    return arr.tolist()
