"""
Data loading for PID-Net v5.

Phase 1: Character-level Shakespeare (same as v3 for comparison).
Later phases: TinyStories, OpenWebText, etc.
"""

import mlx.core as mx
import os
import urllib.request
from typing import Tuple


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_shakespeare(data_dir: str = "data") -> str:
    """Download tiny Shakespeare dataset."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "shakespeare.txt")
    if not os.path.exists(path):
        print("Downloading Shakespeare...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
        print(f"Downloaded to {path}")
    return path


class CharDataset:
    """Character-level dataset for language modeling."""
    
    def __init__(self, text: str, seq_len: int = 128):
        self.text = text
        self.seq_len = seq_len
        
        # Build vocabulary
        chars = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        
        # Encode full text
        self.encoded = [self.char_to_idx[c] for c in text]
        
        print(f"Vocab size: {self.vocab_size}")
        print(f"Text length: {len(self.encoded):,} chars")
        print(f"Sequences: ~{len(self.encoded) // seq_len:,}")
    
    def encode(self, text: str) -> list:
        return [self.char_to_idx.get(c, 0) for c in text]
    
    def decode(self, indices) -> str:
        if hasattr(indices, 'tolist'):
            indices = indices.tolist()
        return ''.join(self.idx_to_char.get(i, '?') for i in indices)
    
    def get_batch(self, batch_size: int) -> Tuple[mx.array, mx.array]:
        """Get a random batch of sequences.
        
        Returns:
            (inputs [batch, seq_len], targets [batch, seq_len])
            where targets = inputs shifted by 1
        """
        max_start = len(self.encoded) - self.seq_len - 1
        starts = [mx.random.randint(0, max_start).item() for _ in range(batch_size)]
        
        inputs = []
        targets = []
        for s in starts:
            inputs.append(self.encoded[s : s + self.seq_len])
            targets.append(self.encoded[s + 1 : s + self.seq_len + 1])
        
        return mx.array(inputs), mx.array(targets)


def load_shakespeare(seq_len: int = 128, val_split: float = 0.1) -> Tuple[CharDataset, CharDataset]:
    """Load Shakespeare dataset split into train/val."""
    path = download_shakespeare()
    with open(path, 'r') as f:
        text = f.read()
    
    split_idx = int(len(text) * (1 - val_split))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Build vocab from full text, but create separate datasets
    full_dataset = CharDataset(text, seq_len)
    
    train_dataset = CharDataset(train_text, seq_len)
    # Ensure val uses same vocab
    train_dataset.char_to_idx = full_dataset.char_to_idx
    train_dataset.idx_to_char = full_dataset.idx_to_char
    train_dataset.vocab_size = full_dataset.vocab_size
    train_dataset.encoded = [full_dataset.char_to_idx[c] for c in train_text]
    
    val_dataset = CharDataset(val_text, seq_len)
    val_dataset.char_to_idx = full_dataset.char_to_idx
    val_dataset.idx_to_char = full_dataset.idx_to_char
    val_dataset.vocab_size = full_dataset.vocab_size
    val_dataset.encoded = [full_dataset.char_to_idx[c] for c in val_text]
    
    return train_dataset, val_dataset
