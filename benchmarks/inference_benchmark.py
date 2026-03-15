"""
Benchmark heterogeneous inference engine on MacBook Air M2.

Compares:
1. Standard generation (model.generate)
2. Heterogeneous engine (tiered compute + D-stream prefetch)

Usage:
    python benchmarks/inference_benchmark.py --load model_5k.safetensors
    python benchmarks/inference_benchmark.py --load model_owt_sp1k_50k.safetensors --dataset openwebtext --tokenizer data/tokenizer_1024.model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))

import mlx.core as mx
import time
import argparse
import json
from pidnet.fractal import FractalPIDNet
from pidnet.model import count_parameters
from pidnet.inference import HeterogeneousEngine, TieredMemoryConfig


def load_model(args):
    """Load model and dataset."""
    # Load dataset for tokenizer
    if args.tokenizer:
        from data_bpe import load_with_sp_tokenizer, download_openwebtext_sample
        if args.dataset == "openwebtext":
            text_path = download_openwebtext_sample(max_mb=args.max_mb)
        else:
            text_path = args.dataset
        train_data, _ = load_with_sp_tokenizer(text_path, args.tokenizer)
        vocab_size = train_data.vocab_size
    else:
        from data import load_shakespeare
        train_data, _ = load_shakespeare(seq_len=args.seq_len)
        vocab_size = train_data.vocab_size
    
    # Build model
    model = FractalPIDNet(
        vocab_size=vocab_size,
        d_model=args.d_model,
        max_nodes=args.seq_len,
        n_rewrite_steps=args.rewrite_steps,
        chunk_size=args.chunk_size,
        n_levels=args.n_levels,
    )
    
    # Load weights
    if args.load:
        import mlx.nn as nn
        weights = mx.load(args.load)
        # Flatten nested weight keys
        flat = {}
        for k, v in weights.items():
            flat[k] = v
        model.load_weights(list(flat.items()))
        print(f"✅ Loaded weights from {args.load}")
    
    params = count_parameters(model)
    print(f"Model: {params:,} parameters")
    
    return model, train_data


def benchmark_standard(model, prompt_tokens, n_tokens, n_runs=5):
    """Benchmark standard model.generate()."""
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        output = model.generate(prompt_tokens, max_new_tokens=n_tokens)
        mx.eval(output)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    
    avg_ms = sum(times) / len(times) * 1000
    tok_per_sec = n_tokens / (avg_ms / 1000)
    return avg_ms, tok_per_sec


def benchmark_heterogeneous(model, prompt_tokens, n_tokens, config, n_runs=5):
    """Benchmark HeterogeneousEngine."""
    times = []
    final_stats = None
    
    for i in range(n_runs):
        engine = HeterogeneousEngine(model, config)
        t0 = time.perf_counter()
        output = engine.generate(prompt_tokens, max_new_tokens=n_tokens)
        mx.eval(output)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        final_stats = engine.stats
    
    avg_ms = sum(times) / len(times) * 1000
    tok_per_sec = n_tokens / (avg_ms / 1000)
    return avg_ms, tok_per_sec, final_stats


def main():
    parser = argparse.ArgumentParser(description="Inference Benchmark")
    parser.add_argument("--load", type=str, default=None, help="Model weights file")
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--rewrite-steps", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--n-levels", type=int, default=3)
    parser.add_argument("--n-tokens", type=int, default=50, help="Tokens to generate")
    parser.add_argument("--n-runs", type=int, default=5, help="Runs for averaging")
    parser.add_argument("--dataset", type=str, default="shakespeare")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--max-mb", type=int, default=50)
    args = parser.parse_args()
    
    print("=" * 60)
    print("PID-Net Heterogeneous Inference Benchmark")
    print("=" * 60)
    
    model, train_data = load_model(args)
    
    # Prepare prompts
    prompts = ["The United States", "Once upon a time", "Scientists have"]
    
    for prompt_text in prompts:
        prompt_tokens = mx.array([train_data.encode(prompt_text)])
        
        print(f"\n{'─' * 60}")
        print(f"Prompt: '{prompt_text}' → {args.n_tokens} new tokens")
        print(f"{'─' * 60}")
        
        # Standard
        std_ms, std_tps = benchmark_standard(
            model, prompt_tokens, args.n_tokens, args.n_runs)
        
        # Heterogeneous
        config = TieredMemoryConfig(
            hot_levels=2,
            prefetch_threshold=0.5,
        )
        het_ms, het_tps, het_stats = benchmark_heterogeneous(
            model, prompt_tokens, args.n_tokens, config, args.n_runs)
        
        speedup = std_ms / het_ms if het_ms > 0 else 0
        
        print(f"  Standard:       {std_ms:7.1f}ms | {std_tps:7.1f} tok/s")
        print(f"  Heterogeneous:  {het_ms:7.1f}ms | {het_tps:7.1f} tok/s | {speedup:.2f}x")
        if het_stats:
            print(f"  Prefetch stats: {het_stats.summary()}")
        
        # Show generated text
        engine = HeterogeneousEngine(model, config)
        output = engine.generate(prompt_tokens, max_new_tokens=args.n_tokens)
        text = train_data.decode(output)
        print(f"  Output: \"{text[:120]}...\"")
    
    print(f"\n{'=' * 60}")
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
