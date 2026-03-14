"""
Scaled Benchmark: PID-Net Fractal vs Transformer at 5-10M params.

Tests whether the BPB advantage holds at scale.
Also measures throughput parity after vectorization.
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from pidnet.fractal import FractalPIDNet
from pidnet.model import count_parameters
from benchmarks.transformer_baseline import TransformerLM, count_params, check_collapse
from train.data import load_shakespeare


def train_model(model, train_data, args, model_name, is_fractal=False):
    """Train a model and return results."""
    
    n_params = count_params(model) if not is_fractal else count_parameters(model)
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"  Parameters: {n_params:,}")
    print(f"{'='*60}\n")
    
    lr_schedule = optim.cosine_decay(args.lr, args.steps)
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.01)
    
    if is_fractal:
        def loss_fn(model, inputs, targets):
            logits, diag = model(inputs)
            ce = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
            ce_mean = mx.mean(ce)
            pred_loss = diag.get('pred_loss', mx.array(0.0))
            return ce_mean + 0.01 * pred_loss, (ce_mean, diag)
    else:
        def loss_fn(model, inputs, targets):
            logits = model(inputs)
            ce = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
            ce_mean = mx.mean(ce)
            return ce_mean, (ce_mean, {})
    
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    
    test_prompts = [
        "To be or not to be",
        "ROMEO: ",
        "The king",
        "What is",
        "Once upon",
    ]
    
    best_loss = float('inf')
    results = {
        'name': model_name,
        'params': n_params,
        'best_loss': float('inf'),
        'best_bpb': float('inf'),
        'final_tps': 0,
        'collapse_count': 0,
    }
    
    for step in range(1, args.steps + 1):
        t0 = time.time()
        inputs, targets = train_data.get_batch(args.batch_size)
        (total_loss, (ce, diag)), grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, total_loss, ce)
        dt = time.time() - t0
        tps = args.batch_size * args.seq_len / dt
        
        ce_val = ce.item()
        if ce_val < best_loss:
            best_loss = ce_val
            results['best_loss'] = best_loss
            results['best_bpb'] = best_loss / 0.6931
        results['final_tps'] = tps
        
        if step % args.log_every == 0 or step == 1:
            bpb = ce_val / 0.6931
            line = f"Step {step}/{args.steps} | Loss: {ce_val:.4f} | BPB: {bpb:.3f} | {tps:.0f} tok/s"
            if is_fractal and diag:
                gp = diag.get('gate_p', mx.array(0.0)).item()
                gi = diag.get('gate_i', mx.array(0.0)).item()
                gd = diag.get('gate_d', mx.array(0.0)).item()
                pe = diag.get('pred_error', mx.array(0.0)).item()
                line += f"\n Gates: P={gp:.3f} I={gi:.3f} D={gd:.3f} | Pred Error: {pe:.4f}"
            print(line)
        
        if step % args.generate_every == 0:
            print(f"\n{'='*40} Generation ({model_name}) {'='*40}")
            for prompt_text in test_prompts:
                prompt_tokens = mx.array([train_data.encode(prompt_text)])
                try:
                    generated = model.generate(prompt_tokens, max_new_tokens=100, temperature=0.8)
                    if isinstance(generated, mx.array):
                        full_text = train_data.decode(generated)
                    else:
                        full_text = train_data.decode(generated)
                    collapsed = check_collapse(full_text[len(prompt_text):])
                    status = "🔴 COLLAPSE" if collapsed else "🟢 OK"
                    if collapsed:
                        results['collapse_count'] += 1
                    display = full_text[:120] + ("..." if len(full_text) > 120 else "")
                    print(f"  [{status}] \"{display}\"")
                except Exception as e:
                    print(f"  [❌ ERROR] \"{prompt_text}\": {e}")
            print(f"{'='*80}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Scaled PID-Net vs Transformer benchmark")
    parser.add_argument("--scale", type=str, default="5M", choices=["5M", "10M"],
                        help="Parameter scale target")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--generate-every", type=int, default=1000)
    parser.add_argument("--model", type=str, default="both", choices=["pidnet", "transformer", "both"])
    args = parser.parse_args()
    
    # Load data
    train_data, val_data = load_shakespeare(seq_len=args.seq_len)
    
    # Scale configs
    if args.scale == "5M":
        # ~5M params each
        pidnet_config = {
            'd_model': 384,
            'n_rewrite_steps': 3,
            'connect_k': 8,
            'chunk_size': 16,
            'n_levels': 3,
        }
        transformer_config = {
            'd_model': 384,
            'n_heads': 6,
            'n_layers': 6,
            'd_ff': 1024,
        }
    else:  # 10M
        pidnet_config = {
            'd_model': 512,
            'n_rewrite_steps': 3,
            'connect_k': 8,
            'chunk_size': 16,
            'n_levels': 3,
        }
        transformer_config = {
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 1536,
        }
    
    results = []
    
    # === PID-Net Fractal ===
    if args.model in ("pidnet", "both"):
        pidnet = FractalPIDNet(
            vocab_size=train_data.vocab_size,
            d_model=pidnet_config['d_model'],
            max_nodes=args.seq_len + 16,
            n_rewrite_steps=pidnet_config['n_rewrite_steps'],
            connect_k=pidnet_config['connect_k'],
            chunk_size=pidnet_config['chunk_size'],
            n_levels=pidnet_config['n_levels'],
        )
        r = train_model(pidnet, train_data, args, f"PID-Net Fractal ({args.scale})", is_fractal=True)
        results.append(r)
        del pidnet  # free memory
    
    # === Transformer ===
    if args.model in ("transformer", "both"):
        transformer = TransformerLM(
            vocab_size=train_data.vocab_size,
            d_model=transformer_config['d_model'],
            n_heads=transformer_config['n_heads'],
            n_layers=transformer_config['n_layers'],
            d_ff=transformer_config['d_ff'],
            max_seq_len=args.seq_len + 16,
        )
        r = train_model(transformer, train_data, args, f"Transformer ({args.scale})", is_fractal=False)
        results.append(r)
    
    # === COMPARISON ===
    print(f"\n{'='*70}")
    print(f"  SCALED BENCHMARK RESULTS ({args.scale} params, {args.steps} steps)")
    print(f"{'='*70}")
    for r in results:
        print(f"\n  {r['name']}:")
        print(f"    Parameters:  {r['params']:,}")
        print(f"    Best Loss:   {r['best_loss']:.4f}")
        print(f"    Best BPB:    {r['best_bpb']:.3f}")
        print(f"    Speed:       {r['final_tps']:.0f} tok/s")
        print(f"    Collapses:   {r['collapse_count']}")
    
    if len(results) == 2:
        bpb_diff = (results[1]['best_bpb'] - results[0]['best_bpb']) / results[1]['best_bpb'] * 100
        speed_ratio = results[1]['final_tps'] / max(results[0]['final_tps'], 1)
        print(f"\n  BPB Δ: {bpb_diff:+.1f}% ({'PID-Net wins' if bpb_diff > 0 else 'Transformer wins'})")
        print(f"  Speed ratio: Transformer {speed_ratio:.1f}× faster")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
