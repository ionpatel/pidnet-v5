"""
PID-Net v5 Training Script — Phase 1

Key principles:
1. ALWAYS show generation samples (lesson from v1-v3: never trust loss alone)
2. Log PID gate values every epoch (catch stream death early)
3. Check for repetition collapse explicitly
4. Energy conservation monitoring
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
import argparse
import os
import sys

# Add parent dir to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from pidnet.model import PIDGraphNet, count_parameters
from data import load_shakespeare


def check_collapse(text: str, threshold: int = 5) -> bool:
    """Check if generated text has repetition collapse.
    
    Returns True if collapsed (BAD).
    """
    # Check for repeated characters
    for i in range(len(text) - threshold):
        if len(set(text[i:i+threshold])) == 1:
            return True
    
    # Check for repeated n-grams
    for n in [3, 4, 5]:
        ngrams = [text[i:i+n] for i in range(len(text) - n)]
        if len(ngrams) > 10:
            unique_ratio = len(set(ngrams)) / len(ngrams)
            if unique_ratio < 0.1:  # >90% repeated
                return True
    
    return False


def train(args):
    """Main training loop."""
    
    print("=" * 60)
    print("PID-Net v5: Hypergraph Rewriting Architecture")
    print("Phase 1: Foundation — Fixed Graph + PID Rewriting")
    print("=" * 60)
    
    # Load data
    print("\n📚 Loading data...")
    train_data, val_data = load_shakespeare(seq_len=args.seq_len)
    
    # Create model
    print("\n🧠 Building model...")
    model = PIDGraphNet(
        vocab_size=train_data.vocab_size,
        d_model=args.d_model,
        max_nodes=args.seq_len + 16,  # buffer for graph
        n_rewrite_steps=args.n_rewrite_steps,
        connect_k=args.connect_k,
    )
    
    # Count parameters
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")
    print(f"d_model: {args.d_model}")
    print(f"Rewrite steps: {args.n_rewrite_steps}")
    print(f"Connect-k: {args.connect_k}")
    print(f"Seq length: {args.seq_len}")
    
    # Optimizer
    lr_schedule = optim.cosine_decay(args.lr, args.steps)
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=args.weight_decay)
    
    # Loss function
    def loss_fn(model, inputs, targets):
        logits, diagnostics = model(inputs)
        # Cross-entropy loss
        # logits: [batch, seq_len, vocab]
        # targets: [batch, seq_len]
        ce_loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        ce_loss = mx.mean(ce_loss)
        
        # Prediction loss: train the D-stream predictor
        # Without this, the predictor never learns → D-gate dies
        # Weight 0.01 — enough to train predictor without overwhelming CE loss
        pred_loss = diagnostics.get('pred_loss', mx.array(0.0))
        
        # Skip penalty: discourage lazy skipping
        skip_val = diagnostics.get('skip', mx.array(0.0))
        skip_penalty = skip_val * 0.05  # gentle push toward engaging
        
        loss = ce_loss + 0.01 * pred_loss + skip_penalty
        
        # Track CE loss separately for monitoring
        diagnostics['ce_loss'] = ce_loss
        
        return loss, diagnostics
    
    # Compile loss + grad
    loss_and_grad_fn = nn.value_and_grad(model, lambda m, x, y: loss_fn(m, x, y)[0])
    
    # Training loop
    print(f"\n🚀 Training for {args.steps} steps...")
    print("-" * 60)
    
    best_loss = float('inf')
    collapse_count = 0
    
    # Test prompts for generation
    test_prompts = [
        "To be or not to be",
        "ROMEO: ",
        "The king",
        "What is",
        "Once upon",
    ]
    
    for step in range(1, args.steps + 1):
        t0 = time.time()
        
        # Get batch
        inputs, targets = train_data.get_batch(args.batch_size)
        
        # Forward + backward
        (loss, grads) = loss_and_grad_fn(model, inputs, targets)
        
        # Get diagnostics (separate forward pass — cheap since MLX is lazy)
        _, diagnostics = loss_fn(model, inputs, targets)
        
        # Update
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        
        dt = time.time() - t0
        tokens_per_sec = args.batch_size * args.seq_len / dt
        
        # Log every N steps
        if step % args.log_every == 0 or step == 1:
            loss_val = loss.item()
            bpb = loss_val / 0.6931  # bits per byte
            
            # Gate values
            gp = diagnostics['gate_p'].item()
            gi = diagnostics['gate_i'].item()
            gd = diagnostics['gate_d'].item()
            skip = diagnostics['skip'].item()
            pred_err = diagnostics['pred_error'].item()
            energy = diagnostics['energy_ratio'].item()
            
            ce_val = diagnostics.get('ce_loss', loss).item() if 'ce_loss' in diagnostics else loss_val
            ce_bpb = ce_val / 0.6931
            
            edge_dens = diagnostics.get('edge_density', mx.array(0.0)).item()
            
            print(f"\nStep {step}/{args.steps} | Loss: {loss_val:.4f} | CE: {ce_val:.4f} | BPB: {ce_bpb:.3f} | {tokens_per_sec:.0f} tok/s")
            print(f"  Gates: P={gp:.3f} I={gi:.3f} D={gd:.3f} | Skip={skip:.3f}")
            print(f"  Pred Error: {pred_err:.4f} | Energy: {energy:.4f} | Edges: {edge_dens:.4f}")
            
            # Warn if any gate is dying
            for name, val in [('P', gp), ('I', gi), ('D', gd)]:
                if val < 0.08:
                    print(f"  ⚠️  {name}-gate is low ({val:.3f}) — may be dying!")
            
            if loss_val < best_loss:
                best_loss = loss_val
        
        # Generate samples every N steps
        if step % args.generate_every == 0:
            print(f"\n{'='*40} Generation Samples {'='*40}")
            
            step_collapse = False
            for prompt_text in test_prompts:
                prompt_tokens = mx.array([train_data.encode(prompt_text)])
                
                try:
                    generated = model.generate(
                        prompt_tokens,
                        max_new_tokens=args.generate_len,
                        temperature=args.temperature,
                    )
                    full_text = train_data.decode(generated)
                    
                    collapsed = check_collapse(full_text[len(prompt_text):])
                    status = "🔴 COLLAPSE" if collapsed else "🟢 OK"
                    if collapsed:
                        step_collapse = True
                    
                    # Truncate for display
                    display = full_text[:120]
                    if len(full_text) > 120:
                        display += "..."
                    print(f"  [{status}] \"{display}\"")
                    
                except Exception as e:
                    print(f"  [❌ ERROR] Prompt \"{prompt_text}\": {e}")
            
            if step_collapse:
                collapse_count += 1
                print(f"\n  ⚠️  Collapse detected! ({collapse_count} total)")
            else:
                print(f"\n  ✅ All prompts OK")
            
            print(f"{'='*80}\n")
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Collapse events: {collapse_count}")
    print(f"  Parameters: {n_params:,}")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PID-Net v5 Training")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n-rewrite-steps", type=int, default=3, help="PID rewriting rounds per token")
    parser.add_argument("--connect-k", type=int, default=8, help="Initial edges per new node")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--generate-len", type=int, default=100, help="Generation length")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--generate-every", type=int, default=50, help="Generate every N steps")
    
    args = parser.parse_args()
    train(args)
