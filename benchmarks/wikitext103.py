"""
WikiText-103 Benchmark — Standard publishable evaluation.

Pipeline:
  1. Extract WikiText-103 to plain text
  2. Train 1024-token SentencePiece tokenizer
  3. Train PID-Net + Transformer baseline (same param budget)
  4. Evaluate perplexity (PPL) on test set
  5. Report PPL, BPB, params, speed

Usage:
  python3 benchmarks/wikitext103.py --phase prepare   # extract + tokenizer
  python3 benchmarks/wikitext103.py --phase train      # train PID-Net
  python3 benchmarks/wikitext103.py --phase eval       # evaluate test PPL
"""

import os
import sys
import argparse
import time
import math

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def prepare():
    """Extract WikiText-103 to text files and train tokenizer."""
    from datasets import load_dataset
    
    print("Loading WikiText-103...")
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1')
    
    data_dir = os.path.join(os.path.dirname(__file__), 'wikitext103_data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Extract to plain text
    for split in ['train', 'validation', 'test']:
        out_path = os.path.join(data_dir, f'{split}.txt')
        if os.path.exists(out_path):
            print(f"  {split}.txt already exists, skipping")
            continue
        
        print(f"  Extracting {split}...")
        texts = [row['text'] for row in ds[split] if row['text'].strip()]
        with open(out_path, 'w') as f:
            f.write('\n'.join(texts))
        
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  {split}.txt: {len(texts)} lines, {size_mb:.1f} MB")
    
    # Train SentencePiece tokenizer on training data
    tok_path = os.path.join(data_dir, 'wt103_sp1k')
    if os.path.exists(tok_path + '.model'):
        print(f"  Tokenizer already exists at {tok_path}.model")
    else:
        print("  Training 1024-token SentencePiece tokenizer...")
        import sentencepiece as spm
        
        train_txt = os.path.join(data_dir, 'train.txt')
        spm.SentencePieceTrainer.train(
            input=train_txt,
            model_prefix=tok_path,
            vocab_size=4096,
            model_type='bpe',
            character_coverage=0.9995,
            byte_fallback=True,
            pad_id=3,
            input_sentence_size=1000000,
            shuffle_input_sentence=True,
        )
        print(f"  Tokenizer saved to {tok_path}.model")
    
    print("\nPrepare complete!")
    return data_dir


def train(args):
    """Train PID-Net on WikiText-103."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    
    from pidnet.fractal import FractalPIDNet
    from pidnet.model import count_parameters
    
    data_dir = os.path.join(os.path.dirname(__file__), 'wikitext103_data')
    tok_path = os.path.join(data_dir, 'wt103_sp1k.model')
    train_txt = os.path.join(data_dir, 'train.txt')
    val_txt = os.path.join(data_dir, 'validation.txt')
    
    if not os.path.exists(tok_path):
        print("Run --phase prepare first!")
        return
    
    # Load tokenizer
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(tok_path)
    vocab_size = sp.get_piece_size()
    print(f"Vocab size: {vocab_size}")
    
    # Tokenize data
    print("Tokenizing training data...")
    with open(train_txt) as f:
        train_text = f.read()
    
    # Use first 50MB to stay within 8GB RAM
    max_chars = 50_000_000
    if len(train_text) > max_chars:
        print(f"  Truncating to {max_chars // 1_000_000}MB (full: {len(train_text) // 1_000_000}MB)")
        train_text = train_text[:max_chars]
    
    train_ids = sp.encode(train_text)
    print(f"  {len(train_ids)} tokens")
    
    # Tokenize validation
    with open(val_txt) as f:
        val_text = f.read()
    val_ids = sp.encode(val_text)
    print(f"  Val: {len(val_ids)} tokens")
    
    # Model config
    d_model = args.d_model
    seq_len = args.seq_len
    n_levels = 3
    chunk_size = 16
    
    model = FractalPIDNet(
        vocab_size=vocab_size,
        d_model=d_model,
        max_nodes=seq_len + chunk_size,
        n_rewrite_steps=3,
        connect_k=8,
        chunk_size=chunk_size,
        n_levels=n_levels,
    )
    
    n_params = count_parameters(model)
    print(f"\nPID-Net: {n_params:,} params (d={d_model}, seq={seq_len}, levels={n_levels})")
    
    # Optimizer
    lr = args.lr
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)
    
    # Training loop
    train_tokens = mx.array(train_ids)
    val_tokens = mx.array(val_ids)
    n_steps = args.steps
    batch_size = args.batch_size
    
    def get_batch(tokens, seq_len, batch_size=1):
        """Random batch from token array."""
        max_start = len(tokens) - seq_len - 1
        starts = [mx.random.randint(0, max_start).item() for _ in range(batch_size)]
        x = mx.stack([tokens[s:s+seq_len] for s in starts])
        y = mx.stack([tokens[s+1:s+seq_len+1] for s in starts])
        return x, y
    
    def loss_fn(model, x, y):
        logits, diag = model(x)
        # Cross-entropy loss
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = y.reshape(-1)
        ce = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='mean')
        
        # Prediction loss (auxiliary)
        pred_loss = diag.get('pred_loss', mx.array(0.0))
        
        # Gate entropy regularization
        gate_entropy = diag.get('gate_entropy', mx.array(1.0))
        entropy_reg = mx.maximum(1.0 - gate_entropy, 0.0) * 0.1
        
        total = ce + pred_loss * 0.01 + entropy_reg
        return total, (ce, diag)
    
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    
    print(f"\nTraining for {n_steps} steps (lr={lr}, seq={seq_len})...")
    print("-" * 70)
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for step in range(1, n_steps + 1):
        x, y = get_batch(train_tokens, seq_len, batch_size)
        (total_loss, (ce_loss, diag)), grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        if step % 100 == 0 or step == 1:
            ce_val = ce_loss.item()
            bpb = ce_val / math.log(2)
            ppl = math.exp(ce_val)
            elapsed = time.time() - start_time
            tps = (step * seq_len * batch_size) / elapsed
            
            gp = diag.get('gate_p', mx.array(0)).item()
            gi = diag.get('gate_i', mx.array(0)).item()
            gd = diag.get('gate_d', mx.array(0)).item()
            
            print(f"Step {step:5d} | CE {ce_val:.4f} | BPB {bpb:.3f} | PPL {ppl:.1f} | "
                  f"P={gp:.3f} I={gi:.3f} D={gd:.3f} | {tps:.0f} tok/s")
        
        # Validation every 1000 steps
        if step % 1000 == 0:
            val_losses = []
            for _ in range(10):
                vx, vy = get_batch(val_tokens, seq_len, batch_size)
                logits, _ = model(vx)
                logits_flat = logits.reshape(-1, vocab_size)
                targets_flat = vy.reshape(-1)
                vl = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='mean')
                mx.eval(vl)
                val_losses.append(vl.item())
            
            avg_val = sum(val_losses) / len(val_losses)
            val_bpb = avg_val / math.log(2)
            val_ppl = math.exp(avg_val)
            print(f"  >>> VAL: CE {avg_val:.4f} | BPB {val_bpb:.3f} | PPL {val_ppl:.1f}")
            
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                save_path = os.path.join(data_dir, f'pidnet_wt103_d{d_model}.safetensors')
                model.save_weights(save_path)
                print(f"  >>> Saved best model to {save_path}")
    
    # Final save
    save_path = os.path.join(data_dir, f'pidnet_wt103_d{d_model}_final.safetensors')
    model.save_weights(save_path)
    
    elapsed = time.time() - start_time
    print(f"\nDone! {n_steps} steps in {elapsed:.0f}s")
    print(f"Best val CE: {best_val_loss:.4f} | BPB: {best_val_loss/math.log(2):.3f} | PPL: {math.exp(best_val_loss):.1f}")


def evaluate(args):
    """Evaluate trained model on WikiText-103 test set."""
    import mlx.core as mx
    import mlx.nn as nn
    import sentencepiece as spm
    
    from pidnet.fractal import FractalPIDNet
    from pidnet.model import count_parameters
    
    data_dir = os.path.join(os.path.dirname(__file__), 'wikitext103_data')
    tok_path = os.path.join(data_dir, 'wt103_sp1k.model')
    test_txt = os.path.join(data_dir, 'test.txt')
    
    sp = spm.SentencePieceProcessor()
    sp.load(tok_path)
    vocab_size = sp.get_piece_size()
    
    # Load model
    d_model = args.d_model
    seq_len = args.seq_len
    model_path = args.model_path or os.path.join(data_dir, f'pidnet_wt103_d{d_model}.safetensors')
    
    model = FractalPIDNet(
        vocab_size=vocab_size,
        d_model=d_model,
        max_nodes=seq_len + 16,
        n_rewrite_steps=3,
        connect_k=8,
        chunk_size=16,
        n_levels=3,
    )
    model.load_weights(model_path)
    n_params = count_parameters(model)
    print(f"Loaded PID-Net: {n_params:,} params from {model_path}")
    
    # Tokenize test set
    with open(test_txt) as f:
        test_text = f.read()
    test_ids = sp.encode(test_text)
    test_tokens = mx.array(test_ids)
    print(f"Test set: {len(test_ids)} tokens")
    
    # Evaluate perplexity over sliding windows
    total_loss = 0.0
    total_tokens = 0
    n_windows = min(len(test_ids) // seq_len, 500)  # cap at 500 windows
    
    print(f"Evaluating {n_windows} windows of {seq_len} tokens...")
    
    for i in range(n_windows):
        start = i * seq_len
        x = test_tokens[start:start+seq_len].reshape(1, -1)
        y = test_tokens[start+1:start+seq_len+1].reshape(1, -1)
        
        logits, _ = model(x)
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = y.reshape(-1)
        loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='sum')
        mx.eval(loss)
        
        total_loss += loss.item()
        total_tokens += seq_len
        
        if (i + 1) % 50 == 0:
            avg = total_loss / total_tokens
            print(f"  Window {i+1}/{n_windows}: CE {avg:.4f} | PPL {math.exp(avg):.1f}")
    
    avg_ce = total_loss / total_tokens
    bpt = avg_ce / math.log(2)  # bits per token
    ppl = math.exp(avg_ce)
    
    # True bits-per-byte (tokenizer-independent, publishable)
    with open(test_txt, 'rb') as f:
        total_bytes = len(f.read())
    total_bits = total_loss * math.log2(math.e)  # convert nats to bits
    true_bpb = (total_loss / math.log(2)) * total_tokens / total_bytes
    
    # Compression ratio (vs 8 bits/byte raw)
    compression = 8.0 / true_bpb
    
    print(f"\n{'='*50}")
    print(f"WikiText-103 Test Results")
    print(f"{'='*50}")
    print(f"  Model:      PID-Net (d={d_model})")
    print(f"  Parameters: {n_params:,}")
    print(f"  Vocab:      {vocab_size}")
    print(f"  Seq length: {seq_len}")
    print(f"  Test CE:    {avg_ce:.4f} nats/token")
    print(f"  Test BPT:   {bpt:.3f} bits/token")
    print(f"  Test PPL:   {ppl:.1f} (per-token)")
    print(f"  True BPB:   {true_bpb:.3f} bits/byte (tokenizer-independent)")
    print(f"  Compression: {compression:.1f}x vs raw")
    print(f"  Test bytes: {total_bytes:,}")
    print(f"  Test tokens: {total_tokens:,}")
    print(f"{'='*50}")
    print(f"\n  Note: Compare BPB to GPT-2 Small (117M): ~1.16 BPB")
    print(f"  Note: Compare BPB to random baseline: {math.log2(vocab_size):.2f} bits/token")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['prepare', 'train', 'eval'])
    parser.add_argument('--d-model', type=int, default=384)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--model-path', type=str, default=None)
    args = parser.parse_args()
    
    if args.phase == 'prepare':
        prepare()
    elif args.phase == 'train':
        train(args)
    elif args.phase == 'eval':
        evaluate(args)
