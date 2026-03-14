"""
Transformer Baseline for fair comparison against PID-Net v5.

Simple character-level Transformer with:
- Same vocab, same data, same training loop
- Matched parameter count (~1.4M)
- Standard causal attention + positional encoding

This is the benchmark to beat.
"""

import mlx.core as mx
import mlx.nn as nn
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.data import load_shakespeare


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
    
    def __call__(self, x, mask=None):
        # Pre-norm attention
        h = self.norm1(x)
        h = self.attn(h, h, h, mask=mask)
        x = x + h
        # Pre-norm FFN
        h = self.norm2(x)
        h = self.ff(h)
        x = x + h
        return x


class TransformerLM(nn.Module):
    """Character-level Transformer language model."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        self.layers = [
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]
        
        self.norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, vocab_size)
    
    def __call__(self, tokens):
        batch, seq_len = tokens.shape
        
        # Embed
        positions = mx.arange(seq_len)
        x = self.embed(tokens) + self.pos_embed(positions)
        
        # Causal mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        # Readout
        x = self.norm(x)
        logits = self.readout(x)
        
        return logits
    
    def generate(self, prompt_tokens, max_new_tokens=100, temperature=0.8, top_k=50):
        tokens = prompt_tokens.tolist()[0]
        generated = list(tokens)
        
        for _ in range(max_new_tokens):
            input_tokens = mx.array([generated])
            logits = self.__call__(input_tokens)
            
            last_logits = logits[0, -1] / temperature
            if top_k > 0:
                top_k_val = mx.sort(last_logits)[-top_k]
                last_logits = mx.where(last_logits < top_k_val, float('-inf'), last_logits)
            probs = mx.softmax(last_logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10)).item()
            generated.append(next_token)
        
        return mx.array(generated)


def count_params(model):
    total = 0
    def _count(p):
        nonlocal total
        if isinstance(p, mx.array):
            total += p.size
        elif isinstance(p, dict):
            for v in p.values():
                _count(v)
        elif isinstance(p, (list, tuple)):
            for v in p:
                _count(v)
    _count(model.parameters())
    return total


def check_collapse(text, threshold=5):
    for i in range(len(text) - threshold):
        if len(set(text[i:i+threshold])) == 1:
            return True
    for n in [3, 4, 5]:
        ngrams = [text[i:i+n] for i in range(len(text) - n)]
        if len(ngrams) > 10:
            if len(set(ngrams)) / len(ngrams) < 0.1:
                return True
    return False


def train_transformer(args):
    """Train a Transformer baseline for comparison."""
    
    print("=" * 60)
    print("TRANSFORMER BASELINE — Fair Comparison")
    print("=" * 60)
    
    # Load data (same as PID-Net)
    train_data, val_data = load_shakespeare(seq_len=args['seq_len'])
    
    # Create model (match ~1.4M params)
    model = TransformerLM(
        vocab_size=train_data.vocab_size,
        d_model=args['d_model'],
        n_heads=args['n_heads'],
        n_layers=args['n_layers'],
        d_ff=args['d_ff'],
        max_seq_len=args['seq_len'] + 16,
    )
    
    n_params = count_params(model)
    print(f"Parameters: {n_params:,}")
    print(f"d_model: {args['d_model']}, n_heads: {args['n_heads']}, n_layers: {args['n_layers']}, d_ff: {args['d_ff']}")
    
    # Optimizer (same as PID-Net)
    import mlx.optimizers as optim
    lr_schedule = optim.cosine_decay(args['lr'], args['steps'])
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.01)
    
    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        return mx.mean(loss)
    
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    
    # Test prompts
    test_prompts = [
        "To be or not to be",
        "ROMEO: ",
        "The king",
        "What is",
        "Once upon",
    ]
    
    print(f"\n🚀 Training for {args['steps']} steps...")
    best_loss = float('inf')
    
    for step in range(1, args['steps'] + 1):
        t0 = time.time()
        inputs, targets = train_data.get_batch(args['batch_size'])
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        dt = time.time() - t0
        tps = args['batch_size'] * args['seq_len'] / dt
        
        if step % args['log_every'] == 0 or step == 1:
            loss_val = loss.item()
            bpb = loss_val / 0.6931
            if loss_val < best_loss:
                best_loss = loss_val
            print(f"\nStep {step}/{args['steps']} | Loss: {loss_val:.4f} | BPB: {bpb:.3f} | {tps:.0f} tok/s")
        
        if step % args['generate_every'] == 0:
            print(f"\n{'='*40} Generation {'='*40}")
            for prompt_text in test_prompts:
                prompt_tokens = mx.array([train_data.encode(prompt_text)])
                try:
                    generated = model.generate(prompt_tokens, max_new_tokens=100, temperature=0.8)
                    full_text = train_data.decode(generated)
                    collapsed = check_collapse(full_text[len(prompt_text):])
                    status = "🔴 COLLAPSE" if collapsed else "🟢 OK"
                    display = full_text[:120] + ("..." if len(full_text) > 120 else "")
                    print(f"  [{status}] \"{display}\"")
                except Exception as e:
                    print(f"  [❌ ERROR] \"{prompt_text}\": {e}")
            print(f"{'='*80}\n")
    
    print(f"\n{'='*60}")
    print(f"TRANSFORMER BASELINE COMPLETE")
    print(f"  Best loss: {best_loss:.4f} (BPB: {best_loss/0.6931:.3f})")
    print(f"  Parameters: {n_params:,}")
    print(f"{'='*60}")
    
    return model, best_loss


if __name__ == "__main__":
    # Match PID-Net v5 Fractal params (~1.4M)
    # Transformer: d_model=192, 4 heads, 4 layers, d_ff=384 → ~1.3M params
    args = {
        'd_model': 192,
        'n_heads': 4,
        'n_layers': 4,
        'd_ff': 384,
        'seq_len': 128,
        'batch_size': 8,
        'steps': 5000,
        'lr': 3e-4,
        'log_every': 100,
        'generate_every': 500,
    }
    
    train_transformer(args)
