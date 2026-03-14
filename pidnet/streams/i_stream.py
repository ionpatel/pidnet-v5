"""
I-Stream (Integral / Integration / Memory)

"What has the graph looked like OVER TIME?"

The I-stream maintains a fast weight matrix W_fast ∈ ℝ^{d×d} that serves
as associative memory. This is mathematically equivalent to:
  - Linear attention (Katharopoulos et al., 2020)
  - Modern Hopfield networks (Ramsauer et al., 2021)
  - Fast weight programmers (Schmidhuber, 1992)
  - Outer-product associative memory

Why fast weights instead of EMA:
  EMA: x̄ = λx̄ + (1-λ)x  → blurry average, nothing is retrievable
  Fast weights: W += η · k ⊗ v  → discrete, retrievable memories

The model WILL use this because stored patterns are retrievable.
(v3 proved EMA is useless — model correctly ignores a blurry average.)

D→I Coupling: Write strength η is modulated by prediction error.
High surprise = write more strongly (like dopamine in the brain).
"""

import mlx.core as mx
import mlx.nn as nn

from typing import Tuple


class IStream(nn.Module):
    """Integral stream — fast weight associative memory.
    
    Write: W_fast += η · outer(key, value)  — store current state
    Read:  I_t = W_fast @ query             — retrieve past context
    Decay: W_fast *= λ                      — forget old memories
    
    The write strength η comes from the D-stream (surprise → remember).
    """
    
    def __init__(
        self,
        d_model: int,
        n_timescales: int = 4,
        base_decay: float = 0.95,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_timescales = n_timescales
        
        # Projections for key/value/query
        self.W_key = nn.Linear(d_model, d_model, bias=False)
        self.W_value = nn.Linear(d_model, d_model, bias=False)
        self.W_query = nn.Linear(d_model, d_model, bias=False)
        
        # Write strength modulation (receives D-stream signal)
        self.W_eta = nn.Linear(d_model, n_timescales)
        
        # Multi-timescale decay rates (power-law spacing)
        # τ = [2, 4, 8, 16, ...] → λ = [0.5, 0.75, 0.875, 0.9375, ...]
        taus = [2 ** (i + 1) for i in range(n_timescales)]
        self.decay_rates = mx.array([1.0 - 1.0 / tau for tau in taus])
        
        # Learned weights for combining timescales
        self.W_combine = nn.Linear(n_timescales * d_model, d_model)
        
        # Output projection
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def __call__(
        self,
        nodes: mx.array,           # [batch, N, d] — current node features
        fast_weights: mx.array,    # [batch, d, d] — current memory state
        d_signal: mx.array,        # [batch, N, d] — D-stream output (for η modulation)
        mask: mx.array,            # [batch, N]
    ) -> Tuple[mx.array, mx.array]:
        """
        Read from and write to fast weight memory.
        
        Args:
            nodes: Current node features
            fast_weights: Current fast weight matrix (will be updated)
            d_signal: D-stream prediction error (modulates write strength)
            mask: Active node mask
        
        Returns:
            (I-stream output [batch, N, d], updated fast_weights [batch, d, d])
        """
        batch_size = nodes.shape[0]
        
        # === WRITE: Store current state in memory ===
        
        # Compute keys and values from current nodes
        # Average over active nodes for global write
        mask_expanded = mx.expand_dims(mask, -1)  # [batch, N, 1]
        n_active = mx.sum(mask, axis=-1, keepdims=True).reshape(batch_size, 1, 1) + 1e-8
        
        # Global node summary for write
        node_summary = mx.sum(nodes * mask_expanded, axis=1) / n_active.squeeze(-1)  # [batch, d]
        
        keys = self.W_key(node_summary)      # [batch, d]
        values = self.W_value(node_summary)   # [batch, d]
        
        # D→I coupling: write strength from prediction error magnitude
        d_summary = mx.sum(d_signal * mask_expanded, axis=1) / n_active.squeeze(-1)  # [batch, d]
        eta = mx.sigmoid(self.W_eta(d_summary))  # [batch, n_timescales]
        
        # Outer product update: W += η · k ⊗ v
        outer = mx.expand_dims(keys, -1) * mx.expand_dims(values, -2)  # [batch, d, d]
        
        # Multi-timescale: decay + write for each timescale
        # For Phase 1 simplicity, use single timescale
        # (multi-timescale banks will be separate arrays in Phase 4)
        avg_eta = mx.mean(eta, axis=-1, keepdims=True).reshape(batch_size, 1, 1)
        avg_decay = mx.mean(self.decay_rates).item()
        
        new_fast_weights = avg_decay * fast_weights + avg_eta * outer
        
        # === READ: Retrieve relevant past context ===
        
        # Query from current nodes
        queries = self.W_query(nodes)  # [batch, N, d]
        
        # Linear attention readout: retrieved = W_fast @ query
        # For each node, retrieve from memory
        retrieved = mx.matmul(
            queries,
            mx.transpose(new_fast_weights, axes=(0, 2, 1))
        )  # [batch, N, d]
        
        # Project and normalize
        output = self.norm(self.proj(retrieved))
        
        # Mask inactive nodes
        output = output * mask_expanded
        
        return output, new_fast_weights
