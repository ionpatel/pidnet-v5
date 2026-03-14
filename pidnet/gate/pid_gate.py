"""
PID Gate — the "consciousness" of the rewriting system.

Decides:
1. How much to weight P vs I vs D (the blend)
2. Whether to skip this rewriting step entirely (adaptive compute)

The gate is what makes PID-Net a controller, not just three parallel streams.
It's the decision-making bottleneck that forces the model to CHOOSE
what matters right now — perception? memory? surprise?

In cognitive terms: this is selective attention + executive function.
In control theory: this is the adaptive gain scheduler.
In Wolfram Physics: this is the rule selector.
"""

import mlx.core as mx
import mlx.nn as nn

from typing import Tuple


class PIDGate(nn.Module):
    """Gate that blends P, I, D streams and controls compute allocation.
    
    Outputs:
        - gate_weights: [gp, gi, gd] softmax blend of streams
        - skip: probability of skipping this rewriting step
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        # Input: concatenated summaries of P, I, D, and current state
        gate_input_dim = d_model * 4  # [P_mean, I_mean, D_mean, X_mean]
        
        # Stream blend: which PID components matter?
        self.W_blend = nn.Linear(gate_input_dim, 3)  # → softmax → [gp, gi, gd]
        
        # Skip gate: should we even apply this rewrite?
        # Initialize bias negative so model defaults to "apply" not "skip"
        self.W_skip = nn.Linear(gate_input_dim, 1)
        
        # Minimum gate values (prevent any stream from dying completely)
        self.min_gate = 0.05
    
    def __call__(
        self,
        p_output: mx.array,   # [batch, N, d] — P-stream result
        i_output: mx.array,   # [batch, N, d] — I-stream result
        d_output: mx.array,   # [batch, N, d] — D-stream result
        nodes: mx.array,      # [batch, N, d] — current node state
        mask: mx.array,       # [batch, N]
    ) -> Tuple[mx.array, mx.array]:
        """
        Compute gate values for blending P, I, D.
        
        Returns:
            gate_weights: [batch, 3] — softmax weights for P, I, D
            skip: [batch, 1] — skip probability for this rewriting step
        """
        mask_expanded = mx.expand_dims(mask, -1)  # [batch, N, 1]
        n_active = mx.sum(mask, axis=-1, keepdims=True) + 1e-8  # [batch, 1]
        
        # Global mean of each stream (summarize entire graph)
        p_mean = mx.sum(p_output * mask_expanded, axis=1) / n_active  # [batch, d]
        i_mean = mx.sum(i_output * mask_expanded, axis=1) / n_active
        d_mean = mx.sum(d_output * mask_expanded, axis=1) / n_active
        x_mean = mx.sum(nodes * mask_expanded, axis=1) / n_active
        
        # Concatenate summaries
        gate_input = mx.concatenate([p_mean, i_mean, d_mean, x_mean], axis=-1)
        # [batch, 4*d]
        
        # Stream blend
        blend_logits = self.W_blend(gate_input)  # [batch, 3]
        gate_weights = mx.softmax(blend_logits, axis=-1)  # [batch, 3]
        
        # Enforce minimum gate values (prevent stream death)
        gate_weights = gate_weights * (1 - 3 * self.min_gate) + self.min_gate
        
        # Skip gate
        skip = mx.sigmoid(self.W_skip(gate_input))  # [batch, 1]
        
        return gate_weights, skip
