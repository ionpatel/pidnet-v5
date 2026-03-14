"""
Edge Evolution — Dynamic Topology (Phase 2)

The graph BREATHES. Edges are born, strengthened, weakened, and removed
based on the I-stream (historical patterns) and D-stream (surprise).

This is what makes PID-Net v5 fundamentally different from GNNs:
- GNNs: fixed topology, only node features change
- PID-Net: topology EVOLVES, the computation graph itself is dynamic

Edge birth/death is driven by:
- I-stream: persistent patterns → stable connections
- D-stream: novel relationships → new edges, broken expectations → edge death
- P-stream provides context but doesn't directly control topology
"""

import mlx.core as mx
import mlx.nn as nn


class EdgeEvolver(nn.Module):
    """Evolve graph topology based on PID signals.
    
    Edges are soft (continuous weights [0, 1]), so birth/death
    is differentiable — gradients flow through topology changes.
    """
    
    def __init__(self, d_model: int, top_k: int = 16):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k  # max edges per node
        
        # Edge score: should an edge exist between nodes i and j?
        # Takes concatenated node features + I/D signals
        self.edge_scorer = nn.Linear(d_model * 2, 1)
        
        # Edge strength modulation from I-stream (historical importance)
        self.W_i_edge = nn.Linear(d_model, d_model, bias=False)
        
        # Edge strength modulation from D-stream (novelty/surprise)
        self.W_d_edge = nn.Linear(d_model, d_model, bias=False)
        
        # Blend between old and new adjacency
        self.evolution_rate = 0.3  # how fast topology changes (0=frozen, 1=full replace)
    
    def __call__(
        self,
        adjacency: mx.array,      # [batch, N, N] current edges
        nodes: mx.array,           # [batch, N, d] current node features
        i_signal: mx.array,        # [batch, N, d] I-stream output
        d_signal: mx.array,        # [batch, N, d] D-stream output
        mask: mx.array,            # [batch, N] active nodes
    ) -> mx.array:
        """
        Evolve edge weights based on PID signals.
        
        Returns:
            Updated adjacency matrix [batch, N, N]
        """
        batch_size, N, d = nodes.shape
        
        # Compute node affinities (which nodes SHOULD be connected?)
        # Use I-stream modulated features for historical relevance
        i_features = self.W_i_edge(i_signal)  # [batch, N, d]
        d_features = self.W_d_edge(d_signal)  # [batch, N, d]
        
        # Combined features for edge scoring
        combined = nodes + 0.5 * i_features + 0.5 * d_features  # [batch, N, d]
        
        # Compute pairwise scores via dot product (efficient)
        # score[i,j] = combined[i] · combined[j] / sqrt(d)
        scores = mx.matmul(combined, mx.transpose(combined, axes=(0, 2, 1)))
        scores = scores / (d ** 0.5)  # [batch, N, N]
        
        # Apply causal mask (can't connect to future nodes)
        causal = mx.tril(mx.ones((N, N)))
        scores = scores * causal
        
        # Apply node mask (can't connect to/from inactive nodes)
        mask_2d = mx.expand_dims(mask, -1) * mx.expand_dims(mask, -2)
        scores = scores * mask_2d + (1 - mask_2d) * (-1e9)
        
        # No self-loops
        eye = mx.eye(N)
        scores = scores * (1 - eye) + eye * (-1e9)
        
        # Sigmoid to get edge probabilities [0, 1]
        new_adjacency = mx.sigmoid(scores)
        
        # Top-k sparsification per node (keep only strongest connections)
        if self.top_k < N:
            # For each node, keep only top_k incoming edges
            # Set others to 0 via masking
            sorted_indices = mx.argsort(-new_adjacency, axis=-1)  # descending
            # Create mask: 1 for top-k, 0 for rest
            rank = mx.argsort(mx.argsort(-new_adjacency, axis=-1), axis=-1)
            topk_mask = (rank < self.top_k).astype(mx.float32)
            new_adjacency = new_adjacency * topk_mask
        
        # Enforce causal mask again (safety)
        new_adjacency = new_adjacency * causal * mask_2d
        
        # Blend old and new (don't change topology too fast)
        evolved = (1 - self.evolution_rate) * adjacency + self.evolution_rate * new_adjacency
        
        return evolved
