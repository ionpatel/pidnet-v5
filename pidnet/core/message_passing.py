"""
Graph Message Passing — the P-stream's perception mechanism.

Nodes aggregate information from their neighbors via weighted edges.
This is how the model "sees" the current state of the cognitive graph.

Unlike attention (O(N²)), message passing uses the learned sparse
adjacency matrix, making it O(|E|) where |E| << N².
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional


class MessagePassing(nn.Module):
    """Single round of message passing over the cognitive graph.
    
    Each node aggregates neighbor features weighted by edge strength:
        messages_i = Σⱼ A[i,j] · W_msg · x_j
        x_i' = LayerNorm(x_i + messages_i)
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.W_msg = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
    
    def __call__(
        self,
        nodes: mx.array,       # [batch, N, d]
        adjacency: mx.array,   # [batch, N, N]
        mask: mx.array,        # [batch, N]
    ) -> mx.array:
        """
        Perform one round of message passing.
        
        Args:
            nodes: Node feature matrix [batch, N, d_model]
            adjacency: Soft adjacency [batch, N, N] 
            mask: Active node mask [batch, N]
        
        Returns:
            Updated node features [batch, N, d_model]
        """
        # Transform neighbor features
        transformed = self.W_msg(nodes)  # [batch, N, d]
        
        # Mask adjacency: zero out edges to/from inactive nodes
        # mask_2d[b, i, j] = mask[b, i] AND mask[b, j]
        mask_2d = mx.expand_dims(mask, -1) * mx.expand_dims(mask, -2)
        masked_adj = adjacency * mask_2d  # [batch, N, N]
        
        # Normalize adjacency (so messages don't explode with many neighbors)
        # Row-normalize: each node's incoming edges sum to 1
        row_sum = mx.sum(masked_adj, axis=-1, keepdims=True) + 1e-8
        normalized_adj = masked_adj / row_sum
        
        # Aggregate: messages_i = Σⱼ A_norm[i,j] · transformed_j
        messages = mx.matmul(normalized_adj, transformed)  # [batch, N, d]
        
        # Residual + norm
        output = self.norm(nodes + messages)
        
        # Zero out inactive nodes
        output = output * mx.expand_dims(mask, -1)
        
        return output


class MultiHopMessagePassing(nn.Module):
    """Multiple rounds of message passing for larger receptive field.
    
    Each hop expands the effective neighborhood by one edge.
    k hops = information from k-hop neighbors.
    """
    
    def __init__(self, d_model: int, n_hops: int = 2):
        super().__init__()
        self.hops = [MessagePassing(d_model) for _ in range(n_hops)]
    
    def __call__(
        self,
        nodes: mx.array,
        adjacency: mx.array,
        mask: mx.array,
    ) -> mx.array:
        """Apply n_hops rounds of message passing."""
        x = nodes
        for hop in self.hops:
            x = hop(x, adjacency, mask)
        return x
