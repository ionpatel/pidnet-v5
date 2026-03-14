"""
P-Stream (Proportional / Perception)

"What's happening RIGHT NOW in the graph?"

The P-stream reads the current cognitive graph via message passing.
Each node aggregates information from its neighbors, producing a
perception of the current state.

This is the simplest stream — it's a graph neural network forward pass.
But within the PID framework, it provides the "current measurement"
that the other streams (I and D) compare against.
"""

import mlx.core as mx
import mlx.nn as nn

from ..core.message_passing import MessagePassing


class PStream(nn.Module):
    """Proportional stream — perceive current graph state.
    
    P_t = MessagePass(X_t, A_t) → "what does the graph look like now?"
    """
    
    def __init__(self, d_model: int, n_hops: int = 1):
        super().__init__()
        self.message_pass = MessagePassing(d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
    
    def __call__(
        self,
        nodes: mx.array,       # [batch, N, d]
        adjacency: mx.array,   # [batch, N, N]
        mask: mx.array,        # [batch, N]
    ) -> mx.array:
        """
        Perceive current graph state.
        
        Returns:
            P-stream output [batch, N, d_model]
        """
        # Message passing: aggregate neighbor information
        perceived = self.message_pass(nodes, adjacency, mask)
        
        # Project through nonlinearity
        output = self.act(self.proj(perceived))
        
        # Mask inactive nodes
        output = output * mx.expand_dims(mask, -1)
        
        return output
