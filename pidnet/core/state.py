"""
Cognitive Hypergraph State — the fundamental data structure of PID-Net v5.

The state represents the "universe" that PID rewriting rules operate on:
  - nodes: feature vectors for each active token/concept
  - adjacency: soft edge weights between nodes (differentiable, [0,1])
  - fast_weights: associative memory matrix (I-stream storage)
  - prediction: predicted next state (D-stream target)
  - mask: which node positions are active (for batching/padding)
"""

import mlx.core as mx
from dataclasses import dataclass
from typing import Optional


@dataclass
class GraphState:
    """The cognitive hypergraph at a single point in time.
    
    This is the "universe" that rewriting rules transform.
    Every PID rewriting step takes a GraphState and produces a new one.
    """
    
    # Node features: [batch, max_nodes, d_model]
    nodes: mx.array
    
    # Soft adjacency matrix: [batch, max_nodes, max_nodes] ∈ [0, 1]
    # Differentiable — edges are continuous weights, not binary
    adjacency: mx.array
    
    # Fast weight memory: [batch, d_model, d_model]
    # The I-stream's associative memory (linear attention / Hopfield)
    fast_weights: mx.array
    
    # Predicted next node state: [batch, max_nodes, d_model]
    # The D-stream predicts this, then computes error against actual
    prediction: mx.array
    
    # Node mask: [batch, max_nodes] — True where nodes are active
    mask: mx.array
    
    # Number of active nodes per batch element
    n_active: mx.array
    
    @property
    def batch_size(self) -> int:
        return self.nodes.shape[0]
    
    @property
    def max_nodes(self) -> int:
        return self.nodes.shape[1]
    
    @property
    def d_model(self) -> int:
        return self.nodes.shape[2]


def create_empty_state(
    batch_size: int,
    max_nodes: int,
    d_model: int,
) -> GraphState:
    """Create an empty cognitive graph with no active nodes.
    
    This is the "blank slate" — a universe before any rewriting has occurred.
    """
    return GraphState(
        nodes=mx.zeros((batch_size, max_nodes, d_model)),
        adjacency=mx.zeros((batch_size, max_nodes, max_nodes)),
        fast_weights=mx.zeros((batch_size, d_model, d_model)),
        prediction=mx.zeros((batch_size, max_nodes, d_model)),
        mask=mx.zeros((batch_size, max_nodes), dtype=mx.bool_),
        n_active=mx.zeros((batch_size,), dtype=mx.int32),
    )


def add_node(
    state: GraphState,
    features: mx.array,        # [batch, d_model]
    connect_k: int = 8,        # connect to k most recent nodes
) -> GraphState:
    """Add a new node to the graph (one per batch element).
    
    New node connects to the k most recent active nodes via
    initial edges. These edges will evolve through PID rewriting.
    
    Args:
        state: Current graph state
        features: Feature vector for the new node [batch, d_model]
        connect_k: Number of previous nodes to connect to
    
    Returns:
        Updated GraphState with new node added
    """
    batch_size = state.batch_size
    max_nodes = state.max_nodes
    
    # Position for new node (next empty slot)
    pos = state.n_active  # [batch]
    
    # Update node features
    # Scatter features into the correct position
    nodes = state.nodes
    for b in range(batch_size):
        p = pos[b].item()
        if p < max_nodes:
            nodes = nodes.at[b, p].add(features[b])
    
    # Update adjacency — connect to k most recent nodes
    adjacency = state.adjacency
    for b in range(batch_size):
        p = pos[b].item()
        if p < max_nodes:
            # Connect to up to k previous nodes with initial weight
            start = max(0, p - connect_k)
            for j in range(start, p):
                # Bidirectional initial edges with decaying strength
                strength = 1.0 / (p - j)  # closer = stronger
                adjacency = adjacency.at[b, p, j].add(strength)
                adjacency = adjacency.at[b, j, p].add(strength * 0.5)  # asymmetric
    
    # Update mask
    mask = state.mask
    for b in range(batch_size):
        p = pos[b].item()
        if p < max_nodes:
            mask = mask.at[b, p].add(True)
    
    return GraphState(
        nodes=nodes,
        adjacency=adjacency,
        fast_weights=state.fast_weights,
        prediction=state.prediction,
        mask=mask,
        n_active=mx.minimum(state.n_active + 1, max_nodes),
    )


def masked_softmax(x: mx.array, mask: mx.array, axis: int = -1) -> mx.array:
    """Softmax that respects node masks (inactive nodes get zero weight)."""
    # Set masked positions to -inf before softmax
    x = mx.where(mask, x, mx.array(float('-inf')))
    return mx.softmax(x, axis=axis)


def graph_energy(state: GraphState) -> mx.array:
    """Compute Hamiltonian energy of the graph state.
    
    H = ½||X||² + ½||W||²_F - Σᵢⱼ Aᵢⱼ · xᵢᵀxⱼ
    
    Used to enforce energy conservation (prevents collapse/explosion).
    Returns: [batch] energy per batch element
    """
    # Kinetic energy: node feature norms
    node_energy = 0.5 * mx.sum(state.nodes ** 2, axis=(-2, -1))
    
    # Memory energy: fast weight matrix norm
    memory_energy = 0.5 * mx.sum(state.fast_weights ** 2, axis=(-2, -1))
    
    # Interaction energy: edge-weighted node similarities
    # XᵀAX summed over edges
    node_sim = mx.matmul(state.nodes, mx.transpose(state.nodes, axes=(0, 2, 1)))
    interaction = -mx.sum(state.adjacency * node_sim, axis=(-2, -1))
    
    return node_energy + memory_energy + interaction
