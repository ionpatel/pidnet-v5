"""
PID-Net v5: Hypergraph Rewriting Cognitive Architecture

The full model that ties everything together:
  1. Embed input tokens as graph nodes
  2. Apply R rounds of PID rewriting
  3. Read out predictions from the evolved graph

Each rewriting step:
  P-stream → perceive current graph (message passing)
  D-stream → compute prediction error (surprise)
  I-stream → read/write fast weight memory (using D for write strength)
  Gate → blend P/I/D and decide whether to skip
  Apply → update nodes with gated blend
"""

import mlx.core as mx
import mlx.nn as nn

from .core.state import GraphState, create_empty_state, add_node, graph_energy
from .streams.p_stream import PStream
from .streams.i_stream import IStream
from .streams.d_stream import DStream
from .gate.pid_gate import PIDGate
from .graph.edge_evolver import EdgeEvolver

from typing import Tuple, Optional, Dict


class PIDRewriteStep(nn.Module):
    """One round of PID rewriting on the cognitive graph.
    
    Takes a GraphState, applies P/I/D streams, gates the result,
    and produces an updated GraphState. This is the "rewriting rule."
    """
    
    def __init__(self, d_model: int, evolve_edges: bool = True):
        super().__init__()
        self.d_model = d_model
        self.evolve_edges = evolve_edges
        
        # The three streams
        self.p_stream = PStream(d_model)
        self.i_stream = IStream(d_model)
        self.d_stream = DStream(d_model)
        
        # The gate (consciousness / rule selector)
        self.gate = PIDGate(d_model)
        
        # Edge evolution (Phase 2) — the graph breathes
        if evolve_edges:
            self.edge_evolver = EdgeEvolver(d_model, top_k=16)
        
        # Layer norm for residual
        self.norm = nn.LayerNorm(d_model)
    
    def __call__(
        self,
        state: GraphState,
    ) -> Tuple[GraphState, Dict[str, mx.array]]:
        """
        Apply one PID rewriting step to the cognitive graph.
        
        Returns:
            (updated GraphState, diagnostics dict)
        """
        nodes = state.nodes
        adjacency = state.adjacency
        mask = state.mask
        
        # === P-STREAM: Perceive current graph ===
        p_out = self.p_stream(nodes, adjacency, mask)
        
        # === D-STREAM: Compute prediction error ===
        d_out, new_prediction = self.d_stream(nodes, state.prediction, mask)
        
        # === I-STREAM: Read/write memory (D→I coupling) ===
        i_out, new_fast_weights = self.i_stream(
            nodes, state.fast_weights, d_out, mask
        )
        
        # === GATE: Blend P/I/D ===
        gate_weights, skip = self.gate(p_out, i_out, d_out, nodes, mask)
        # gate_weights: [batch, 3] — gp, gi, gd
        # skip: [batch, 1] — skip probability
        
        # Expand gate weights for broadcasting
        gp = gate_weights[:, 0:1].reshape(-1, 1, 1)  # [batch, 1, 1]
        gi = gate_weights[:, 1:2].reshape(-1, 1, 1)
        gd = gate_weights[:, 2:3].reshape(-1, 1, 1)
        skip_expanded = skip.reshape(-1, 1, 1)  # [batch, 1, 1]
        
        # Blend streams
        blended = gp * p_out + gi * i_out + gd * d_out  # [batch, N, d]
        
        # Apply with skip gate (residual)
        new_nodes = skip_expanded * nodes + (1 - skip_expanded) * self.norm(blended)
        
        # Mask inactive nodes
        new_nodes = new_nodes * mx.expand_dims(mask, -1)
        
        # Edge evolution: the graph topology changes based on I and D signals
        if self.evolve_edges:
            new_adjacency = self.edge_evolver(
                adjacency, new_nodes, i_out, d_out, mask
            )
        else:
            new_adjacency = adjacency
        
        # Build updated state
        new_state = GraphState(
            nodes=new_nodes,
            adjacency=new_adjacency,
            fast_weights=new_fast_weights,
            prediction=new_prediction,
            mask=mask,
            n_active=state.n_active,
        )
        
        # Prediction loss (train the predictor directly!)
        pred_loss = self.d_stream.prediction_loss(
            actual=nodes,
            predicted=state.prediction,
            mask=mask,
        )
        
        # Edge diagnostics
        edge_density = mx.sum(new_adjacency * mx.expand_dims(mask, -1) * mx.expand_dims(mask, -2)) / (mx.sum(mask) ** 2 + 1e-8)
        
        # Diagnostics for monitoring
        diagnostics = {
            'gate_p': mx.mean(gate_weights[:, 0]),
            'gate_i': mx.mean(gate_weights[:, 1]),
            'gate_d': mx.mean(gate_weights[:, 2]),
            'skip': mx.mean(skip),
            'pred_loss': pred_loss,
            'pred_error': mx.mean(mx.sqrt(mx.sum(
                (nodes - state.prediction) ** 2 * mx.expand_dims(mask, -1),
                axis=-1
            ) + 1e-8)),
            'edge_density': edge_density,
        }
        
        return new_state, diagnostics


class PIDGraphNet(nn.Module):
    """Full PID-Net v5 model for sequence modeling.
    
    Architecture:
      Input → Embed → [Add node to graph → Apply R rewriting steps] × T → Readout
    
    For Phase 1: Fixed sequential graph topology, R rewriting steps per token.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        max_nodes: int = 256,
        n_rewrite_steps: int = 3,
        connect_k: int = 8,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_nodes = max_nodes
        self.n_rewrite_steps = n_rewrite_steps
        self.connect_k = connect_k
        
        # Token embedding
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Position encoding (learned)
        self.pos_embed = nn.Embedding(max_nodes, d_model)
        
        # PID rewriting steps (shared weights = the fractal property)
        # Phase 1: single shared rewriting rule
        self.rewrite_step = PIDRewriteStep(d_model)
        
        # Readout: predict next token from current node
        self.readout_norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, vocab_size)
        
        # Hamiltonian energy conservation strength
        self.energy_lambda = 0.01
        
        # Pre-compute causal adjacency template (reused every forward pass)
        self._causal_adj = self._build_causal_adjacency(max_nodes, connect_k)
    
    def _build_causal_adjacency(self, size: int, k: int) -> mx.array:
        """Build causal adjacency matrix: node i connects to previous k nodes."""
        adj = mx.zeros((size, size))
        for i in range(size):
            for j in range(max(0, i - k), i):
                strength = 1.0 / (i - j)
                adj = adj.at[i, j].add(strength)
                adj = adj.at[j, i].add(strength * 0.5)
        causal = mx.tril(mx.ones((size, size)))
        return adj * causal
    
    def __call__(
        self,
        tokens: mx.array,     # [batch, seq_len] — token IDs
    ) -> Tuple[mx.array, Dict]:
        """
        Process a sequence of tokens through the hypergraph rewriting system.
        
        PARALLEL MODE (for training): Build full causal graph at once,
        run R rewriting steps on the complete graph. ~10-50x faster than
        sequential token-by-token processing.
        
        The causal adjacency mask ensures each node only sees previous nodes,
        preserving autoregressive property.
        
        Args:
            tokens: Input token IDs [batch, seq_len]
        
        Returns:
            (logits [batch, seq_len, vocab_size], diagnostics dict)
        """
        batch_size, seq_len = tokens.shape
        
        # === BUILD FULL GRAPH AT ONCE (no Python loops!) ===
        
        # Embed all tokens in parallel
        positions = mx.arange(seq_len)
        tok_embeds = self.embed(tokens)                    # [batch, seq_len, d]
        pos_embeds = self.pos_embed(positions)             # [seq_len, d]
        nodes = tok_embeds + pos_embeds                    # [batch, seq_len, d]
        
        # Use pre-computed causal adjacency (sliced to seq_len)
        adjacency = self._causal_adj[:seq_len, :seq_len]
        adjacency = mx.broadcast_to(adjacency, (batch_size, seq_len, seq_len))
        
        # All nodes active
        mask = mx.ones((batch_size, seq_len), dtype=mx.bool_)
        
        # Initialize fast weights — use cached from previous sequence if available
        # This is segment-level recurrence: fast weights carry memory across windows
        if hasattr(self, '_cached_fast_weights') and self._cached_fast_weights is not None:
            # Decay old memory and reuse (cross-window memory!)
            fast_weights = self._cached_fast_weights * 0.9
            # Detach from previous graph to avoid backprop through segments
            fast_weights = mx.stop_gradient(fast_weights)
        else:
            fast_weights = mx.zeros((batch_size, self.d_model, self.d_model))
        prediction = mx.zeros_like(nodes)
        
        state = GraphState(
            nodes=nodes,
            adjacency=adjacency,
            fast_weights=fast_weights,
            prediction=prediction,
            mask=mask,
            n_active=mx.full((batch_size,), seq_len, dtype=mx.int32),
        )
        
        # === APPLY R ROUNDS OF PID REWRITING (on full graph) ===
        # Track node norms before rewriting (for energy conservation)
        node_norm_before = mx.sqrt(mx.sum(state.nodes ** 2, axis=-1, keepdims=True) + 1e-8)
        
        all_diagnostics = []
        for r in range(self.n_rewrite_steps):
            state, diag = self.rewrite_step(state)
            all_diagnostics.append(diag)
        
        # === ENERGY CONSERVATION (per-node norm preservation) ===
        # Instead of global energy ratio (breaks with large graphs),
        # preserve each node's L2 norm through rewriting.
        # This prevents explosion/collapse at the node level.
        node_norm_after = mx.sqrt(mx.sum(state.nodes ** 2, axis=-1, keepdims=True) + 1e-8)
        norm_ratio = node_norm_before / node_norm_after
        norm_ratio = mx.clip(norm_ratio, 0.8, 1.2)  # allow some change, prevent extremes
        state = GraphState(
            nodes=state.nodes * norm_ratio,
            adjacency=state.adjacency,
            fast_weights=state.fast_weights,
            prediction=state.prediction,
            mask=state.mask,
            n_active=state.n_active,
        )
        
        # Cache fast weights for cross-window memory (segment recurrence)
        self._cached_fast_weights = mx.stop_gradient(state.fast_weights)
        
        # === READOUT ALL AT ONCE (no Python loops!) ===
        logits = self.readout(self.readout_norm(state.nodes))  # [batch, seq_len, vocab]
        
        # Aggregate diagnostics
        avg_diagnostics = {}
        if all_diagnostics:
            for key in all_diagnostics[0]:
                values = [d[key] for d in all_diagnostics]
                avg_diagnostics[key] = mx.mean(mx.stack(values))
        
        avg_diagnostics['energy_ratio'] = mx.mean(norm_ratio)
        
        return logits, avg_diagnostics
    
    def generate(
        self,
        prompt_tokens: mx.array,    # [1, prompt_len]
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> mx.array:
        """
        Autoregressive generation using the SAME parallel infrastructure as training.
        
        Key insight: we must use the same processing mode for generation as training,
        otherwise we get train-inference mismatch (the v1-v3 killer bug).
        
        Approach: re-run the full parallel forward pass with the growing sequence
        each time we generate a token. O(T²) total but correct.
        """
        tokens = prompt_tokens.tolist()[0]
        generated = list(tokens)
        
        # Clear cached fast weights (training batch size != generation batch size)
        self._cached_fast_weights = None
        
        for _ in range(max_new_tokens):
            seq_len = len(generated)
            if seq_len >= self.max_nodes:
                break
            
            # Build full sequence tensor
            input_tokens = mx.array([generated])  # [1, current_len]
            
            # Run full parallel forward pass (same as training!)
            logits, _ = self.__call__(input_tokens)  # [1, current_len, vocab]
            
            # Get prediction from LAST position
            last_logits = logits[0, -1]  # [vocab]
            
            # Temperature + top-k sampling
            last_logits = last_logits / temperature
            if top_k > 0:
                top_k_val = mx.sort(last_logits)[-top_k]
                last_logits = mx.where(last_logits < top_k_val, float('-inf'), last_logits)
            probs = mx.softmax(last_logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10)).item()
            
            generated.append(next_token)
        
        return mx.array(generated)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    total = 0
    
    def _count(params):
        nonlocal total
        if isinstance(params, mx.array):
            total += params.size
        elif isinstance(params, dict):
            for v in params.values():
                _count(v)
        elif isinstance(params, (list, tuple)):
            for v in params:
                _count(v)
    
    _count(model.parameters())
    return total
