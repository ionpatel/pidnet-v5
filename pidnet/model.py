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

from typing import Tuple, Optional, Dict


class PIDRewriteStep(nn.Module):
    """One round of PID rewriting on the cognitive graph.
    
    Takes a GraphState, applies P/I/D streams, gates the result,
    and produces an updated GraphState. This is the "rewriting rule."
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # The three streams
        self.p_stream = PStream(d_model)
        self.i_stream = IStream(d_model)
        self.d_stream = DStream(d_model)
        
        # The gate (consciousness / rule selector)
        self.gate = PIDGate(d_model)
        
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
        
        # Build updated state
        new_state = GraphState(
            nodes=new_nodes,
            adjacency=adjacency,  # Fixed topology in Phase 1
            fast_weights=new_fast_weights,
            prediction=new_prediction,
            mask=mask,
            n_active=state.n_active,
        )
        
        # Diagnostics for monitoring
        diagnostics = {
            'gate_p': mx.mean(gate_weights[:, 0]),
            'gate_i': mx.mean(gate_weights[:, 1]),
            'gate_d': mx.mean(gate_weights[:, 2]),
            'skip': mx.mean(skip),
            'pred_error': mx.mean(mx.sqrt(mx.sum(
                (nodes - state.prediction) ** 2 * mx.expand_dims(mask, -1),
                axis=-1
            ) + 1e-8)),
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
    
    def __call__(
        self,
        tokens: mx.array,     # [batch, seq_len] — token IDs
    ) -> Tuple[mx.array, Dict]:
        """
        Process a sequence of tokens through the hypergraph rewriting system.
        
        Args:
            tokens: Input token IDs [batch, seq_len]
        
        Returns:
            (logits [batch, seq_len, vocab_size], diagnostics dict)
        """
        batch_size, seq_len = tokens.shape
        
        # Initialize empty graph
        state = create_empty_state(batch_size, self.max_nodes, self.d_model)
        
        all_logits = []
        all_diagnostics = []
        initial_energy = None
        
        for t in range(seq_len):
            # Embed current token
            tok_embed = self.embed(tokens[:, t])         # [batch, d]
            pos_embed = self.pos_embed(mx.array(t))      # [d]
            features = tok_embed + pos_embed              # [batch, d]
            
            # Add node to graph
            state = add_node(state, features, connect_k=self.connect_k)
            
            # Track initial energy (for Hamiltonian conservation)
            if initial_energy is None:
                initial_energy = graph_energy(state)
            
            # Apply R rounds of PID rewriting
            step_diagnostics = []
            for r in range(self.n_rewrite_steps):
                state, diag = self.rewrite_step(state)
                step_diagnostics.append(diag)
            
            # Energy conservation: project back to energy-preserving manifold
            current_energy = graph_energy(state)
            energy_ratio = mx.sqrt(
                mx.abs(initial_energy) / (mx.abs(current_energy) + 1e-8)
            )
            energy_ratio = mx.clip(energy_ratio, 0.9, 1.1)  # gentle correction
            state = GraphState(
                nodes=state.nodes * energy_ratio.reshape(-1, 1, 1),
                adjacency=state.adjacency,
                fast_weights=state.fast_weights,
                prediction=state.prediction,
                mask=state.mask,
                n_active=state.n_active,
            )
            
            # Readout: predict next token from current node
            current_pos = mx.minimum(state.n_active - 1, self.max_nodes - 1)
            # Get the most recently added node's features
            current_node = mx.zeros((batch_size, self.d_model))
            for b in range(batch_size):
                p = current_pos[b].item()
                current_node = current_node.at[b].add(state.nodes[b, p])
            
            logits = self.readout(self.readout_norm(current_node))  # [batch, vocab]
            all_logits.append(logits)
            
            # Aggregate diagnostics
            if step_diagnostics:
                all_diagnostics.append(step_diagnostics[-1])
        
        # Stack logits: [batch, seq_len, vocab_size]
        logits = mx.stack(all_logits, axis=1)
        
        # Aggregate diagnostics across time
        avg_diagnostics = {}
        if all_diagnostics:
            for key in all_diagnostics[0]:
                values = [d[key] for d in all_diagnostics]
                avg_diagnostics[key] = mx.mean(mx.stack(values))
        
        # Add energy conservation diagnostic
        final_energy = graph_energy(state)
        avg_diagnostics['energy_ratio'] = mx.mean(
            mx.abs(final_energy) / (mx.abs(initial_energy) + 1e-8)
        )
        
        return logits, avg_diagnostics
    
    def generate(
        self,
        prompt_tokens: mx.array,    # [1, prompt_len]
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> mx.array:
        """
        Autoregressive generation from a prompt.
        
        This is the CRITICAL test — v1-v3 collapsed here.
        The hypergraph structure + Hamiltonian conservation
        + predictive coding should prevent collapse.
        """
        tokens = prompt_tokens.tolist()[0]
        state = create_empty_state(1, self.max_nodes, self.d_model)
        
        # Process prompt
        for t, tok in enumerate(tokens):
            tok_embed = self.embed(mx.array([[tok]]))[:, 0]
            pos_embed = self.pos_embed(mx.array(t))
            features = tok_embed + pos_embed
            state = add_node(state, features, connect_k=self.connect_k)
            
            for _ in range(self.n_rewrite_steps):
                state, _ = self.rewrite_step(state)
        
        # Generate new tokens
        generated = list(tokens)
        for _ in range(max_new_tokens):
            t = len(generated)
            if t >= self.max_nodes:
                break
            
            # Get logits from current state
            current_pos = mx.minimum(state.n_active - 1, self.max_nodes - 1)
            p = current_pos[0].item()
            current_node = state.nodes[0:1, p:p+1, :].reshape(1, self.d_model)
            logits = self.readout(self.readout_norm(current_node))[0]  # [vocab]
            
            # Temperature + top-k sampling
            logits = logits / temperature
            if top_k > 0:
                top_k_vals = mx.sort(logits)[-top_k]
                logits = mx.where(logits < top_k_vals, float('-inf'), logits)
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10)).item()
            
            generated.append(next_token)
            
            # Add new token to graph and rewrite
            tok_embed = self.embed(mx.array([[next_token]]))[:, 0]
            pos_embed = self.pos_embed(mx.array(t))
            features = tok_embed + pos_embed
            state = add_node(state, features, connect_k=self.connect_k)
            
            for _ in range(self.n_rewrite_steps):
                state, _ = self.rewrite_step(state)
        
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
