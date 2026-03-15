"""
Fractal Multi-Scale PID — The Sierpiński Property

The SAME PID rewriting rule operates at multiple scales simultaneously:
  Level 0 (Token):  individual characters/tokens as nodes
  Level 1 (Chunk):  groups of k tokens pooled into super-nodes
  Level 2 (Block):  groups of chunks pooled into block-nodes

Cross-scale communication:
  Bottom-up: pool token features → chunk features → block features
  Top-down:  broadcast block context → chunk context → token context

The shared rewriting rule is the fractal property — same weights at every
scale, different behavior emerges from different graph structures.

Complexity: O(k_levels × d²) per token — linear cost, exponential reach.
With chunk_size=16 and 3 levels: reach = 16³ = 4096 tokens per block.
"""

import mlx.core as mx
import mlx.nn as nn

from .core.state import GraphState
from .model import PIDRewriteStep

from typing import Tuple, Dict, Optional


class FractalPool(nn.Module):
    """Pool token-level features into chunk-level super-nodes.
    
    Takes [batch, N, d] → [batch, N//chunk_size, d]
    Uses learned weighted pooling (not just mean).
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.pool_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def __call__(
        self,
        nodes: mx.array,        # [batch, N, d]
        chunk_size: int,
    ) -> mx.array:
        """Pool N nodes into N//chunk_size super-nodes."""
        batch, N, d = nodes.shape
        n_chunks = N // chunk_size
        
        # Truncate to exact multiple of chunk_size
        truncated = nodes[:, :n_chunks * chunk_size, :]
        
        # Reshape into chunks: [batch, n_chunks, chunk_size, d]
        chunked = truncated.reshape(batch, n_chunks, chunk_size, d)
        
        # Mean pool within each chunk
        pooled = mx.mean(chunked, axis=2)  # [batch, n_chunks, d]
        
        # Project and normalize
        pooled = self.norm(self.pool_proj(pooled))
        
        return pooled


class FractalBroadcast(nn.Module):
    """Broadcast chunk-level context back to token level.
    
    Each token receives its chunk's summary as additional context.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.broadcast_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model * 2, 1)  # gate how much chunk info to use
    
    def __call__(
        self,
        token_nodes: mx.array,    # [batch, N, d]
        chunk_nodes: mx.array,    # [batch, n_chunks, d]
        chunk_size: int,
    ) -> mx.array:
        """Add chunk-level context to each token."""
        batch, N, d = token_nodes.shape
        n_chunks = chunk_nodes.shape[1]
        
        # Project chunk features
        chunk_context = self.broadcast_proj(chunk_nodes)  # [batch, n_chunks, d]
        
        # Repeat each chunk's context for all tokens in that chunk
        # [batch, n_chunks, d] → [batch, n_chunks, chunk_size, d] → [batch, N_trunc, d]
        expanded = mx.repeat(chunk_context[:, :, None, :], chunk_size, axis=2)
        expanded = expanded.reshape(batch, n_chunks * chunk_size, d)
        
        # Pad if needed (when N > n_chunks * chunk_size)
        if expanded.shape[1] < N:
            pad_size = N - expanded.shape[1]
            padding = mx.zeros((batch, pad_size, d))
            expanded = mx.concatenate([expanded, padding], axis=1)
        else:
            expanded = expanded[:, :N, :]
        
        # Gated addition: let the model decide how much chunk context to use
        gate_input = mx.concatenate([token_nodes, expanded], axis=-1)  # [batch, N, 2d]
        gate_val = mx.sigmoid(self.gate(gate_input))  # [batch, N, 1]
        
        # Add gated chunk context
        output = token_nodes + gate_val * expanded
        
        return output


class FractalPIDNet(nn.Module):
    """Multi-scale PID-Net with fractal shared rewriting rule.
    
    The SAME PIDRewriteStep operates at every scale.
    This is the Sierpiński property: self-similar processing.
    
    Architecture:
      Embed → [Token PID Rewrite × R] 
           → Pool → [Chunk PID Rewrite × R]
           → Pool → [Block PID Rewrite × R]  
           → Broadcast Block → Chunk
           → Broadcast Chunk → Token
           → Readout
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        max_nodes: int = 256,
        n_rewrite_steps: int = 3,
        connect_k: int = 8,
        chunk_size: int = 16,
        n_levels: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_nodes = max_nodes
        self.n_rewrite_steps = n_rewrite_steps
        self.connect_k = connect_k
        self.chunk_size = chunk_size
        self.n_levels = n_levels
        
        # Token embedding
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_nodes, d_model)
        
        # SHARED PID rewriting step — used at ALL scales (fractal!)
        self.rewrite_step = PIDRewriteStep(d_model, evolve_edges=True)
        
        # Multi-scale pooling and broadcasting
        # (one per transition between levels)
        self.pools = [FractalPool(d_model) for _ in range(n_levels - 1)]
        self.broadcasts = [FractalBroadcast(d_model) for _ in range(n_levels - 1)]
        
        # Readout (weight-tied with embedding for large vocabs)
        self.readout_norm = nn.LayerNorm(d_model)
        self._tie_weights = (vocab_size > 1000)  # tie for BPE, not char-level
        if not self._tie_weights:
            self.readout = nn.Linear(d_model, vocab_size)
        
        # Pre-compute causal adjacency templates for each level
        self._adj_cache = {}
    
    def _get_causal_adjacency(self, size: int) -> mx.array:
        """Get or build causal adjacency for given size. VECTORIZED."""
        if size not in self._adj_cache:
            k = min(self.connect_k, size)
            indices = mx.arange(size)
            dist = indices[:, None] - indices[None, :]
            valid = (dist > 0) & (dist <= k)
            strength = mx.where(valid, 1.0 / dist, 0.0)
            valid_back = (dist < 0) & (-dist <= k)
            strength_back = mx.where(valid_back, 0.5 / (-dist), 0.0)
            adj = strength + strength_back
            causal = mx.tril(mx.ones((size, size)))
            self._adj_cache[size] = adj * causal
        return self._adj_cache[size]
    
    def _build_state(self, nodes: mx.array) -> GraphState:
        """Build a GraphState from node features."""
        batch, N, d = nodes.shape
        adj = self._get_causal_adjacency(N)
        adj = mx.broadcast_to(adj, (batch, N, N))
        
        return GraphState(
            nodes=nodes,
            adjacency=adj,
            fast_weights=mx.zeros((batch, d, d)),
            prediction=mx.zeros_like(nodes),
            mask=mx.ones((batch, N), dtype=mx.bool_),
            n_active=mx.full((batch,), N, dtype=mx.int32),
        )
    
    def __call__(
        self,
        tokens: mx.array,  # [batch, seq_len]
    ) -> Tuple[mx.array, Dict]:
        """
        Multi-scale fractal forward pass.
        
        Processing flow:
          1. Embed tokens → Level 0 nodes
          2. For each level (bottom-up):
             - Run R PID rewriting steps (SHARED rule)
             - Pool to next level
          3. For each level (top-down):
             - Broadcast higher-level context to lower level
          4. Readout from Level 0
        """
        batch_size, seq_len = tokens.shape
        
        # === EMBED ===
        positions = mx.arange(seq_len)
        nodes_l0 = self.embed(tokens) + self.pos_embed(positions)
        
        # === BOTTOM-UP: Rewrite at each scale, then pool ===
        level_nodes = [nodes_l0]  # store each level's nodes
        level_diagnostics = []
        
        current_nodes = nodes_l0
        for level in range(self.n_levels):
            # Build graph state for this level
            state = self._build_state(current_nodes)
            
            # Preserve node norms
            norm_before = mx.sqrt(mx.sum(state.nodes ** 2, axis=-1, keepdims=True) + 1e-8)
            
            # Apply R rounds of PID rewriting (SAME shared rule!)
            for r in range(self.n_rewrite_steps):
                state, diag = self.rewrite_step(state)
            
            # Norm preservation
            norm_after = mx.sqrt(mx.sum(state.nodes ** 2, axis=-1, keepdims=True) + 1e-8)
            ratio = mx.clip(norm_before / norm_after, 0.8, 1.2)
            state = GraphState(
                nodes=state.nodes * ratio,
                adjacency=state.adjacency,
                fast_weights=state.fast_weights,
                prediction=state.prediction,
                mask=state.mask,
                n_active=state.n_active,
            )
            
            level_nodes[level] = state.nodes  # update with rewritten nodes
            level_diagnostics.append(diag)
            
            # Pool to next level (if not last)
            if level < self.n_levels - 1:
                n_nodes = current_nodes.shape[1]
                if n_nodes >= self.chunk_size * 2:  # need at least 2 chunks
                    current_nodes = self.pools[level](state.nodes, self.chunk_size)
                    level_nodes.append(current_nodes)
                else:
                    # Too few nodes to pool further — stop going up
                    break
        
        # === TOP-DOWN: Broadcast higher-level context ===
        for level in range(len(level_nodes) - 1, 0, -1):
            higher = level_nodes[level]
            lower = level_nodes[level - 1]
            
            # Broadcast: add higher-level context to lower level
            level_nodes[level - 1] = self.broadcasts[level - 1](
                lower, higher, self.chunk_size
            )
        
        # === READOUT from Level 0 (token level) ===
        final_nodes = level_nodes[0]
        normed = self.readout_norm(final_nodes)
        if self._tie_weights:
            # Weight tying: reuse embedding weights for readout (saves ~50% params for large vocab)
            logits = normed @ self.embed.weight.T
        else:
            logits = self.readout(normed)
        
        # Aggregate diagnostics
        avg_diagnostics = {}
        if level_diagnostics:
            for key in level_diagnostics[0]:
                values = [d[key] for d in level_diagnostics if key in d]
                if values:
                    avg_diagnostics[key] = mx.mean(mx.stack(values))
        
        avg_diagnostics['n_levels_active'] = mx.array(float(len(level_nodes)))
        avg_diagnostics['energy_ratio'] = mx.mean(ratio)
        
        return logits, avg_diagnostics
    
    def _compiled_forward(self, tokens: mx.array) -> mx.array:
        """Compiled forward pass — returns last token logits only.
        
        mx.compile traces this once, then runs optimized Metal directly.
        Eliminates Python dispatch overhead on subsequent calls.
        """
        logits, _ = self.__call__(tokens)
        return logits[:, -1, :]  # Only last token logits needed for generation
    
    def _apply_penalties_gpu(
        self, 
        logits: mx.array,
        recent_tokens: mx.array,
        penalty: float,
        temperature: float,
        top_k: int,
    ) -> mx.array:
        """Apply repetition penalty + sampling on GPU (no Python loops).
        
        Instead of Python dict/Counter, we compute frequency counts
        as GPU tensor operations. ~10x faster than Python loop.
        """
        vocab_size = logits.shape[-1]
        
        # Build frequency vector on GPU: count occurrences of each token
        # One-hot encode recent tokens and sum → frequency per vocab entry
        if recent_tokens.shape[0] > 0:
            one_hot = mx.zeros((recent_tokens.shape[0], vocab_size))
            one_hot = mx.scatter(
                one_hot, 
                recent_tokens[:, None],
                mx.ones_like(recent_tokens[:, None], dtype=mx.float32),
                axes=[1]
            )
            freq_counts = mx.sum(one_hot, axis=0)  # [vocab_size]
            
            # penalty^count for each vocab entry (0 count → penalty^0 = 1.0, no effect)
            penalties = mx.power(penalty, freq_counts)
            
            # Apply: positive logits divided, negative logits multiplied
            pos_mask = logits > 0
            logits = mx.where(pos_mask, logits / penalties, logits * penalties)
        
        # Temperature
        logits = logits / temperature
        
        # Top-k filtering on GPU
        if top_k > 0 and top_k < vocab_size:
            top_k_val = mx.sort(logits)[-top_k]
            logits = mx.where(logits < top_k_val, float('-inf'), logits)
        
        return logits
    
    def generate(
        self,
        prompt_tokens: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        repetition_penalty: float = 1.5,
        penalty_window: int = 32,
        use_compile: bool = True,
    ) -> mx.array:
        """Autoregressive generation with compiled forward pass.
        
        Optimizations:
        1. mx.compile on forward pass — eliminates Python dispatch overhead
        2. GPU-based repetition penalty — no Python loops for frequency counting
        3. GPU-based top-k + sampling — stays on device
        
        Penalty scales exponentially with token frequency in the window:
        penalty^count. This breaks the PID equilibrium trap.
        """
        # Compile forward pass (traced once, then runs native Metal)
        if use_compile:
            try:
                forward_fn = mx.compile(self._compiled_forward)
            except Exception:
                forward_fn = self._compiled_forward
        else:
            forward_fn = self._compiled_forward
        
        tokens = prompt_tokens.tolist()[0] if prompt_tokens.ndim > 1 else prompt_tokens.tolist()
        generated = list(tokens)
        
        # Warmup: first call traces the graph
        warmup_input = mx.array([generated])
        _ = forward_fn(warmup_input)
        mx.eval(_)
        
        for _ in range(max_new_tokens):
            if len(generated) >= self.max_nodes:
                break
            
            input_tokens = mx.array([generated])
            last_logits = forward_fn(input_tokens)[0]  # [vocab_size]
            
            # GPU-based repetition penalty
            recent = mx.array(generated[-penalty_window:]) if len(generated) > 0 else mx.array([], dtype=mx.int32)
            last_logits = self._apply_penalties_gpu(
                last_logits, recent, repetition_penalty, temperature, top_k
            )
            
            # Sample on GPU
            probs = mx.softmax(last_logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
            mx.eval(next_token)  # Force evaluation
            generated.append(next_token.item())
        
        return mx.array(generated)
