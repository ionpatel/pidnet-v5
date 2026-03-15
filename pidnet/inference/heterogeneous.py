"""
Heterogeneous Inference Engine for PID-Net on Apple Silicon.

Maps PID-Net operations to optimal compute tiers:
- GPU: Matrix multiplications (message passing, edge evolution, fractal ops)
- CPU: Tokenization, control flow, weight paging
- Neural Engine: (future) Embedding lookups, gate softmax

Tiered memory:
- RAM: Active PID weights + hot graph states (Level 0, 1)
- SSD: Cold graph states (Level 2+), checkpoints

D-stream predictive prefetching:
- High prediction error → prefetch next fractal level from SSD
- Model tells hardware what to load BEFORE it's needed
"""

import mlx.core as mx
import numpy as np
import os
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TieredMemoryConfig:
    """Configuration for tiered memory management."""
    # RAM budget for graph states (in bytes). Rest goes to SSD.
    ram_budget_mb: int = 512
    # SSD cache directory for cold graph states
    ssd_cache_dir: str = "cache/graph_states"
    # Prefetch threshold: D-stream error norm above this triggers prefetch
    prefetch_threshold: float = 0.5
    # Max fractal levels to keep in RAM (rest on SSD)
    hot_levels: int = 2
    # Enable async prefetching
    async_prefetch: bool = True


@dataclass
class InferenceStats:
    """Track inference performance metrics."""
    total_tokens: int = 0
    total_time_ms: float = 0.0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    ssd_loads: int = 0
    ssd_load_time_ms: float = 0.0
    gpu_compute_time_ms: float = 0.0
    
    @property
    def tokens_per_second(self) -> float:
        if self.total_time_ms == 0:
            return 0.0
        return self.total_tokens / (self.total_time_ms / 1000.0)
    
    @property
    def prefetch_hit_rate(self) -> float:
        total = self.prefetch_hits + self.prefetch_misses
        return self.prefetch_hits / total if total > 0 else 0.0
    
    def summary(self) -> str:
        return (
            f"Tokens: {self.total_tokens} | "
            f"{self.tokens_per_second:.0f} tok/s | "
            f"Prefetch hit: {self.prefetch_hit_rate:.1%} | "
            f"SSD loads: {self.ssd_loads} ({self.ssd_load_time_ms:.1f}ms total) | "
            f"GPU: {self.gpu_compute_time_ms:.1f}ms"
        )


class GraphStateCache:
    """Tiered cache for fractal graph states (RAM + SSD).
    
    Hot levels stay in RAM. Cold levels are serialized to SSD
    and loaded on demand or via D-stream prefetch.
    """
    
    def __init__(self, config: TieredMemoryConfig):
        self.config = config
        self.ram_cache: Dict[int, dict] = {}  # level -> graph state
        self.ssd_paths: Dict[int, str] = {}   # level -> file path
        self._prefetch_queue: set = set()      # levels being prefetched
        
        os.makedirs(config.ssd_cache_dir, exist_ok=True)
    
    def put(self, level: int, state: dict):
        """Store a graph state. Hot levels go to RAM, cold to SSD."""
        if level < self.config.hot_levels:
            self.ram_cache[level] = state
        else:
            # Serialize to SSD
            path = os.path.join(self.config.ssd_cache_dir, f"level_{level}.npz")
            self._save_state(state, path)
            self.ssd_paths[level] = path
            # Also keep in RAM if budget allows
            if self._ram_usage_ok():
                self.ram_cache[level] = state
    
    def get(self, level: int, stats: InferenceStats) -> Optional[dict]:
        """Get a graph state. Check RAM first, then SSD."""
        # RAM hit
        if level in self.ram_cache:
            stats.prefetch_hits += 1
            return self.ram_cache[level]
        
        # SSD load
        if level in self.ssd_paths:
            stats.prefetch_misses += 1
            t0 = time.perf_counter()
            state = self._load_state(self.ssd_paths[level])
            load_ms = (time.perf_counter() - t0) * 1000
            stats.ssd_loads += 1
            stats.ssd_load_time_ms += load_ms
            
            # Promote to RAM cache
            if self._ram_usage_ok():
                self.ram_cache[level] = state
            
            return state
        
        return None
    
    def prefetch(self, level: int, stats: InferenceStats):
        """Prefetch a level from SSD to RAM (sync for now, async TODO)."""
        if level in self.ram_cache:
            return  # Already hot
        if level in self.ssd_paths:
            t0 = time.perf_counter()
            state = self._load_state(self.ssd_paths[level])
            load_ms = (time.perf_counter() - t0) * 1000
            stats.ssd_loads += 1
            stats.ssd_load_time_ms += load_ms
            self.ram_cache[level] = state
    
    def evict(self, level: int):
        """Evict a level from RAM (keeps SSD copy)."""
        if level in self.ram_cache and level >= self.config.hot_levels:
            # Save to SSD first if not already there
            if level not in self.ssd_paths:
                path = os.path.join(self.config.ssd_cache_dir, f"level_{level}.npz")
                self._save_state(self.ram_cache[level], path)
                self.ssd_paths[level] = path
            del self.ram_cache[level]
    
    def _ram_usage_ok(self) -> bool:
        """Check if we're within RAM budget (rough estimate)."""
        # Rough: count cached levels * estimated state size
        return len(self.ram_cache) < self.config.hot_levels + 2
    
    def _save_state(self, state: dict, path: str):
        """Serialize graph state to SSD."""
        arrays = {}
        for key, val in state.items():
            if isinstance(val, mx.array):
                arrays[key] = np.array(val)
        np.savez(path, **arrays)
    
    def _load_state(self, path: str) -> dict:
        """Load graph state from SSD."""
        data = np.load(path)
        return {key: mx.array(data[key]) for key in data.files}


class HeterogeneousEngine:
    """PID-Net inference engine with heterogeneous compute + tiered memory.
    
    Usage:
        engine = HeterogeneousEngine(model, config)
        tokens = engine.generate(prompt_ids, max_tokens=100)
        print(engine.stats.summary())
    """
    
    def __init__(self, model, config: TieredMemoryConfig = None):
        self.model = model
        self.config = config or TieredMemoryConfig()
        self.cache = GraphStateCache(self.config)
        self.stats = InferenceStats()
    
    def generate(
        self,
        prompt_tokens: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        repetition_penalty: float = 1.5,
        penalty_window: int = 32,
    ) -> mx.array:
        """Generate tokens with heterogeneous compute and D-stream prefetching.
        
        The generation loop is split across compute tiers:
        1. CPU: Token management, repetition penalty, scheduling
        2. GPU: PID rewriting (message passing, edge evolution, fractal ops)
        3. Prefetch: D-stream error drives SSD→RAM prefetching
        """
        from collections import Counter
        
        tokens = prompt_tokens.tolist()[0] if prompt_tokens.ndim > 1 else prompt_tokens.tolist()
        generated = list(tokens)
        
        for step in range(max_new_tokens):
            if len(generated) >= self.model.max_nodes:
                break
            
            t_start = time.perf_counter()
            
            # === GPU: Forward pass (PID rewriting) — COMPILED ===
            input_tokens = mx.array([generated])
            t_gpu_start = time.perf_counter()
            
            # Use compiled forward if available
            if hasattr(self.model, '_compiled_forward'):
                if not hasattr(self, '_forward_fn'):
                    try:
                        self._forward_fn = mx.compile(self.model._compiled_forward)
                        # Warmup trace
                        _ = self._forward_fn(input_tokens)
                        mx.eval(_)
                    except Exception:
                        self._forward_fn = self.model._compiled_forward
                last_logits = self._forward_fn(input_tokens)[0]
                diagnostics = {}
            else:
                logits, diagnostics = self.model(input_tokens)
                last_logits = logits[0, -1]
            
            mx.eval(last_logits)
            gpu_ms = (time.perf_counter() - t_gpu_start) * 1000
            self.stats.gpu_compute_time_ms += gpu_ms
            
            # === D-STREAM PREFETCH CHECK ===
            if 'pred_error' in diagnostics:
                pred_error_norm = float(diagnostics['pred_error'])
                if pred_error_norm > self.config.prefetch_threshold:
                    # High surprise → prefetch next fractal level
                    for level in range(self.config.hot_levels, 
                                      getattr(self.model, 'n_levels', 3)):
                        self.cache.prefetch(level, self.stats)
            
            # === CPU: Repetition penalty ===
            if repetition_penalty > 1.0:
                recent = generated[-penalty_window:]
                freq = Counter(recent)
                for token_id, count in freq.items():
                    if token_id < last_logits.shape[0]:
                        val = last_logits[token_id].item()
                        pen = repetition_penalty ** count
                        if val > 0:
                            last_logits = last_logits.at[token_id].add(
                                val * (1.0 / pen - 1.0))
                        else:
                            last_logits = last_logits.at[token_id].add(
                                val * (pen - 1.0))
            
            # === CPU: Sampling ===
            last_logits = last_logits / temperature
            if top_k > 0:
                top_k_val = mx.sort(last_logits)[-top_k]
                last_logits = mx.where(
                    last_logits < top_k_val, float('-inf'), last_logits)
            
            probs = mx.softmax(last_logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10)).item()
            generated.append(next_token)
            
            # === Stats ===
            total_ms = (time.perf_counter() - t_start) * 1000
            self.stats.total_tokens += 1
            self.stats.total_time_ms += total_ms
        
        return mx.array(generated)
    
    def reset_stats(self):
        """Reset inference statistics."""
        self.stats = InferenceStats()
    
    def benchmark(self, prompt_tokens: mx.array, n_tokens: int = 50) -> str:
        """Run a benchmark and return performance summary."""
        self.reset_stats()
        _ = self.generate(prompt_tokens, max_new_tokens=n_tokens)
        return self.stats.summary()
