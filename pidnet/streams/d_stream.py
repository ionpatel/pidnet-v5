"""
D-Stream (Derivative / Differentiation / Prediction Error)

"What SURPRISED me about the current state?"

The D-stream implements predictive coding:
  1. At step t-1, predict what step t should look like
  2. At step t, compute prediction error: ε = actual - predicted
  3. Only propagate the ERROR (not the full state)

Why prediction error, not finite difference:
  - Finite diff (Xₜ - Xₜ₋₁) measures raw change — noisy, uninformative
  - Prediction error measures SURPRISE — deviation from expectations
  - The brain does predictive coding: top-down predictions, bottom-up errors
  - Only error propagates → 10-100x more efficient than full states

STAGNATION-AWARE D-GATE (v3 lesson, v5 upgrade):
  When the PID system enters repetition equilibrium:
    - D→0 (predictor correctly predicts repeated tokens)
    - I accumulates the pattern (reinforcing it)
    - P tracks it (following it)
  This is a stable fixed point. The D-gate must DETECT stagnation
  and inject artificial surprise to break the equilibrium.
  
  Detection: cosine similarity between consecutive states.
  If cos_sim > threshold → inject learned kickout perturbation.
  The kickout is LEARNED (not random) and GATED (model controls strength).
"""

import mlx.core as mx
import mlx.nn as nn

from typing import Tuple, Optional


class DStream(nn.Module):
    """Derivative stream — prediction error as surprise signal.
    
    Maintains a predictor that learns to forecast the next graph state.
    The ERROR between prediction and reality is the D-stream output.
    
    Includes stagnation-aware kickout (ported from v3, improved for v5):
    - Cosine similarity detection between current and predicted states
    - Learned kickout vector with learned gating
    - Diversity penalty: detects when neighboring nodes become too similar
    """
    
    def __init__(self, d_model: int, stagnation_threshold: float = 0.95):
        super().__init__()
        self.d_model = d_model
        self.stagnation_threshold = stagnation_threshold
        
        # Predictor: predict next state from current (2-layer for better predictions)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # Error projection: transform raw error into useful signal
        self.W_err = nn.Linear(d_model, d_model)
        
        # Output projection
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        # === STAGNATION-AWARE KICKOUT (v3 → v5) ===
        
        # Learned kickout vector: the perturbation to inject when stagnant
        # Initialized small — the model learns what kind of surprise is useful
        self.kickout_proj = nn.Linear(d_model, d_model)
        
        # Learned gate: how much kickout to apply (starts conservative)
        # Input: current node features → scalar strength
        self.kickout_gate = nn.Linear(d_model, 1)
        
        # Diversity detector: penalize when consecutive nodes are too similar
        # This catches repetition at the feature level, not just prediction error
        self.diversity_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Track stagnation rate for diagnostics
        self._last_stagnation_rate = 0.0
    
    def _detect_stagnation(
        self,
        nodes: mx.array,        # [batch, N, d]
        prediction: mx.array,   # [batch, N, d]
        mask: mx.array,         # [batch, N]
    ) -> Tuple[mx.array, mx.array]:
        """
        Detect stagnation via cosine similarity between actual and predicted.
        
        High cos_sim (> threshold) means the predictor is TOO GOOD at
        predicting the state — which happens during repetition.
        
        Also detects DIVERSITY collapse: when consecutive nodes have
        nearly identical features (the model is outputting the same thing).
        
        Returns:
            stagnation_mask: [batch, N, 1] — 1.0 where stagnant
            diversity_penalty: [batch, N, d] — perturbation for similar neighbors
        """
        mask_expanded = mx.expand_dims(mask, -1)  # [batch, N, 1]
        
        # === PREDICTION STAGNATION ===
        # Cosine similarity between actual and predicted
        # High cos_sim = predictor is accurate = state is predictable = stagnant
        nodes_norm = mx.sqrt(mx.sum(nodes ** 2, axis=-1, keepdims=True) + 1e-8)
        pred_norm = mx.sqrt(mx.sum(prediction ** 2, axis=-1, keepdims=True) + 1e-8)
        cos_sim = mx.sum(nodes * prediction, axis=-1, keepdims=True) / (nodes_norm * pred_norm)
        # cos_sim: [batch, N, 1]
        
        # Stagnation where cos_sim > threshold
        stagnation = (cos_sim > self.stagnation_threshold).astype(mx.float32)
        stagnation = stagnation * mask_expanded
        
        # Track for diagnostics
        self._last_stagnation_rate = mx.mean(stagnation).item()
        
        # === DIVERSITY COLLAPSE ===
        # Check if consecutive nodes are too similar (token repetition at feature level)
        # shifted[i] = nodes[i-1] (with zero padding at position 0)
        batch, N, d = nodes.shape
        shifted = mx.concatenate([mx.zeros((batch, 1, d)), nodes[:, :-1, :]], axis=1)
        
        # Cosine similarity between consecutive nodes
        shifted_norm = mx.sqrt(mx.sum(shifted ** 2, axis=-1, keepdims=True) + 1e-8)
        neighbor_cos = mx.sum(nodes * shifted, axis=-1, keepdims=True) / (nodes_norm * shifted_norm)
        
        # High similarity between consecutive nodes → diversity collapsed
        neighbor_stagnant = (neighbor_cos > self.stagnation_threshold).astype(mx.float32)
        neighbor_stagnant = neighbor_stagnant * mask_expanded
        
        # Combined stagnation (either prediction is too accurate OR neighbors are too similar)
        combined_stagnation = mx.maximum(stagnation, neighbor_stagnant)
        
        # Diversity perturbation: project the difference to create useful signal
        diff = nodes - shifted
        diversity_penalty = self.diversity_proj(diff) * neighbor_stagnant
        
        return combined_stagnation, diversity_penalty
    
    def __call__(
        self,
        nodes: mx.array,           # [batch, N, d] — current (actual) state
        prediction: mx.array,      # [batch, N, d] — predicted state (from last step)
        mask: mx.array,            # [batch, N]
    ) -> Tuple[mx.array, mx.array]:
        """
        Compute prediction error with stagnation-aware kickout.
        
        When the predictor is too accurate (cos_sim > 0.95) or consecutive
        nodes are too similar, inject a learned perturbation to break
        the repetition equilibrium.
        """
        mask_expanded = mx.expand_dims(mask, -1)  # [batch, N, 1]
        
        # === PREDICTION ERROR ===
        epsilon = nodes - prediction  # [batch, N, d]
        epsilon = epsilon * mask_expanded
        
        # === STAGNATION DETECTION ===
        stagnation_mask, diversity_penalty = self._detect_stagnation(
            nodes, prediction, mask
        )
        
        # === LEARNED KICKOUT ===
        # Generate content-dependent perturbation (not random!)
        kickout = self.kickout_proj(nodes)  # [batch, N, d] — perturbation based on current state
        
        # Gate: model decides how much kickout to apply
        # Starts conservative (the bias should be initialized negative)
        kickout_strength = mx.sigmoid(self.kickout_gate(nodes))  # [batch, N, 1]
        
        # Apply kickout where stagnant
        gated_kickout = stagnation_mask * kickout_strength * kickout  # [batch, N, d]
        
        # Inject kickout + diversity penalty into error signal
        epsilon = epsilon + gated_kickout + diversity_penalty * 0.5
        
        # Project error into useful signal
        d_signal = self.W_err(epsilon)  # [batch, N, d]
        
        # Output projection + norm
        output = self.norm(self.proj(d_signal))
        output = output * mask_expanded
        
        # === NEW PREDICTION ===
        new_prediction = self.predictor(nodes)  # [batch, N, d]
        new_prediction = new_prediction * mask_expanded
        
        return output, new_prediction
    
    @property
    def stagnation_rate(self) -> float:
        """Return the last computed stagnation rate for monitoring."""
        return self._last_stagnation_rate
    
    def prediction_loss(
        self,
        actual: mx.array,
        predicted: mx.array,
        mask: mx.array,
    ) -> mx.array:
        """Auxiliary loss to train the predictor."""
        mask_expanded = mx.expand_dims(mask, -1)
        error = (actual - predicted) ** 2
        error = error * mask_expanded
        n_active = mx.sum(mask) + 1e-8
        return mx.sum(error) / n_active
