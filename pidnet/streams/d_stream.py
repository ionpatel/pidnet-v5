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
  - Prediction error naturally decreases on predictable sequences
    and spikes on novel/surprising input

The D-stream also drives:
  - I-stream write strength (D→I coupling): surprise → remember
  - Adaptive compute: high error → more rewriting steps
  - Chunk boundaries: high error → segment break (Phase 4)
"""

import mlx.core as mx
import mlx.nn as nn

from typing import Tuple


class DStream(nn.Module):
    """Derivative stream — prediction error as surprise signal.
    
    Maintains a predictor that learns to forecast the next graph state.
    The ERROR between prediction and reality is the D-stream output.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Predictor: predict next state from current
        self.predictor = nn.Linear(d_model, d_model)
        
        # Error projection: transform raw error into useful signal
        self.W_err = nn.Linear(d_model, d_model)
        
        # Output projection
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        # Stagnation detector: when error is too LOW, inject perturbation
        # (prevents repetition collapse — learned from v3)
        self.kickout_vector = mx.random.normal((d_model,)) * 0.1
        self.stagnation_threshold = 0.05  # if error norm < this, inject kickout
    
    def __call__(
        self,
        nodes: mx.array,           # [batch, N, d] — current (actual) state
        prediction: mx.array,      # [batch, N, d] — predicted state (from last step)
        mask: mx.array,            # [batch, N]
    ) -> Tuple[mx.array, mx.array]:
        """
        Compute prediction error and generate new prediction.
        
        Args:
            nodes: Actual current node features
            prediction: What we predicted these features would be
            mask: Active node mask
        
        Returns:
            (D-stream output [batch, N, d],
             new prediction for next step [batch, N, d])
        """
        mask_expanded = mx.expand_dims(mask, -1)  # [batch, N, 1]
        
        # === PREDICTION ERROR ===
        # ε = actual - predicted (what surprised us)
        epsilon = nodes - prediction  # [batch, N, d]
        epsilon = epsilon * mask_expanded
        
        # Error magnitude per node (for stagnation detection)
        error_norm = mx.sqrt(mx.sum(epsilon ** 2, axis=-1, keepdims=True) + 1e-8)
        # [batch, N, 1]
        
        # Stagnation detection: if error is too small, inject perturbation
        # This prevents the PID system from settling into repetition
        is_stagnant = (error_norm < self.stagnation_threshold).astype(mx.float32)
        kickout = mx.broadcast_to(self.kickout_vector, epsilon.shape)
        epsilon = epsilon + is_stagnant * kickout * 0.1  # gentle perturbation
        
        # Project error into useful signal
        d_signal = self.W_err(epsilon)  # [batch, N, d]
        
        # Output projection + norm
        output = self.norm(self.proj(d_signal))
        output = output * mask_expanded
        
        # === NEW PREDICTION ===
        # Predict what the NEXT state will look like
        new_prediction = self.predictor(nodes)  # [batch, N, d]
        new_prediction = new_prediction * mask_expanded
        
        return output, new_prediction
    
    def prediction_loss(
        self,
        actual: mx.array,
        predicted: mx.array,
        mask: mx.array,
    ) -> mx.array:
        """Auxiliary loss to train the predictor.
        
        This loss is added to the main training loss to help the
        predictor learn to forecast graph evolution.
        
        Returns: scalar loss
        """
        mask_expanded = mx.expand_dims(mask, -1)
        error = (actual - predicted) ** 2
        error = error * mask_expanded
        n_active = mx.sum(mask) + 1e-8
        return mx.sum(error) / n_active
