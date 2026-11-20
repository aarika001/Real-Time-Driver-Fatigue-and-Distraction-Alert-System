"""Fixed-Threshold Baseline Policy (Ablation Baseline) (R7).

Implements the standard heuristic baseline:
  Fires a fixed alert whenever P(drowsy) or P(distracted) exceeds 0.5.
Ignores utility theory, asymmetric miss/false-alarm costs, and annoyance penalties.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .utility_matrix import AlertAction


class FixedThresholdBaselinePolicy:
    """Ablation baseline using fixed probability thresholding (default: 0.5)."""

    def __init__(self, threshold: float = 0.50):
        self.threshold = threshold

    def decide(self, state_probs: Sequence[float] | np.ndarray) -> AlertAction:
        """Return AlertAction.VISUAL_AUDIO if P(fatigue) + P(distraction) > threshold, else NONE."""
        probs = np.asarray(state_probs, dtype=np.float32)
        p_risk = float(probs[1] + probs[2])  # P(fatigue) + P(distraction)
        if p_risk > self.threshold:
            return AlertAction.VISUAL_AUDIO
        return AlertAction.NONE
