"""Reliability-Aware Multi-Cue Fusion (R7).

Combines sensory cues (eye closure, mouth MAR / yawn, head deviation)
weighted dynamically by their learned reliability scores, down-weighting degraded
sensors (e.g. eye under glare, mouth when turning head).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class FusedPercept:
    """Multi-cue percept representation for downstream temporal modeling (R6) and policy (R7)."""
    fused_fatigue_score: float      # Composite fatigue indicator [0, 1]
    fused_distraction_score: float  # Composite distraction indicator [0, 1]
    cue_weights: Dict[str, float]   # Normalized weights assigned to each cue
    cue_reliabilities: Dict[str, float]
    all_unreliable: bool            # True if all cues fall below minimal reliability threshold


class ReliabilityAwareFusion:
    """Down-weights unreliable cues dynamically according to their predicted quality."""

    def __init__(self, min_reliability_floor: float = 0.25):
        """
        Parameters
        ----------
        min_reliability_floor:
            Reliability threshold below which a cue is treated as uninformative.
        """
        self.min_reliability_floor = min_reliability_floor

    def fuse(
        self,
        p_closed: float,
        mar_norm: float,
        head_angle_z: float,
        reliabilities: Dict[str, float],
    ) -> FusedPercept:
        """Fuse eye, mouth, and head pose into fatigue and distraction scores.

        Parameters
        ----------
        p_closed:
            Eye closure probability [0, 1] (or NaN if missing).
        mar_norm:
            Driver-calibrated mouth aspect ratio (~1.0 normal, >1.8-2.5 yawn).
        head_angle_z:
            Head pose deviation from neutral in z-scores.
        reliabilities:
            Dict containing 'eye', 'mouth', 'head_pose' reliability in [0, 1].

        Returns
        -------
        FusedPercept
        """
        r_eye = max(reliabilities.get("eye", 0.0), 0.0)
        r_mouth = max(reliabilities.get("mouth", 0.0), 0.0)
        r_head = max(reliabilities.get("head_pose", 0.0), 0.0)

        # Check if all cues are below reliability floor
        all_unreliable = (r_eye < self.min_reliability_floor and
                          r_mouth < self.min_reliability_floor and
                          r_head < self.min_reliability_floor)

        # Base nominal fatigue cues:
        # Eye closure (scaled [0, 1])
        c_eye = p_closed if (not np.isnan(p_closed) and r_eye >= self.min_reliability_floor) else 0.0

        # Mouth yawn cue (MAR norm scaled: 1.0 -> 0.0, 2.5 -> 1.0)
        if not np.isnan(mar_norm) and r_mouth >= self.min_reliability_floor:
            c_mouth = float(np.clip((mar_norm - 1.2) / 1.3, 0.0, 1.0))
        else:
            c_mouth = 0.0

        # Head pitch nod / drop cue (head_angle_z contributing to drowsiness or distraction)
        c_head = float(np.clip(head_angle_z / 3.0, 0.0, 1.0)) if r_head >= self.min_reliability_floor else 0.0

        # Dynamic weights for fatigue:
        # Eye closure is primary (prior 0.55), yawn is secondary (prior 0.35), head drop is tertiary (prior 0.10)
        # Cues below the reliability floor are suppressed (weight = 0.0)
        w_eye = (0.55 * r_eye) if (not np.isnan(p_closed) and r_eye >= self.min_reliability_floor) else 0.0
        w_mouth = (0.35 * r_mouth) if (not np.isnan(mar_norm) and r_mouth >= self.min_reliability_floor) else 0.0
        w_head = (0.10 * r_head) if r_head >= self.min_reliability_floor else 0.0

        total_fatigue_w = w_eye + w_mouth + w_head
        if total_fatigue_w > 1e-6:
            fused_fatigue = (w_eye * c_eye + w_mouth * c_mouth + w_head * c_head) / total_fatigue_w
        else:
            fused_fatigue = 0.0

        # Distraction score: primarily driven by head pose off-axis, modulated by eye reliability
        # If head angle is high and head pose is reliable -> high distraction
        if r_head >= self.min_reliability_floor:
            fused_distraction = float(np.clip(head_angle_z / 2.5, 0.0, 1.0))
        else:
            fused_distraction = 0.0

        norm_weights = {
            "eye": float(w_eye / total_fatigue_w) if total_fatigue_w > 0 else 0.0,
            "mouth": float(w_mouth / total_fatigue_w) if total_fatigue_w > 0 else 0.0,
            "head_pose": float(w_head / total_fatigue_w) if total_fatigue_w > 0 else 0.0,
        }

        return FusedPercept(
            fused_fatigue_score=float(np.clip(fused_fatigue, 0.0, 1.0)),
            fused_distraction_score=float(np.clip(fused_distraction, 0.0, 1.0)),
            cue_weights=norm_weights,
            cue_reliabilities=reliabilities,
            all_unreliable=all_unreliable,
        )
