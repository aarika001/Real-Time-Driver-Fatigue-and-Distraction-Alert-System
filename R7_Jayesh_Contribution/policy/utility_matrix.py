"""Decision-Theoretic Utility Matrix U(s, a) (R7).

Formalizes the von Neumann-Morgenstern (vNM) utility / cost matrix across
driver states and alert actions, parameterized by the miss-to-false-alarm cost ratio
and dynamic driver annoyance penalties.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np


class DriverState(IntEnum):
    ALERT = 0
    FATIGUE = 1
    DISTRACTION = 2


class AlertAction(IntEnum):
    NONE = 0
    SOFT_VISUAL = 1
    VISUAL_AUDIO = 2
    URGENT = 3


class UtilityMatrix:
    """Constructs and manages the |S| x |A| utility matrix U(s, a).

    Conventions:
      - Utilities are non-positive: U(s, a) = -Cost(s, a).
      - Optimal match yields 0.0 penalty.
      - Miss cost >> False-Alarm cost (ratio rho).
      - Annoyance penalty increases the false alarm cost for non-NONE actions when state is ALERT.
    """

    def __init__(
        self,
        cost_ratio: float = 10.0,
        base_false_alarm_cost: float = 1.0,
        annoyance_penalty: float = 0.0,
    ):
        """
        Parameters
        ----------
        cost_ratio:
            The ratio rho = C_miss / C_fa (proposal sweeps 3:1 to 20:1; default: 10.0).
        base_false_alarm_cost:
            Base false alarm cost for a medium visual+audio alert on an alert driver.
        annoyance_penalty:
            Dynamic additive penalty added to false alarm costs due to recent frequent alerts.
        """
        self.cost_ratio = cost_ratio
        self.base_fa_cost = base_false_alarm_cost
        self.annoyance_penalty = annoyance_penalty

    @property
    def miss_cost(self) -> float:
        """Cost of missing a safety event (fatigue or sustained distraction)."""
        return self.cost_ratio * self.base_fa_cost

    def get_matrix(self) -> np.ndarray:
        """Compute and return the (|S|, |A|) = (3, 4) utility matrix U(s, a).

        Returns
        -------
        np.ndarray of shape (3, 4), dtype float32
        """
        # Costs for ALERT driver (false alarms):
        # Taking no action has 0 cost.
        # Cost escalates with alert intensity + annoyance penalty
        c_fa_soft = 0.3 * self.base_fa_cost + 0.5 * self.annoyance_penalty
        c_fa_med = 1.0 * self.base_fa_cost + 1.0 * self.annoyance_penalty
        c_fa_urgent = 2.5 * self.base_fa_cost + 2.0 * self.annoyance_penalty

        # Costs for FATIGUE driver:
        # Missing completely incurs full miss cost.
        # Soft visual is under-alerting (40% miss penalty).
        # Visual+Audio is adequate (10% penalty).
        # Urgent is optimal (0.0 penalty).
        c_miss = self.miss_cost
        c_fatigue_none = c_miss
        c_fatigue_soft = 0.40 * c_miss
        c_fatigue_med = 0.10 * c_miss
        c_fatigue_urgent = 0.0

        # Costs for DISTRACTION driver:
        # Missing completely incurs full miss cost.
        # Soft visual redirects gaze well (15% penalty).
        # Visual+audio is optimal (0.0 penalty).
        # Urgent is slightly jarring for simple distraction (5% penalty).
        c_dist_none = c_miss
        c_dist_soft = 0.15 * c_miss
        c_dist_med = 0.0
        c_dist_urgent = 0.05 * c_miss

        # Matrix: rows = [ALERT, FATIGUE, DISTRACTION], cols = [NONE, SOFT_VISUAL, VISUAL_AUDIO, URGENT]
        cost_matrix = np.array(
            [
                [0.0, c_fa_soft, c_fa_med, c_fa_urgent],           # ALERT
                [c_fatigue_none, c_fatigue_soft, c_fatigue_med, c_fatigue_urgent],  # FATIGUE
                [c_dist_none, c_dist_soft, c_dist_med, c_dist_urgent],              # DISTRACTION
            ],
            dtype=np.float32,
        )

        # Utility is negative cost
        return -cost_matrix

    def expected_utilities(self, state_probs: np.ndarray) -> np.ndarray:
        """Compute EU(a) = sum_s P(s) * U(s, a) for all actions.

        Parameters
        ----------
        state_probs:
            Array of shape (3,) with P(ALERT), P(FATIGUE), P(DISTRACTION). Must sum to ~1.0.

        Returns
        -------
        np.ndarray of shape (4,), expected utilities for [NONE, SOFT_VISUAL, VISUAL_AUDIO, URGENT].
        """
        probs = np.asarray(state_probs, dtype=np.float32)
        assert probs.shape == (3,), f"Expected shape (3,), got {probs.shape}"
        U = self.get_matrix()  # (3, 4)
        return np.dot(probs, U)  # (4,)
