"""Dynamic Driver Annoyance Tracker (R7).

Maintains a sliding 10-minute history of alerts and computes an escalating
annoyance penalty that increases false-alarm costs (proposal Section 3.2 & L3).
Prevents alert fatigue and driver nuisance shutdowns.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Deque, List, Tuple

from .utility_matrix import AlertAction


@dataclass
class AlertRecord:
    timestamp_s: float
    action: AlertAction
    intensity_weight: float


class AnnoyanceTracker:
    """Tracks driver alert history over a 10-minute window (600 s) and computes annoyance."""

    def __init__(
        self,
        window_duration_s: float = 600.0,  # 10 minutes per proposal
        penalty_per_alert: float = 0.40,
        decay_exponent: float = 1.0,
    ):
        """
        Parameters
        ----------
        window_duration_s:
            Sliding window in seconds over which alert history affects annoyance.
        penalty_per_alert:
            Scale factor for additive cost penalty per recent alert event.
        decay_exponent:
            Nonlinear escalation power (>= 1.0).
        """
        self.window_duration_s = window_duration_s
        self.penalty_per_alert = penalty_per_alert
        self.decay_exponent = decay_exponent

        self._history: Deque[AlertRecord] = collections.deque()

        # Intensity weights per action
        self._action_weights = {
            AlertAction.NONE: 0.0,
            AlertAction.SOFT_VISUAL: 0.25,   # Unobtrusive
            AlertAction.VISUAL_AUDIO: 1.0,   # Standard chime
            AlertAction.URGENT: 2.0,         # Jarring loud tone
        }

    def record_alert(self, timestamp_s: float, action: AlertAction) -> None:
        """Record an alert action taken by the system."""
        if action == AlertAction.NONE:
            return

        weight = self._action_weights.get(action, 1.0)
        self._history.append(AlertRecord(timestamp_s, action, weight))
        self._prune(timestamp_s)

    def _prune(self, current_ts: float) -> None:
        """Remove alerts older than the sliding window."""
        cutoff = current_ts - self.window_duration_s
        while self._history and self._history[0].timestamp_s < cutoff:
            self._history.popleft()

    def get_annoyance_penalty(self, current_ts: float) -> float:
        """Compute current annoyance penalty based on alert history in the last 10 minutes.

        Returns
        -------
        float:
            Additive cost penalty >= 0.0 to be added to false-alarm costs.
        """
        self._prune(current_ts)
        if not self._history:
            return 0.0

        # Weighted sum of recent alerts with linear time-recency weighting
        total_effective_alerts = 0.0
        for rec in self._history:
            age_s = current_ts - rec.timestamp_s
            recency = max(0.0, 1.0 - (age_s / self.window_duration_s))
            total_effective_alerts += rec.intensity_weight * (0.5 + 0.5 * recency)

        penalty = self.penalty_per_alert * (total_effective_alerts ** self.decay_exponent)
        return float(penalty)

    def count_recent_alerts(self, current_ts: float) -> int:
        """Return the count of distinct alerts fired in the last 10 minutes."""
        self._prune(current_ts)
        return len(self._history)

    def reset(self) -> None:
        self._history.clear()
