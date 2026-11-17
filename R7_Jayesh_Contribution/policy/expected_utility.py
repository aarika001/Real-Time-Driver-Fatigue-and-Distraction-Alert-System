"""Expected-Utility Alert Policy Engine (R7).

Implements the rational agent decision rule:
    a* = argmax_a sum_s P(s) U(s, a)
under uncertainty, trading off safety-critical miss costs against false-alarm
intrusiveness and driver annoyance penalties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .utility_matrix import AlertAction, DriverState, UtilityMatrix
from .annoyance import AnnoyanceTracker
from ..reliability.degraded_state import DegradedStatus


@dataclass
class DecisionOutput:
    """Diagnostic output returned for each evaluated frame / temporal window."""
    timestamp_s: float
    selected_action: AlertAction
    action_name: str
    expected_utilities: Dict[str, float]
    state_probabilities: Dict[str, float]
    annoyance_penalty: float
    is_monitoring_degraded: bool
    status_message: str


class ExpectedUtilityAlertPolicy:
    """Utility-based rational agent policy maximizing EU(a) = sum_s P(s) U(s, a)."""

    def __init__(
        self,
        cost_ratio: float = 10.0,
        base_false_alarm_cost: float = 1.0,
        annoyance_tracker: Optional[AnnoyanceTracker] = None,
        min_alert_cooldown_s: float = 1.0,
    ):
        """
        Parameters
        ----------
        cost_ratio:
            The ratio rho = C_miss / C_fa (swept 3:1 to 20:1; default: 10.0).
        base_false_alarm_cost:
            Base false alarm cost for non-urgent alert.
        annoyance_tracker:
            Annoyance tracker instance (created with defaults if None).
        min_alert_cooldown_s:
            Minimum spacing in seconds before escalating or re-firing loud alerts.
        """
        self.cost_ratio = cost_ratio
        self.base_fa_cost = base_false_alarm_cost
        self.annoyance = annoyance_tracker or AnnoyanceTracker()
        self.min_alert_cooldown_s = min_alert_cooldown_s

        self._last_alert_ts: float = -999.0
        self._last_action: AlertAction = AlertAction.NONE

    def decide(
        self,
        state_probs: Sequence[float] | np.ndarray,
        timestamp_s: float,
        degraded_status: Optional[DegradedStatus] = None,
    ) -> DecisionOutput:
        """Select action maximizing expected utility given state belief distribution P(s).

        Parameters
        ----------
        state_probs:
            [P(ALERT), P(FATIGUE), P(DISTRACTION)] from GRU / temporal model. Must sum to ~1.0.
        timestamp_s:
            Current timestamp in seconds.
        degraded_status:
            Output from MonitoringDegradedTracker.

        Returns
        -------
        DecisionOutput
        """
        probs = np.asarray(state_probs, dtype=np.float32)
        assert len(probs) == 3, f"Expected 3 probabilities, got {len(probs)}"

        # Normalize probabilities in case of slight numerical drift
        total_p = float(np.sum(probs))
        if total_p > 0:
            probs = probs / total_p
        else:
            probs = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Check monitoring degraded status
        is_degraded = degraded_status.is_degraded if degraded_status else False

        # If system has been degraded for sustained T seconds, show banner and suppress ungrounded alarms
        if is_degraded:
            return DecisionOutput(
                timestamp_s=timestamp_s,
                selected_action=AlertAction.NONE,
                action_name="MONITORING_DEGRADED",
                expected_utilities={a.name: 0.0 for a in AlertAction},
                state_probabilities={
                    "ALERT": float(probs[0]),
                    "FATIGUE": float(probs[1]),
                    "DISTRACTION": float(probs[2]),
                },
                annoyance_penalty=self.annoyance.get_annoyance_penalty(timestamp_s),
                is_monitoring_degraded=True,
                status_message="Monitoring Degraded: sensors occluded or unreliable.",
            )

        # Compute dynamic annoyance penalty
        cur_annoyance = self.annoyance.get_annoyance_penalty(timestamp_s)

        # Construct current utility matrix with dynamic annoyance
        u_matrix = UtilityMatrix(
            cost_ratio=self.cost_ratio,
            base_false_alarm_cost=self.base_fa_cost,
            annoyance_penalty=cur_annoyance,
        )

        # Compute EU(a) for all candidate actions: (4,) array
        eu = u_matrix.expected_utilities(probs)

        # Select action maximizing EU
        best_action_idx = int(np.argmax(eu))
        best_action = AlertAction(best_action_idx)

        # Cooldown check: if an alert was triggered very recently (< min_alert_cooldown_s),
        # prevent fluttering back and forth, but ALWAYS allow immediate escalation if risk increases
        if best_action != AlertAction.NONE:
            if (timestamp_s - self._last_alert_ts) < self.min_alert_cooldown_s:
                if best_action.value > self._last_action.value:
                    self._last_action = best_action
                else:
                    best_action = self._last_action
            else:
                self._last_alert_ts = timestamp_s
                self._last_action = best_action
                self.annoyance.record_alert(timestamp_s, best_action)
        else:
            self._last_action = AlertAction.NONE

        action_names = [a.name for a in AlertAction]
        eu_dict = {name: float(val) for name, val in zip(action_names, eu)}

        return DecisionOutput(
            timestamp_s=timestamp_s,
            selected_action=best_action,
            action_name=best_action.name,
            expected_utilities=eu_dict,
            state_probabilities={
                "ALERT": float(probs[0]),
                "FATIGUE": float(probs[1]),
                "DISTRACTION": float(probs[2]),
            },
            annoyance_penalty=cur_annoyance,
            is_monitoring_degraded=False,
            status_message=f"Optimal Action: {best_action.name} (EU={eu[best_action_idx]:.2f})",
        )

    def set_cost_ratio(self, new_ratio: float) -> None:
        """Update miss-to-false-alarm cost ratio (used during cost sweeps)."""
        self.cost_ratio = float(new_ratio)
