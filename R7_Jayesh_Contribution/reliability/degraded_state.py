"""Monitoring Degraded State Machine (R7).

Manages temporal persistence of low-reliability states and raises the
'Monitoring Degraded' banner if all/overall reliability stays low for
T consecutive seconds (proposal requirement).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DegradedStatus:
    is_degraded: bool
    degraded_duration_s: float
    reason: str


class MonitoringDegradedTracker:
    """Tracks low-reliability persistence and fires 'monitoring degraded' after T seconds."""

    def __init__(
        self,
        threshold: float = 0.30,
        required_duration_s: float = 3.0,
        recovery_duration_s: float = 1.0,
    ):
        """
        Parameters
        ----------
        threshold:
            Overall reliability threshold below which monitoring is considered degraded.
        required_duration_s:
            Time (in seconds) overall reliability must remain below threshold before banner fires.
        recovery_duration_s:
            Time (in seconds) reliability must remain healthy before banner clears.
        """
        self.threshold = threshold
        self.required_duration_s = required_duration_s
        self.recovery_duration_s = recovery_duration_s

        self._low_start_ts: Optional[float] = None
        self._recovery_start_ts: Optional[float] = None
        self._is_degraded: bool = False
        self._last_ts: float = 0.0

    def update(self, overall_reliability: float, timestamp_s: float) -> DegradedStatus:
        """Update tracker with current frame's overall reliability and timestamp.

        Parameters
        ----------
        overall_reliability:
            Overall reliability score in [0, 1].
        timestamp_s:
            Current monotonic or frame timestamp in seconds.

        Returns
        -------
        DegradedStatus
        """
        self._last_ts = timestamp_s

        if overall_reliability < self.threshold:
            # Under degraded threshold
            self._recovery_start_ts = None
            if self._low_start_ts is None:
                self._low_start_ts = timestamp_s

            elapsed = timestamp_s - self._low_start_ts
            if elapsed >= self.required_duration_s:
                self._is_degraded = True
                return DegradedStatus(
                    is_degraded=True,
                    degraded_duration_s=elapsed,
                    reason=f"Reliability below {self.threshold:.2f} for {elapsed:.1f}s",
                )
            else:
                return DegradedStatus(
                    is_degraded=False,
                    degraded_duration_s=elapsed,
                    reason=f"Pending degradation ({elapsed:.1f}s / {self.required_duration_s:.1f}s)",
                )
        else:
            # Healthy reliability
            self._low_start_ts = None
            if self._is_degraded:
                if self._recovery_start_ts is None:
                    self._recovery_start_ts = timestamp_s

                rec_elapsed = timestamp_s - self._recovery_start_ts
                if rec_elapsed >= self.recovery_duration_s:
                    self._is_degraded = False
                    self._recovery_start_ts = None
                    return DegradedStatus(
                        is_degraded=False,
                        degraded_duration_s=0.0,
                        reason="Recovered",
                    )
                else:
                    return DegradedStatus(
                        is_degraded=True,
                        degraded_duration_s=0.0,
                        reason=f"Recovering ({rec_elapsed:.1f}s / {self.recovery_duration_s:.1f}s)",
                    )

            return DegradedStatus(is_degraded=False, degraded_duration_s=0.0, reason="Healthy")

    def reset(self) -> None:
        self._low_start_ts = None
        self._recovery_start_ts = None
        self._is_degraded = False
