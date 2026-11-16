"""Cost Ratio Parameter Sweep (R7).

Sweeps the miss-to-false-alarm cost ratio (3:1 to 20:1) across validation data
to find the optimal operational point satisfying the project success criteria
(Recall >= 85%, False alarms/hour <= 3.0) per proposal Section 3.2 and 3.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..policy.expected_utility import ExpectedUtilityAlertPolicy
from ..policy.utility_matrix import AlertAction, DriverState


@dataclass
class SweepPointResult:
    cost_ratio: float
    event_recall: float
    false_alarms_per_hour: float
    miss_rate: float
    avg_annoyance_penalty: float
    meets_criteria: bool


@dataclass
class SweepResult:
    points: List[SweepPointResult]
    optimal_ratio: float
    recommendation_summary: str


class CostRatioSweep:
    """Evaluates ExpectedUtilityAlertPolicy across a grid of cost ratios rho in [3, 20]."""

    def __init__(
        self,
        ratios: Optional[Sequence[float]] = None,
        target_min_recall: float = 0.85,
        target_max_fa_per_hour: float = 3.0,
    ):
        self.ratios = list(ratios) if ratios else [3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
        self.target_min_recall = target_min_recall
        self.target_max_fa_per_hour = target_max_fa_per_hour

    def run_sweep(
        self,
        validation_sequences: Sequence[Dict[str, np.ndarray]],
        fps: float = 15.0,
    ) -> SweepResult:
        """Run cost sweep over validation sequence dictionaries.

        Each sequence dict should have:
          - 'state_probs': array of shape (N, 3)
          - 'ground_truth': array of shape (N,) with DriverState ints (0=alert, 1=fatigue, 2=distraction)
          - 'timestamps': array of shape (N,) in seconds

        Returns
        -------
        SweepResult
        """
        results: List[SweepPointResult] = []

        for rho in self.ratios:
            total_fatigue_distraction_frames = 0
            detected_events = 0
            missed_events = 0
            false_alarm_frames = 0
            total_alert_duration_s = 0.0
            total_annoyance = 0.0
            total_evaluated_frames = 0

            policy = ExpectedUtilityAlertPolicy(cost_ratio=rho)

            for seq in validation_sequences:
                probs = seq["state_probs"]
                gt = seq["ground_truth"]
                ts = seq["timestamps"]
                N = len(ts)

                in_risk_event = False
                event_detected = False

                for i in range(N):
                    t = float(ts[i])
                    p = probs[i]
                    true_state = int(gt[i])

                    out = policy.decide(p, t)
                    alert_fired = (out.selected_action != AlertAction.NONE)

                    total_annoyance += out.annoyance_penalty
                    total_evaluated_frames += 1

                    if true_state == DriverState.ALERT:
                        total_alert_duration_s += (1.0 / fps)
                        if alert_fired:
                            false_alarm_frames += 1

                    elif true_state in (DriverState.FATIGUE, DriverState.DISTRACTION):
                        if not in_risk_event:
                            # Start of a new risk event
                            in_risk_event = True
                            event_detected = False

                        if alert_fired:
                            event_detected = True
                    else:
                        if in_risk_event:
                            # End of risk event
                            if event_detected:
                                detected_events += 1
                            else:
                                missed_events += 1
                            in_risk_event = False

                # Handle event at end of sequence
                if in_risk_event:
                    if event_detected:
                        detected_events += 1
                    else:
                        missed_events += 1

            total_events = detected_events + missed_events
            recall = (detected_events / total_events) if total_events > 0 else 1.0
            miss_rate = 1.0 - recall

            # Normalize false alarms to rate per hour of normal ALERT time
            total_normal_hours = max(total_alert_duration_s / 3600.0, 1e-4)
            # Group consecutive false alarm frames into distinct alert episodes
            approx_fa_events = false_alarm_frames / (fps * 2.0)  # assume ~2s average alarm episode
            fa_per_hour = approx_fa_events / total_normal_hours

            avg_annoyance = total_annoyance / max(total_evaluated_frames, 1)
            meets = (recall >= self.target_min_recall and fa_per_hour <= self.target_max_fa_per_hour)

            results.append(
                SweepPointResult(
                    cost_ratio=rho,
                    event_recall=float(recall),
                    false_alarms_per_hour=float(fa_per_hour),
                    miss_rate=float(miss_rate),
                    avg_annoyance_penalty=float(avg_annoyance),
                    meets_criteria=meets,
                )
            )

        # Select optimal ratio: smallest ratio meeting criteria, or highest recall if none strictly meet
        qualifying = [p for p in results if p.meets_criteria]
        if qualifying:
            # Pick ratio with lowest false alarms among qualifying
            optimal_point = min(qualifying, key=lambda x: x.false_alarms_per_hour)
            opt_ratio = optimal_point.cost_ratio
            summary = (
                f"Selected optimal cost ratio rho={opt_ratio:.1f}:1. "
                f"Achieves {optimal_point.event_recall*100:.1f}% recall and "
                f"{optimal_point.false_alarms_per_hour:.2f} false alarms/hr, satisfying project criteria."
            )
        else:
            optimal_point = max(results, key=lambda x: x.event_recall)
            opt_ratio = optimal_point.cost_ratio
            summary = (
                f"No ratio strictly satisfied both constraints; recommended rho={opt_ratio:.1f}:1 "
                f"for highest event recall ({optimal_point.event_recall*100:.1f}%)."
            )

        return SweepResult(points=results, optimal_ratio=opt_ratio, recommendation_summary=summary)
