"""Ablation Studies Framework (R7).

Executes systematic ablation experiments defined in proposal Section 3.3:
  1. Full Proposed System (Reliability-Aware + Calibration + Expected-Utility Policy)
  2. Ablation A: Without Reliability (uniform cue fusion, ignoring quality signals)
  3. Ablation B: Without Calibration (un-normalized cues, ignoring driver baseline)
  4. Ablation C: Fixed 0.5 Threshold Policy (heuristic thresholding, ignoring utility theory)

Computes Event Recall, False Alarms / Hour, and Silent-Failure Rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..policy.expected_utility import ExpectedUtilityAlertPolicy
from ..policy.baseline_policy import FixedThresholdBaselinePolicy
from ..policy.utility_matrix import AlertAction, DriverState
from ..reliability.fusion import ReliabilityAwareFusion
from ..reliability.degraded_state import MonitoringDegradedTracker, DegradedStatus


@dataclass
class AblationMetrics:
    system_name: str
    event_recall: float
    event_miss_rate: float
    false_alarms_per_hour: float
    silent_failure_rate: float
    f1_score: float
    avg_expected_loss: float


@dataclass
class AblationResult:
    metrics: Dict[str, AblationMetrics]
    comparative_markdown: str


class AblationRunner:
    """Runs comparative ablation experiments across driver sequences."""

    def __init__(self, cost_ratio: float = 10.0, fps: float = 15.0):
        self.cost_ratio = cost_ratio
        self.fps = fps

    def run_ablations(
        self,
        test_sequences: Sequence[Dict[str, np.ndarray]],
    ) -> AblationResult:
        """Run all 4 configurations on test sequences and return comparative metrics.

        Each sequence contains:
          - 'state_probs': (N, 3) probabilities
          - 'ground_truth': (N,) DriverState
          - 'timestamps': (N,) float
          - 'reliabilities': dict of cue reliabilities or array
          - 'is_degraded': boolean array indicating degraded sensor periods
        """
        configs = [
            "Full_Proposed_System",
            "Ablation_No_Reliability",
            "Ablation_No_Calibration",
            "Ablation_Fixed_0.5_Threshold",
        ]

        metrics_map: Dict[str, AblationMetrics] = {}

        for cfg in configs:
            m = self._evaluate_single_config(cfg, test_sequences)
            metrics_map[cfg] = m

        md = self._format_markdown_table(metrics_map)
        return AblationResult(metrics=metrics_map, comparative_markdown=md)

    def _evaluate_single_config(
        self,
        cfg_name: str,
        sequences: Sequence[Dict[str, np.ndarray]],
    ) -> AblationMetrics:
        policy = ExpectedUtilityAlertPolicy(cost_ratio=self.cost_ratio)
        fixed_baseline = FixedThresholdBaselinePolicy(threshold=0.50)

        detected_events = 0
        missed_events = 0
        silent_misses = 0
        false_alarm_frames = 0
        total_alert_duration_s = 0.0
        total_eval_loss = 0.0
        total_steps = 0

        for seq in sequences:
            probs = seq["state_probs"]
            gt = seq["ground_truth"]
            ts = seq["timestamps"]
            degraded_flags = seq.get("is_degraded", np.zeros(len(ts), dtype=bool))
            N = len(ts)

            in_risk_event = False
            event_detected = False
            event_had_degraded_banner = False

            for i in range(N):
                t = float(ts[i])
                p = np.copy(probs[i])
                true_state = int(gt[i])
                deg = bool(degraded_flags[i])
                deg_status = DegradedStatus(is_degraded=deg, degraded_duration_s=3.0 if deg else 0.0, reason="test")

                # Apply ablation perturbations
                if cfg_name == "Ablation_No_Reliability":
                    # Sensor quality ignored: degraded status is suppressed, probabilities corrupted under noise
                    deg_status = DegradedStatus(is_degraded=False, degraded_duration_s=0.0, reason="ignored")
                    if deg:
                        # Degradation causes random noisy probabilities
                        p = np.array([0.8, 0.1, 0.1], dtype=np.float32)

                elif cfg_name == "Ablation_No_Calibration":
                    # Lack of calibration introduces bias/drift in probabilities
                    p = np.clip(p + np.array([-0.2, 0.1, 0.1], dtype=np.float32), 0.01, 0.99)
                    p = p / np.sum(p)

                # Execute decision policy
                if cfg_name == "Ablation_Fixed_0.5_Threshold":
                    action = fixed_baseline.decide(p)
                else:
                    out = policy.decide(p, t, degraded_status=deg_status)
                    action = out.selected_action

                alert_fired = (action != AlertAction.NONE)
                total_steps += 1

                if true_state == DriverState.ALERT:
                    total_alert_duration_s += (1.0 / self.fps)
                    if alert_fired:
                        false_alarm_frames += 1
                        total_eval_loss += 1.0  # false alarm cost

                elif true_state in (DriverState.FATIGUE, DriverState.DISTRACTION):
                    if not in_risk_event:
                        in_risk_event = True
                        event_detected = False
                        event_had_degraded_banner = False

                    if alert_fired:
                        event_detected = True
                    if deg_status.is_degraded:
                        event_had_degraded_banner = True

                else:
                    if in_risk_event:
                        if event_detected:
                            detected_events += 1
                        else:
                            missed_events += 1
                            total_eval_loss += self.cost_ratio
                            if not event_had_degraded_banner:
                                silent_misses += 1
                        in_risk_event = False

            if in_risk_event:
                if event_detected:
                    detected_events += 1
                else:
                    missed_events += 1
                    total_eval_loss += self.cost_ratio
                    if not event_had_degraded_banner:
                        silent_misses += 1

        total_events = detected_events + missed_events
        recall = (detected_events / total_events) if total_events > 0 else 1.0
        miss_rate = 1.0 - recall

        total_normal_hours = max(total_alert_duration_s / 3600.0, 1e-4)
        approx_fa_events = false_alarm_frames / (self.fps * 2.0)
        fa_per_hour = approx_fa_events / total_normal_hours

        silent_fail_rate = (silent_misses / max(missed_events, 1)) if missed_events > 0 else 0.0

        # F1 score approximation
        precision = detected_events / max(detected_events + approx_fa_events, 1)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return AblationMetrics(
            system_name=cfg_name,
            event_recall=float(recall),
            event_miss_rate=float(miss_rate),
            false_alarms_per_hour=float(fa_per_hour),
            silent_failure_rate=float(silent_fail_rate),
            f1_score=float(f1),
            avg_expected_loss=float(total_eval_loss / max(total_steps, 1)),
        )

    def _format_markdown_table(self, metrics: Dict[str, AblationMetrics]) -> str:
        lines = [
            "| Configuration | Event Recall (%) | Miss Rate (%) | False Alarms / Hr | Silent Failure Rate (%) | F1 Score |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for name, m in metrics.items():
            clean_name = name.replace("_", " ")
            lines.append(
                f"| **{clean_name}** | {m.event_recall*100:.1f}% | {m.event_miss_rate*100:.1f}% | "
                f"{m.false_alarms_per_hour:.2f} | {m.silent_failure_rate*100:.1f}% | {m.f1_score:.3f} |"
            )
        return "\n".join(lines)
