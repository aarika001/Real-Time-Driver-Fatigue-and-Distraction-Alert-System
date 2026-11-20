"""Unit tests for R7 Cost Sweep and Ablation Experiments.
"""

import numpy as np
import pytest

from R7_Jayesh_Contribution.evaluation.cost_sweep import CostRatioSweep
from R7_Jayesh_Contribution.evaluation.ablations import AblationRunner
from R7_Jayesh_Contribution.policy.utility_matrix import DriverState


def generate_synthetic_driver_sequence(n_frames: int = 300, fps: float = 15.0):
    """Generate a realistic test driving sequence with alert driving, fatigue, and distraction episodes."""
    ts = np.arange(n_frames, dtype=np.float32) / fps
    gt = np.zeros(n_frames, dtype=np.int32)  # default ALERT

    # Frames 50-90: Fatigue episode
    gt[50:90] = DriverState.FATIGUE
    # Frames 180-210: Distraction episode
    gt[180:210] = DriverState.DISTRACTION

    probs = np.zeros((n_frames, 3), dtype=np.float32)
    degraded = np.zeros(n_frames, dtype=bool)

    for i in range(n_frames):
        if gt[i] == DriverState.ALERT:
            # High alert probability with occasional minor noise
            probs[i] = [0.92, 0.05, 0.03]
        elif gt[i] == DriverState.FATIGUE:
            probs[i] = [0.08, 0.85, 0.07]
        elif gt[i] == DriverState.DISTRACTION:
            probs[i] = [0.10, 0.05, 0.85]

    # Add a degraded period during frames 120-140 (e.g. lens glare / tunnel)
    degraded[120:140] = True

    return {
        "state_probs": probs,
        "ground_truth": gt,
        "timestamps": ts,
        "is_degraded": degraded,
    }


def test_cost_ratio_sweep_execution():
    seq = generate_synthetic_driver_sequence(n_frames=300)
    sweep = CostRatioSweep(ratios=[3.0, 5.0, 10.0, 15.0, 20.0])

    result = sweep.run_sweep([seq])

    assert len(result.points) == 5
    assert result.optimal_ratio in [3.0, 5.0, 10.0, 15.0, 20.0]
    # Recall should be high across the episodes
    assert result.points[-1].event_recall >= 0.80
    assert len(result.recommendation_summary) > 0


def test_ablation_runner_execution():
    seq = generate_synthetic_driver_sequence(n_frames=300)
    runner = AblationRunner(cost_ratio=10.0)

    result = runner.run_ablations([seq])

    assert "Full_Proposed_System" in result.metrics
    assert "Ablation_No_Reliability" in result.metrics
    assert "Ablation_No_Calibration" in result.metrics
    assert "Ablation_Fixed_0.5_Threshold" in result.metrics

    full_m = result.metrics["Full_Proposed_System"]
    assert full_m.event_recall >= 0.80

    # Comparative markdown should contain table headers
    assert "| Configuration |" in result.comparative_markdown
    assert "Full Proposed System" in result.comparative_markdown
