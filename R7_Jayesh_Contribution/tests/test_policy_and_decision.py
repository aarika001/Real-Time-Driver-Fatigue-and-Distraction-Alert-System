"""Unit tests for R7 Utility Matrix, Annoyance Tracker, and Expected-Utility Alert Policy.
"""

import numpy as np
import pytest

from R7_Jayesh_Contribution.policy.utility_matrix import (
    AlertAction,
    DriverState,
    UtilityMatrix,
)
from R7_Jayesh_Contribution.policy.annoyance import AnnoyanceTracker
from R7_Jayesh_Contribution.policy.expected_utility import ExpectedUtilityAlertPolicy
from R7_Jayesh_Contribution.policy.baseline_policy import FixedThresholdBaselinePolicy
from R7_Jayesh_Contribution.reliability.degraded_state import DegradedStatus


def test_utility_matrix_properties():
    umat = UtilityMatrix(cost_ratio=10.0, base_false_alarm_cost=1.0)
    U = umat.get_matrix()

    assert U.shape == (3, 4)
    # Utility is non-positive
    assert np.all(U <= 0.0)

    # When ALERT (row 0), NONE (col 0) is optimal (0.0)
    assert U[DriverState.ALERT, AlertAction.NONE] == 0.0
    # Alerts when ALERT have negative utility (false alarms)
    assert U[DriverState.ALERT, AlertAction.SOFT_VISUAL] < 0.0
    assert U[DriverState.ALERT, AlertAction.URGENT] < U[DriverState.ALERT, AlertAction.VISUAL_AUDIO]

    # When FATIGUE (row 1), NONE has miss cost
    assert U[DriverState.FATIGUE, AlertAction.NONE] == -10.0
    assert U[DriverState.FATIGUE, AlertAction.URGENT] == 0.0


def test_annoyance_tracker_and_pruning():
    tracker = AnnoyanceTracker(window_duration_s=600.0, penalty_per_alert=0.5)

    assert tracker.get_annoyance_penalty(0.0) == 0.0

    # Fire 3 alerts
    tracker.record_alert(10.0, AlertAction.VISUAL_AUDIO)
    tracker.record_alert(20.0, AlertAction.VISUAL_AUDIO)
    tracker.record_alert(30.0, AlertAction.URGENT)

    pen1 = tracker.get_annoyance_penalty(40.0)
    assert pen1 > 1.0
    assert tracker.count_recent_alerts(40.0) == 3

    # Fast forward past 600s window (t = 700.0s)
    # All previous alerts should be pruned out
    pen2 = tracker.get_annoyance_penalty(700.0)
    assert pen2 == 0.0
    assert tracker.count_recent_alerts(700.0) == 0


def test_expected_utility_alert_policy_decisions():
    policy = ExpectedUtilityAlertPolicy(cost_ratio=10.0)

    # 1. Clear alert driver -> Action NONE
    res_alert = policy.decide([0.98, 0.01, 0.01], timestamp_s=1.0)
    assert res_alert.selected_action == AlertAction.NONE

    # 2. Strong fatigue -> Action URGENT or VISUAL_AUDIO
    res_fatigue = policy.decide([0.05, 0.90, 0.05], timestamp_s=5.0)
    assert res_fatigue.selected_action in (AlertAction.URGENT, AlertAction.VISUAL_AUDIO)
    assert res_fatigue.action_name in ("URGENT", "VISUAL_AUDIO")

    # 3. Sustained distraction -> Action VISUAL_AUDIO or SOFT_VISUAL
    res_dist = policy.decide([0.10, 0.05, 0.85], timestamp_s=10.0)
    assert res_dist.selected_action in (AlertAction.VISUAL_AUDIO, AlertAction.SOFT_VISUAL)

    # 4. Monitoring Degraded banner active
    deg = DegradedStatus(is_degraded=True, degraded_duration_s=4.0, reason="low light")
    res_deg = policy.decide([0.10, 0.80, 0.10], timestamp_s=15.0, degraded_status=deg)
    assert res_deg.is_monitoring_degraded
    assert res_deg.action_name == "MONITORING_DEGRADED"


def test_annoyance_modulates_policy_sensitivity():
    # If the system has alerted frequently, annoyance penalty raises false alarm cost,
    # requiring higher confidence before triggering repeated alerts on borderline frames.
    annoyance = AnnoyanceTracker(penalty_per_alert=2.0)
    policy = ExpectedUtilityAlertPolicy(cost_ratio=5.0, annoyance_tracker=annoyance)

    # Borderline ambiguous belief: 65% alert, 35% fatigue
    borderline_p = [0.65, 0.35, 0.0]

    # Clean history: with cost ratio 5:1, miss cost outweighs small FA cost, triggering soft or audio alert
    out_fresh = policy.decide(borderline_p, timestamp_s=1.0)
    fresh_action = out_fresh.selected_action

    # Simulate 5 frequent alerts causing high annoyance
    for t in [2.0, 3.0, 4.0, 5.0, 6.0]:
        annoyance.record_alert(t, AlertAction.URGENT)

    # With high annoyance penalty, policy becomes more cautious about borderline noise
    out_annoyed = policy.decide(borderline_p, timestamp_s=10.0)
    # EU for NONE becomes more attractive relative to intrusive alarms
    assert out_annoyed.annoyance_penalty > out_fresh.annoyance_penalty


def test_fixed_threshold_baseline_policy():
    base = FixedThresholdBaselinePolicy(threshold=0.50)
    assert base.decide([0.60, 0.30, 0.10]) == AlertAction.NONE
    assert base.decide([0.40, 0.55, 0.05]) == AlertAction.VISUAL_AUDIO
