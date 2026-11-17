"""Unit tests for R7 Quality Signals, Logistic Regression Reliability, and Degraded Tracker.
"""

import math
import numpy as np
import pytest

from R7_Jayesh_Contribution.reliability.quality_features import (
    QualityFeatureExtractor,
    QualitySignals,
)
from R7_Jayesh_Contribution.reliability.estimator import (
    CueReliabilityEstimator,
    sigmoid,
)
from R7_Jayesh_Contribution.reliability.fusion import ReliabilityAwareFusion
from R7_Jayesh_Contribution.reliability.degraded_state import MonitoringDegradedTracker


def test_quality_feature_extractor_jitter():
    ext = QualityFeatureExtractor(jitter_window_len=5)

    # Empty / none landmark
    assert math.isnan(ext.update_landmarks(None))

    # Static landmarks (zero jitter)
    pts = np.array([[100.0, 100.0], [200.0, 200.0]])
    for _ in range(5):
        j = ext.update_landmarks(pts)
    assert j == pytest.approx(0.0, abs=1e-5)

    # Jittering landmarks
    rng = np.random.RandomState(42)
    for _ in range(5):
        noisy_pts = pts + rng.normal(0, 3.0, pts.shape)
        j = ext.update_landmarks(noisy_pts)
    assert j > 0.5


def test_cue_reliability_estimator_defaults():
    estimator = CueReliabilityEstimator()

    # Good conditions: well-lit, good contrast, zero glare, sharp, centered pose
    good_signals = QualitySignals(
        timestamp_s=1.0,
        face_detected=True,
        eye_brightness=110.0,
        eye_contrast=45.0,
        eye_glare_frac=0.0,
        eye_sharpness=120.0,
        head_angle_z=0.2,
        landmark_jitter=0.5,
        detector_confidence=0.99,
    )

    r_good = estimator.predict_all(good_signals)
    assert 0.0 <= r_good["eye"] <= 1.0
    assert 0.0 <= r_good["mouth"] <= 1.0
    assert 0.0 <= r_good["head_pose"] <= 1.0
    assert r_good["eye"] > 0.70  # Eye cue should have high reliability under good conditions

    # Severe degradation: high glare and blur
    bad_signals = QualitySignals(
        timestamp_s=2.0,
        face_detected=True,
        eye_brightness=240.0,
        eye_contrast=8.0,
        eye_glare_frac=0.85,  # strong glare blob
        eye_sharpness=10.0,   # severe blur
        head_angle_z=3.5,     # severe head turn
        landmark_jitter=8.0,
        detector_confidence=0.50,
    )

    r_bad = estimator.predict_all(bad_signals)
    assert r_bad["eye"] < r_good["eye"]
    assert r_bad["eye"] < 0.20  # Glare & blur drop reliability sharply
    assert r_bad["overall"] < r_good["overall"]


def test_quality_feature_extractor_r4_scaling():
    """Verify that inputs normalized to [0, 1] by R4 (divided by 255) are correctly scaled."""
    ext = QualityFeatureExtractor()
    estimator = CueReliabilityEstimator(feature_extractor=ext)

    # R4 contract: brightness in [0, 1], contrast in [0, 0.5]
    r4_signals = QualitySignals(
        timestamp_s=1.0,
        face_detected=True,
        eye_brightness=110.0 / 255.0,  # ~0.431 from R4
        eye_contrast=45.0 / 255.0,     # ~0.176 from R4
        eye_glare_frac=0.0,
        eye_sharpness=120.0,
        head_angle_z=0.2,
        landmark_jitter=0.5,
        detector_confidence=0.99,
    )

    vec = ext.extract_eye_quality_vector(r4_signals)
    # Feature 0: brightness deviation should be ~0.0
    assert vec[0] == pytest.approx(0.0, abs=1e-3)
    # Feature 1: contrast normalized by 50 should be 45/50 = 0.90
    assert vec[1] == pytest.approx(0.90, abs=1e-3)

    # Predicted eye reliability should be high (> 0.70) under nominal R4 conditions
    res = estimator.predict_all(r4_signals)
    assert res["eye"] > 0.70


def test_cue_reliability_estimator_fit():
    estimator = CueReliabilityEstimator()
    rng = np.random.RandomState(123)

    # Generate synthetic training samples for eye cue: 7 features
    # High glare (idx 2) correlates with label 0 (unreliable)
    N = 200
    X = rng.uniform(0.0, 1.0, (N, 7)).astype(np.float32)
    y = np.where(X[:, 2] > 0.5, 0, 1)

    estimator.fit_cue("eye", X, y)
    assert estimator.models["eye"] is not None

    # High glare sample should yield low reliability
    test_bad = np.array([0.1, 0.9, 0.9, 0.8, 0.1, 0.1, 1.0], dtype=np.float32)
    prob_bad = estimator.predict_cue_reliability("eye", test_bad)
    assert prob_bad < 0.5


def test_reliability_aware_fusion():
    fusion = ReliabilityAwareFusion()

    # Case 1: Eye is healthy, mouth is healthy
    percept1 = fusion.fuse(
        p_closed=0.9,
        mar_norm=1.0,
        head_angle_z=0.2,
        reliabilities={"eye": 0.9, "mouth": 0.85, "head_pose": 0.8},
    )
    assert percept1.fused_fatigue_score > 0.4
    assert percept1.cue_weights["eye"] > percept1.cue_weights["head_pose"]

    # Case 2: Glare makes eye completely unreliable (r_eye = 0.05), but driver yawns (mar_norm = 2.4)
    percept2 = fusion.fuse(
        p_closed=0.0,  # misleading eye closure cue under glare
        mar_norm=2.4,  # evident yawn
        head_angle_z=0.1,
        reliabilities={"eye": 0.05, "mouth": 0.90, "head_pose": 0.8},
    )
    # The unreliable eye cue should be down-weighted, allowing mouth yawn to dominate
    assert percept2.cue_weights["eye"] == 0.0
    assert percept2.cue_weights["mouth"] > 0.70
    assert percept2.fused_fatigue_score > 0.5


def test_monitoring_degraded_tracker():
    tracker = MonitoringDegradedTracker(
        threshold=0.30,
        required_duration_s=3.0,
        recovery_duration_s=1.0,
    )

    # Healthy frames
    st = tracker.update(0.80, timestamp_s=0.0)
    assert not st.is_degraded

    # Drops to low reliability at 1.0 s; remains low for 1.5 s at t=2.5s (< 3.0s required)
    st = tracker.update(0.15, timestamp_s=1.0)
    assert not st.is_degraded
    st = tracker.update(0.15, timestamp_s=2.5)
    assert not st.is_degraded

    # Remains low at 4.2 s (3.2s elapsed since 1.0s >= 3.0s required) -> Degraded banner fires
    st = tracker.update(0.10, timestamp_s=4.2)
    assert st.is_degraded
    assert "Reliability below" in st.reason

    # Healthy reading at 4.5 s (needs 1.0s continuous recovery)
    st = tracker.update(0.85, timestamp_s=4.5)
    assert st.is_degraded  # still within recovery buffer

    # Reaches 5.6 s (1.1s > 1.0s recovery) -> Cleared
    st = tracker.update(0.90, timestamp_s=5.6)
    assert not st.is_degraded
