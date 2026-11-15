"""
tests/test_head_pose_core.py
=============================
R5 unit tests for head_pose_core.py

Run from the head-pose/ directory:
    python -m pytest tests/ -v

Tests
-----
  1. test_identity_pose            - Synthetic front-facing landmarks → near-zero angles
  2. test_deviation_zero_at_baseline - deviation = 0 when raw == baseline mean
  3. test_deviation_known_values   - deviation arithmetic is correct
  4. test_head_angle_from_neutral  - Euclidean magnitude calculation
  5. test_feature_vector_shape     - build_head_feature_vector returns (6,) float32
  6. test_feature_vector_values    - first 3 elements are raw, last 3 are delta
  7. test_apply_baseline_shape     - apply_baseline_to_sequence returns (N, 6)
  8. test_apply_baseline_undetected_zeros - undetected frames get Δ = 0
  9. test_compute_head_baseline_basic - correct mean/std over calibration window
 10. test_compute_head_baseline_fallback - triggers fallback when too few clean frames
 11. test_std_floor                - std is never below _STD_FLOOR_DEG (1°)
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Make sure the parent head-pose/ directory is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))
from head_pose_core import (
    MIN_CALIBRATION_FRAMES,
    _STD_FLOOR_DEG,
    apply_baseline_to_sequence,
    build_head_feature_vector,
    compute_head_baseline,
    compute_head_deviation,
    extract_head_pose,
    head_angle_from_neutral,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_baseline(
    mean_pitch=0.0, mean_yaw=0.0, mean_roll=0.0,
    std_pitch=5.0,  std_yaw=5.0,  std_roll=5.0,
) -> dict:
    return dict(
        mean_pitch=mean_pitch, mean_yaw=mean_yaw,  mean_roll=mean_roll,
        std_pitch=std_pitch,   std_yaw=std_yaw,    std_roll=std_roll,
        n_clean_frames=450,    fallback_used=False,
    )


def _make_synthetic_landmarks(
    frame_w: int = 640,
    frame_h: int = 480,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> list:
    """Create a list of 478 dummy landmarks that approximate a front-facing pose.

    We use the canonical 6-point 3-D model projected back to 2-D to produce
    near-zero-rotation landmarks without needing a real video frame.
    """
    import cv2
    from head_pose_core import FACE_3D_MODEL, POSE_2D_IDXS

    # Build a rotation matrix for the requested angles
    yaw_r = np.deg2rad(yaw_deg)
    pitch_r = np.deg2rad(pitch_deg)
    Ry = np.array([[np.cos(yaw_r),  0, np.sin(yaw_r)],
                   [0,              1, 0             ],
                   [-np.sin(yaw_r), 0, np.cos(yaw_r)]], dtype=np.float64)
    Rx = np.array([[1, 0,               0              ],
                   [0, np.cos(pitch_r), -np.sin(pitch_r)],
                   [0, np.sin(pitch_r),  np.cos(pitch_r)]], dtype=np.float64)
    R = Rx @ Ry

    focal = float(frame_w)
    cam = np.array([[focal, 0, frame_w / 2],
                    [0, focal, frame_h / 2],
                    [0, 0,     1          ]], dtype=np.float64)

    pts_3d = (R @ FACE_3D_MODEL.T).T
    pts_2d, _ = cv2.projectPoints(pts_3d, np.zeros(3), np.zeros(3), cam, None)
    pts_2d = pts_2d.reshape(-1, 2)

    # Build 478 landmarks; only the 6 at POSE_2D_IDXS matter for solvePnP.
    lms = []
    for i in range(478):
        lms.append(SimpleNamespace(x=0.5, y=0.5))
    for enum_i, lm_idx in enumerate(POSE_2D_IDXS):
        x_norm = pts_2d[enum_i, 0] / frame_w
        y_norm = pts_2d[enum_i, 1] / frame_h
        lms[lm_idx] = SimpleNamespace(x=float(x_norm), y=float(y_norm))

    return lms


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractHeadPose:
    def test_identity_pose_near_zero(self):
        """Front-facing synthetic landmarks produce angles close to 0°."""
        lms = _make_synthetic_landmarks(pitch_deg=0.0, yaw_deg=0.0)
        pitch, yaw, roll = extract_head_pose(lms, 640, 480)
        # Tolerance is generous because solvePnP with synthetic data can have
        # small numerical errors (10° tolerance).
        assert abs(pitch) < 10.0, f"pitch={pitch}"
        assert abs(yaw) < 10.0, f"yaw={yaw}"

    def test_returns_floats(self):
        lms = _make_synthetic_landmarks()
        result = extract_head_pose(lms, 320, 240)
        assert all(isinstance(v, float) for v in result)

    def test_returns_tuple_of_three(self):
        lms = _make_synthetic_landmarks()
        result = extract_head_pose(lms, 640, 480)
        assert len(result) == 3


class TestComputeHeadDeviation:
    def test_zero_deviation_at_baseline_mean(self):
        """When raw == baseline mean, all deviations are 0."""
        bl = _make_baseline(mean_pitch=5.0, mean_yaw=-3.0, mean_roll=1.0)
        dp, dy, dr = compute_head_deviation(5.0, -3.0, 1.0, bl)
        assert dp == pytest.approx(0.0, abs=1e-6)
        assert dy == pytest.approx(0.0, abs=1e-6)
        assert dr == pytest.approx(0.0, abs=1e-6)

    def test_known_deviation_values(self):
        """Z-score arithmetic: (raw - mean) / std."""
        bl = _make_baseline(
            mean_pitch=0.0, mean_yaw=0.0, mean_roll=0.0,
            std_pitch=10.0, std_yaw=5.0,  std_roll=2.0,
        )
        dp, dy, dr = compute_head_deviation(10.0, 5.0, 2.0, bl)
        assert dp == pytest.approx(1.0, rel=1e-5)
        assert dy == pytest.approx(1.0, rel=1e-5)
        assert dr == pytest.approx(1.0, rel=1e-5)

    def test_negative_deviation(self):
        bl = _make_baseline(mean_yaw=10.0, std_yaw=5.0)
        _, dy, _ = compute_head_deviation(0.0, 0.0, 0.0, bl)
        assert dy == pytest.approx(-2.0, rel=1e-5)


class TestHeadAngleFromNeutral:
    def test_zero(self):
        assert head_angle_from_neutral(0.0, 0.0) == pytest.approx(0.0)

    def test_pythagorean(self):
        # 3-4-5 triangle in z-score space
        result = head_angle_from_neutral(3.0, 4.0)
        assert result == pytest.approx(5.0, rel=1e-5)

    def test_non_negative(self):
        assert head_angle_from_neutral(-2.0, -3.0) >= 0.0


class TestBuildHeadFeatureVector:
    def test_shape_and_dtype(self):
        bl = _make_baseline()
        vec = build_head_feature_vector(5.0, -3.0, 1.0, bl)
        assert vec.shape == (6,)
        assert vec.dtype == np.float32

    def test_first_three_are_raw(self):
        bl = _make_baseline()
        vec = build_head_feature_vector(7.0, -2.0, 3.0, bl)
        assert vec[0] == pytest.approx(7.0, abs=1e-4)
        assert vec[1] == pytest.approx(-2.0, abs=1e-4)
        assert vec[2] == pytest.approx(3.0, abs=1e-4)

    def test_last_three_are_delta(self):
        bl = _make_baseline(
            mean_pitch=0.0, mean_yaw=0.0, mean_roll=0.0,
            std_pitch=10.0, std_yaw=10.0, std_roll=10.0,
        )
        vec = build_head_feature_vector(10.0, -10.0, 20.0, bl)
        assert vec[3] == pytest.approx(1.0,  abs=1e-4)   # Δpitch
        assert vec[4] == pytest.approx(-1.0, abs=1e-4)   # Δyaw
        assert vec[5] == pytest.approx(2.0,  abs=1e-4)   # Δroll


class TestApplyBaselineToSequence:
    def test_output_shape(self):
        N = 50
        head_pose = np.random.randn(N, 3).astype(np.float32)
        face_det  = np.ones(N, dtype=np.uint8)
        bl = _make_baseline()
        out = apply_baseline_to_sequence(head_pose, face_det, bl)
        assert out.shape == (N, 6)
        assert out.dtype == np.float32

    def test_undetected_frames_get_zero_delta(self):
        """Frames where face_detected==0 must have Δ columns = 0.0."""
        N = 10
        head_pose = np.ones((N, 3), dtype=np.float32) * 15.0
        face_det  = np.zeros(N, dtype=np.uint8)   # all lost
        bl = _make_baseline(mean_pitch=0.0, std_pitch=5.0)
        out = apply_baseline_to_sequence(head_pose, face_det, bl)
        assert np.all(out[:, 3:] == 0.0), "delta columns should be 0 for undetected frames"

    def test_detected_frames_have_nonzero_delta(self):
        N = 5
        head_pose = np.ones((N, 3), dtype=np.float32) * 20.0
        face_det  = np.ones(N, dtype=np.uint8)
        bl = _make_baseline(mean_pitch=0.0, std_pitch=10.0)
        out = apply_baseline_to_sequence(head_pose, face_det, bl)
        # Δpitch should be 20/10 = 2.0
        assert out[0, 3] == pytest.approx(2.0, abs=1e-4)


class TestComputeHeadBaseline:
    def _make_pose_array(self, N=500) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        head_pose = rng.normal(
            loc=[2.0, -1.0, 0.5], scale=[4.0, 3.0, 2.0], size=(N, 3)
        ).astype(np.float32)
        face_det = np.ones(N, dtype=np.uint8)
        return head_pose, face_det

    def test_basic_mean_correctness(self):
        head_pose, face_det = self._make_pose_array(600)
        baseline = compute_head_baseline(head_pose, face_det, calibration_frames=450)
        # Mean should be close to [2, -1, 0.5] from the first 450 frames
        assert abs(baseline["mean_pitch"] - head_pose[:450, 0].mean()) < 0.5
        assert abs(baseline["mean_yaw"]   - head_pose[:450, 1].mean()) < 0.5
        assert not baseline["fallback_used"]

    def test_fallback_triggered_on_few_clean_frames(self):
        N = 500
        head_pose = np.zeros((N, 3), dtype=np.float32)
        # Only 10 detected frames — below MIN_CALIBRATION_FRAMES (150)
        face_det = np.zeros(N, dtype=np.uint8)
        face_det[:10] = 1
        baseline = compute_head_baseline(head_pose, face_det, calibration_frames=450)
        assert baseline["fallback_used"]

    def test_std_floor_applied(self):
        """When a driver holds perfectly still, std is floored at _STD_FLOOR_DEG."""
        N = 500
        head_pose = np.full((N, 3), 5.0, dtype=np.float32)  # zero variance
        face_det  = np.ones(N, dtype=np.uint8)
        baseline = compute_head_baseline(head_pose, face_det, calibration_frames=450)
        assert baseline["std_pitch"] >= _STD_FLOOR_DEG
        assert baseline["std_yaw"]   >= _STD_FLOOR_DEG
        assert baseline["std_roll"]  >= _STD_FLOOR_DEG

    def test_uses_only_calibration_window(self):
        """Only the first calibration_frames rows should be considered."""
        N = 600
        head_pose = np.zeros((N, 3), dtype=np.float32)
        head_pose[450:] = 999.0   # wild values after the window
        face_det = np.ones(N, dtype=np.uint8)
        baseline = compute_head_baseline(head_pose, face_det, calibration_frames=450)
        assert abs(baseline["mean_pitch"]) < 1.0, (
            "Baseline should only use the first 450 frames, not the 999-filled tail"
        )
