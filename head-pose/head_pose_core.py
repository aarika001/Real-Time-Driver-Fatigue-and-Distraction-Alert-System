"""
head_pose_core.py
=================
R5 — Head Pose & Off-Axis Tests (Aaditya)
Team NeuroNova · Real-Time Driver Fatigue and Distraction Alert System

Canonical, importable functions for:
  1. Extracting pitch/yaw/roll from MediaPipe face landmarks (solvePnP).
  2. Computing z-scored head-deviation features relative to a personal baseline.
  3. Building the 6-element feature vector consumed by R6's GRU.
  4. Computing a scalar head-angle-from-neutral signal consumed by R7's
     reliability estimator.

Design constraints
------------------
- extract_head_pose() must run in <2 ms on a laptop CPU (R8 real-time budget).
- All functions are stateless and free of side effects.
- The solvePnP geometry is intentionally identical to R3's preprocess-rldd.py
  so that offline and live outputs are bit-compatible. The parity test in
  tests/test_head_pose_core.py enforces this.

Interface contracts
-------------------
  R6 (GRU input)   : build_head_feature_vector(pitch, yaw, roll, baseline) → (6,)
  R7 (reliability) : head_angle_from_neutral(Δpitch, Δyaw) → float
  R8 (live loop)   : extract_head_pose(landmarks, w, h) → (pitch, yaw, roll)
"""

from __future__ import annotations

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Constants — 6-point 3-D head model (metres × 10⁻³, OpenCV convention)
# Identical to R3's preprocess-rldd.py so offline and live produce the same
# rotation vectors.
# ---------------------------------------------------------------------------

FACE_3D_MODEL: np.ndarray = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip        — landmark 1
        (0.0, -330.0, -65.0),     # Chin            — landmark 152
        (-225.0, 170.0, -135.0),  # Left eye corner — landmark 33
        (225.0, 170.0, -135.0),   # Right eye corner— landmark 263
        (-150.0, -150.0, -125.0), # Left mouth      — landmark 61
        (150.0, -150.0, -125.0),  # Right mouth     — landmark 291
    ],
    dtype=np.float64,
)

POSE_2D_IDXS: list[int] = [1, 152, 33, 263, 61, 291]

# Minimum std floor so z-scores never explode when a driver holds perfectly
# still during calibration.
_STD_FLOOR_DEG: float = 1.0

# Minimum frames required in the calibration window before we trust the
# personal baseline (otherwise the population fallback is used).
MIN_CALIBRATION_FRAMES: int = 150


# ---------------------------------------------------------------------------
# 1. Pose extraction (stateless, <2 ms per call)
# ---------------------------------------------------------------------------

def extract_head_pose(
    landmarks,          # mediapipe FaceLandmarker landmark list (normalized)
    frame_w: int,
    frame_h: int,
) -> tuple[float, float, float]:
    """Return (pitch, yaw, roll) in degrees from MediaPipe face landmarks.

    Parameters
    ----------
    landmarks:
        The first element of ``result.face_landmarks`` returned by a
        MediaPipe FaceLandmarker call.  Each element has ``.x``, ``.y``
        attributes normalized to [0, 1].
    frame_w, frame_h:
        Frame width and height in pixels.

    Returns
    -------
    pitch, yaw, roll : float
        Head angles in degrees.  Returns (0.0, 0.0, 0.0) if solvePnP fails.
    """
    pts_2d = np.array(
        [[landmarks[i].x * frame_w, landmarks[i].y * frame_h] for i in POSE_2D_IDXS],
        dtype=np.float64,
    )

    focal_length = float(frame_w)
    cam_matrix = np.array(
        [
            [focal_length, 0.0,          frame_w / 2.0],
            [0.0,          focal_length, frame_h / 2.0],
            [0.0,          0.0,          1.0           ],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, _ = cv2.solvePnP(
        FACE_3D_MODEL, pts_2d, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return float(angles[0]), float(angles[1]), float(angles[2])


# ---------------------------------------------------------------------------
# 2. Calibration baseline helpers
# ---------------------------------------------------------------------------

def compute_head_baseline(
    head_pose_array: np.ndarray,
    face_detected_array: np.ndarray,
    calibration_frames: int = 450,
) -> dict:
    """Compute mean and std head-pose angles from the calibration window.

    Parameters
    ----------
    head_pose_array : np.ndarray, shape (N, 3)
        Per-frame ``[pitch, yaw, roll]`` as stored in a processed .npz file
        by R3's preprocess-rldd.py.
    face_detected_array : np.ndarray, shape (N,)
        Binary detection flag (1 = face visible, 0 = lost); also from R3.
    calibration_frames : int
        Number of leading frames to use (default 450 = 30 s × 15 FPS).

    Returns
    -------
    dict with keys:
        mean_pitch, mean_yaw, mean_roll   : float
        std_pitch,  std_yaw,  std_roll    : float  (floored at _STD_FLOOR_DEG)
        n_clean_frames                    : int
        fallback_used                     : bool
    """
    n = min(calibration_frames, len(head_pose_array))
    pose_window = head_pose_array[:n]            # (n, 3)
    det_window  = face_detected_array[:n].astype(bool)

    clean = pose_window[det_window]              # only detected frames

    fallback_used = len(clean) < MIN_CALIBRATION_FRAMES

    if fallback_used or len(clean) == 0:
        # Population-level defaults for a front-facing camera.
        # These will be overwritten by calibrate_head_baseline.py with the
        # actual across-subject mean once baselines are computed for all
        # subjects.
        mean_pitch, mean_yaw, mean_roll = 0.0, 0.0, 0.0
        std_pitch  = std_yaw = std_roll = _STD_FLOOR_DEG
    else:
        mean_pitch, mean_yaw, mean_roll = clean.mean(axis=0).tolist()
        std_pitch,  std_yaw,  std_roll  = clean.std(axis=0).tolist()

    return {
        "mean_pitch":    float(mean_pitch),
        "mean_yaw":      float(mean_yaw),
        "mean_roll":     float(mean_roll),
        "std_pitch":     float(max(std_pitch, _STD_FLOOR_DEG)),
        "std_yaw":       float(max(std_yaw,   _STD_FLOOR_DEG)),
        "std_roll":      float(max(std_roll,  _STD_FLOOR_DEG)),
        "n_clean_frames": int(len(clean)),
        "fallback_used":  bool(fallback_used),
    }


# ---------------------------------------------------------------------------
# 3. Deviation features (z-scored relative to personal baseline)
# ---------------------------------------------------------------------------

def compute_head_deviation(
    pitch: float,
    yaw:   float,
    roll:  float,
    baseline: dict,
) -> tuple[float, float, float]:
    """Z-score raw head angles relative to the driver's calibrated neutral.

    Parameters
    ----------
    pitch, yaw, roll : float
        Raw head angles in degrees for the current frame.
    baseline : dict
        Output of ``compute_head_baseline`` (or a row loaded from
        ``head_baselines.csv``).

    Returns
    -------
    delta_pitch, delta_yaw, delta_roll : float
        (raw - mean) / std for each axis.
        A value of ±1.0 means one personal standard-deviation from neutral.
    """
    delta_pitch = (pitch - baseline["mean_pitch"]) / baseline["std_pitch"]
    delta_yaw   = (yaw   - baseline["mean_yaw"])   / baseline["std_yaw"]
    delta_roll  = (roll  - baseline["mean_roll"])  / baseline["std_roll"]
    return float(delta_pitch), float(delta_yaw), float(delta_roll)


def head_angle_from_neutral(delta_pitch: float, delta_yaw: float) -> float:
    """Euclidean magnitude of (Δpitch, Δyaw) in z-score units.

    This is the scalar fed to R7's reliability estimator.  Large values
    indicate the driver is looking far from their calibrated neutral, which
    degrades eye-landmark quality and may indicate distraction.

    Returns
    -------
    float ≥ 0.0
    """
    return float(np.sqrt(delta_pitch ** 2 + delta_yaw ** 2))


# ---------------------------------------------------------------------------
# 4. GRU feature vector (R6 interface)
# ---------------------------------------------------------------------------

def build_head_feature_vector(
    pitch: float,
    yaw:   float,
    roll:  float,
    baseline: dict,
) -> np.ndarray:
    """Build the 6-element head-pose feature vector for R6's GRU.

    Layout:  [pitch, yaw, roll,  Δpitch, Δyaw, Δroll]
              ↑ raw (°)          ↑ z-scored relative to personal baseline

    Parameters
    ----------
    pitch, yaw, roll : float
        Raw angles in degrees.
    baseline : dict
        Output of ``compute_head_baseline``.

    Returns
    -------
    np.ndarray, shape (6,), dtype float32
    """
    delta_pitch, delta_yaw, delta_roll = compute_head_deviation(
        pitch, yaw, roll, baseline
    )
    return np.array(
        [pitch, yaw, roll, delta_pitch, delta_yaw, delta_roll],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# 5. Batch helper: apply deviation features to a whole .npz sequence
# ---------------------------------------------------------------------------

def apply_baseline_to_sequence(
    head_pose_array: np.ndarray,
    face_detected_array: np.ndarray,
    baseline: dict,
) -> np.ndarray:
    """Vectorised version of build_head_feature_vector over an entire sequence.

    Parameters
    ----------
    head_pose_array : np.ndarray, shape (N, 3)
        Raw [pitch, yaw, roll] per frame.
    face_detected_array : np.ndarray, shape (N,)
        Binary detection flag; undetected frames stay at their zero-padded
        raw values and get deviation = 0.0.
    baseline : dict
        Personal baseline from ``compute_head_baseline``.

    Returns
    -------
    np.ndarray, shape (N, 6), dtype float32
        Columns: [pitch, yaw, roll, Δpitch, Δyaw, Δroll]
    """
    N = len(head_pose_array)
    out = np.zeros((N, 6), dtype=np.float32)
    out[:, :3] = head_pose_array

    # Only z-score frames where the face was detected; undetected frames
    # keep Δ = 0.0 (neutral by convention, they carry no information).
    detected = face_detected_array.astype(bool)
    if detected.any():
        raw_det = head_pose_array[detected]          # (M, 3)
        delta = (raw_det - np.array([
            baseline["mean_pitch"],
            baseline["mean_yaw"],
            baseline["mean_roll"],
        ])) / np.array([
            baseline["std_pitch"],
            baseline["std_yaw"],
            baseline["std_roll"],
        ])
        out[detected, 3:] = delta.astype(np.float32)

    return out
