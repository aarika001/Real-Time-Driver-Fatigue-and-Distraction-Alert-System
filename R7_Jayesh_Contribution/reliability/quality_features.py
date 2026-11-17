"""Quality feature extraction for cue reliability estimation (R7).

Aggregates raw quality signals from R4 (eye crops & mouth) and R5 (head pose),
computes landmark jitter over a sliding temporal window, and formats feature
vectors for logistic regression cue reliability estimation.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class QualitySignals:
    """Per-frame raw quality metrics collected from perception modules."""
    timestamp_s: float
    face_detected: bool
    # Eye signals from R4 (cues/eye_crop.py) and R3 (preprocess-rldd.py)
    # Brightness in [0, 1] (mean / 255.0), contrast in [0, 0.5] (std / 255.0)
    eye_brightness: float = float("nan")
    eye_contrast: float = float("nan")
    eye_glare_frac: float = float("nan")  # in [0, 1]
    eye_sharpness: float = float("nan")   # Laplacian variance
    # Head pose signal from R5 (head-pose/head_pose_core.py)
    head_angle_z: float = float("nan")  # from head_angle_from_neutral(delta_pitch, delta_yaw)
    # Landmark stability
    landmark_jitter: float = float("nan")
    detector_confidence: float = 1.0


class QualityFeatureExtractor:
    """Maintains short temporal history to compute jitter and normalized quality features."""

    def __init__(self, jitter_window_len: int = 15, fps: float = 15.0):
        """
        Parameters
        ----------
        jitter_window_len:
            Number of recent frames used to compute landmark coordinate variance (jitter).
        fps:
            Expected frame rate (default: 15 FPS per project spec).
        """
        self.jitter_window_len = jitter_window_len
        self.fps = fps
        # Circular buffer storing key landmark centroids (e.g. nose tip / eye corners)
        self._landmark_history: collections.deque[np.ndarray] = collections.deque(
            maxlen=jitter_window_len
        )

    def update_landmarks(self, keypoints: Optional[np.ndarray]) -> float:
        """Update landmark history and return current jitter (mean std in px).

        Parameters
        ----------
        keypoints:
            Array of shape (K, 2) or (K, 3) representing key facial landmark coordinates,
            or None if face lost.

        Returns
        -------
        float:
            Spatial standard deviation in pixels, or NaN if face lost or buffer < 3.
        """
        if keypoints is None or len(keypoints) == 0:
            self._landmark_history.clear()
            return float("nan")

        coords = np.asarray(keypoints, dtype=np.float32)[:, :2]
        self._landmark_history.append(coords)

        if len(self._landmark_history) < 3:
            return 0.0

        stacked = np.stack(self._landmark_history, axis=0)  # (W, K, 2)
        # Compute std across time per landmark, then average across landmarks and x,y
        std_per_lm = np.std(stacked, axis=0)  # (K, 2)
        jitter = float(np.mean(std_per_lm))
        return jitter

    def extract_eye_quality_vector(self, signals: QualitySignals) -> np.ndarray:
        """Extract feature vector for Eye Reliability Model.

        Features:
          0: brightness deviation from ideal (ideal ~ 110, normalised)
          1: contrast (RMS contrast, normalized by 50.0)
          2: glare fraction ([0, 1])
          3: sharpness (normalized by 100.0)
          4: head angle in z-scores (large angle degrades eye visibility)
          5: landmark jitter (px)
          6: face detected flag (1.0 or 0.0)

        Returns
        -------
        np.ndarray of shape (7,), float32
        """
        if not signals.face_detected or math.isnan(signals.eye_brightness):
            return np.array([1.0, 0.0, 1.0, 0.0, 5.0, 10.0, 0.0], dtype=np.float32)

        # Normalizations
        # Note: R4 (cues/eye_crop.py) and R3 (preprocess-rldd.py) provide brightness
        # and contrast divided by 255.0 (range [0, 1]). We scale by 255.0 to map to nominal
        # pixel intensity range [0, 255]. Supporting <= 1.0 ensures seamless compatibility
        # whether receiving normalized [0, 1] inputs or direct [0, 255] values.
        b_val = signals.eye_brightness * 255.0 if signals.eye_brightness <= 1.0 else signals.eye_brightness
        c_val = signals.eye_contrast * 255.0 if signals.eye_contrast <= 1.0 else signals.eye_contrast

        b_dev = abs(b_val - 110.0) / 110.0
        c_norm = np.clip(c_val / 50.0, 0.0, 3.0)
        g_frac = np.clip(signals.eye_glare_frac, 0.0, 1.0)
        s_norm = np.clip(signals.eye_sharpness / 100.0, 0.0, 5.0)

        head_z = signals.head_angle_z if not math.isnan(signals.head_angle_z) else 0.0
        jitter = signals.landmark_jitter if not math.isnan(signals.landmark_jitter) else 0.0
        det = 1.0 if signals.face_detected else 0.0

        return np.array(
            [b_dev, c_norm, g_frac, s_norm, head_z, jitter, det],
            dtype=np.float32,
        )

    def extract_mouth_quality_vector(self, signals: QualitySignals) -> np.ndarray:
        """Extract feature vector for Mouth Reliability Model.

        Features:
          0: head angle in z-scores (severe turns occlude mouth)
          1: landmark jitter (px)
          2: eye/face sharpness as proxy for camera focus
          3: face detected flag
        """
        if not signals.face_detected:
            return np.array([5.0, 10.0, 0.0, 0.0], dtype=np.float32)

        head_z = signals.head_angle_z if not math.isnan(signals.head_angle_z) else 0.0
        jitter = signals.landmark_jitter if not math.isnan(signals.landmark_jitter) else 0.0
        s_norm = np.clip(signals.eye_sharpness / 100.0, 0.0, 5.0) if not math.isnan(signals.eye_sharpness) else 1.0
        det = 1.0 if signals.face_detected else 0.0

        return np.array([head_z, jitter, s_norm, det], dtype=np.float32)

    def extract_head_pose_quality_vector(self, signals: QualitySignals) -> np.ndarray:
        """Extract feature vector for Head Pose Reliability Model.

        Features:
          0: extreme angle magnitude (beyond ±45 deg / 3.5 z-scores, PnP accuracy drops)
          1: landmark jitter
          2: detector confidence
          3: face detected flag
        """
        if not signals.face_detected:
            return np.array([5.0, 10.0, 0.0, 0.0], dtype=np.float32)

        head_z = signals.head_angle_z if not math.isnan(signals.head_angle_z) else 0.0
        jitter = signals.landmark_jitter if not math.isnan(signals.landmark_jitter) else 0.0
        conf = signals.detector_confidence if not math.isnan(signals.detector_confidence) else 1.0
        det = 1.0 if signals.face_detected else 0.0

        return np.array([head_z, jitter, conf, det], dtype=np.float32)
