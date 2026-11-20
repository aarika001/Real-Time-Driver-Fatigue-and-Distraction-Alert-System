"""Cue Reliability Estimator using Logistic Regression (R7).

Predicts the probability that each sensory cue (eye state, mouth MAR/yawn,
head pose) is trustworthy under real-world degradations (glare, blur, low light,
steep camera angle, occlusion, or landmark jitter).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from .quality_features import QualityFeatureExtractor, QualitySignals


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


class CueReliabilityEstimator:
    """Logistic regression reliability estimator for eye, mouth, and head pose cues."""

    def __init__(self, feature_extractor: Optional[QualityFeatureExtractor] = None):
        self.extractor = feature_extractor or QualityFeatureExtractor()

        # Sklearn LogisticRegression models per cue
        self.models: Dict[str, Optional[LogisticRegression]] = {
            "eye": None,
            "mouth": None,
            "head_pose": None,
        }

        # Calibrated default weights based on physical optics & stress sweep data
        # Eye weights: [b_dev, contrast, glare, sharpness, head_z, jitter, face_detected]
        self._default_weights = {
            "eye": (
                np.array([-1.5, 1.8, -4.5, 1.2, -0.9, -0.6, 2.5], dtype=np.float32),
                0.8,  # intercept
            ),
            # Mouth weights: [head_z, jitter, sharpness, face_detected]
            "mouth": (
                np.array([-1.2, -0.5, 0.8, 2.0], dtype=np.float32),
                1.0,
            ),
            # Head pose weights: [head_z, jitter, conf, face_detected]
            "head_pose": (
                np.array([-1.0, -0.8, 1.5, 2.5], dtype=np.float32),
                1.2,
            ),
        }

    def fit_cue(self, cue_name: str, X: np.ndarray, y: np.ndarray, C: float = 1.0) -> None:
        """Fit logistic regression model for a specific cue using quality features X and ground-truth agreement y.

        Parameters
        ----------
        cue_name: 'eye', 'mouth', or 'head_pose'
        X: shape (N, D) quality feature matrix
        y: binary array (1 = cue prediction agreed with ground truth, 0 = incorrect/unreliable)
        C: inverse regularization strength
        """
        if cue_name not in self.models:
            raise ValueError(f"Unknown cue: {cue_name}. Must be 'eye', 'mouth', or 'head_pose'.")

        clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
        clf.fit(X, y)
        self.models[cue_name] = clf

    def predict_cue_reliability(self, cue_name: str, x_feat: np.ndarray) -> float:
        """Return scalar probability in [0, 1] that the given cue is reliable."""
        # Hard zero if face is lost
        if cue_name == "eye" and x_feat[-1] == 0.0:
            return 0.0
        if cue_name in ("mouth", "head_pose") and x_feat[-1] == 0.0:
            return 0.0

        model = self.models.get(cue_name)
        if model is not None:
            # Predict P(y=1) using fitted scikit-learn model
            prob = model.predict_proba(x_feat.reshape(1, -1))[0, 1]
            return float(prob)

        # Fallback to analytical calibrated weights
        weights, intercept = self._default_weights[cue_name]
        logit = float(np.dot(weights, x_feat) + intercept)
        return float(sigmoid(logit))

    def predict_all(self, signals: QualitySignals) -> Dict[str, float]:
        """Compute reliability scores for all cues for a given frame.

        Returns
        -------
        dict with keys: 'eye', 'mouth', 'head_pose', 'overall'
        """
        if not signals.face_detected:
            return {
                "eye": 0.0,
                "mouth": 0.0,
                "head_pose": 0.0,
                "overall": 0.0,
            }

        x_eye = self.extractor.extract_eye_quality_vector(signals)
        x_mouth = self.extractor.extract_mouth_quality_vector(signals)
        x_head = self.extractor.extract_head_pose_quality_vector(signals)

        r_eye = self.predict_cue_reliability("eye", x_eye)
        r_mouth = self.predict_cue_reliability("mouth", x_mouth)
        r_head = self.predict_cue_reliability("head_pose", x_head)

        # Overall reliability: harmonic mean or weighted average
        # If any single cue is strong, overall is not necessarily 0, but if all are degraded, it drops
        overall = float(0.45 * r_eye + 0.30 * r_mouth + 0.25 * r_head)

        return {
            "eye": r_eye,
            "mouth": r_mouth,
            "head_pose": r_head,
            "overall": overall,
        }
