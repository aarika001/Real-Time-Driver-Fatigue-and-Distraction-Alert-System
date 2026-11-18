"""Eye-state model. STUB: R4 replaces this with the fine-tuned MobileNetV2.

Contract: predict_eye_closed(eye_crops, ear) -> float in [0, 1] = P(eye closed).
eye_crops is (left, right), each a 64x64 uint8 grayscale crop (or None).
"""
import numpy as np


def predict_eye_closed(eye_crops, ear: float) -> float:
    # Placeholder: squash EAR around a typical closed/open boundary.
    return float(1.0 / (1.0 + np.exp((ear - 0.20) / 0.03)))
