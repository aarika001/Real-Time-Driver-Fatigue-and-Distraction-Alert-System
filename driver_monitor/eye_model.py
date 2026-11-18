"""Eye-state model (wraps R4's ONNX MobileNetV2 predictor)."""
import os
import numpy as np
from cues.predictors import load_eye_predictor

predictor = None


def _get_predictor():
    global predictor
    if predictor is None:
        model_path = os.path.join(os.path.dirname(__file__), "models", "eye_cnn.onnx")
        if os.path.exists(model_path):
            predictor = load_eye_predictor(model_path, num_threads=2)
        else: print(f"[WARNING] Missing {model_path} — falling back to EAR")
    return predictor


def predict_eye_closed(eye_crops, ear: float) -> float:
    pred = _get_predictor()
    if pred is None or eye_crops is None or eye_crops[0] is None or eye_crops[1] is None:
        return float(1.0 / (1.0 + np.exp((ear - 0.20) / 0.03)))

    batch = np.stack(eye_crops)
    p = pred.predict_crops(batch)
    return float(np.mean(p))