"""Eye-state model."""
import os
from cues.predictors import load_eye_predictor

predictor = None

def _get_predictor():
    global predictor
    if predictor is None:
        # Load R4's ONNX model from the models folder
        model_path = os.path.join(os.path.dirname(__file__), "models", "eye_cnn.onnx")
        if os.path.exists(model_path):
            predictor = load_eye_predictor(model_path)
    return predictor

def predict_eye_closed(eye_crops, ear: float) -> float:
    pred = _get_predictor()
    if pred is None or eye_crops is None or eye_crops[0] is None or eye_crops[1] is None:
        return 0.0
    
    import numpy as np
    # R4 expects shape (B, 64, 64)
    batch = np.stack(eye_crops)
    p = pred.predict_crops(batch)
    # Return average P(closed) of both eyes
    return float(np.mean(p))