"""Eye-state predictors behind one interface: ``predict_crops(crops) -> P(closed)``.

* ``.pt``   -> ``eye_model.EyeStatePredictor`` (PyTorch)
* ``.onnx`` -> ``OnnxEyePredictor`` (ONNX Runtime; no PyTorch needed, usually
  faster on a laptop CPU). Create the file with scripts/r4/export_onnx.py.

Both take (B, 64, 64) uint8 CLAHE-normalised crops and return (B,) float32
calibrated P(closed); the ONNX graph already contains the temperature and the
sigmoid.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class OnnxEyePredictor:
    def __init__(self, onnx_path: str | Path, num_threads: Optional[int] = None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if num_threads:
            opts.intra_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(onnx_path), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict_crops(self, crops: np.ndarray) -> np.ndarray:
        crops = np.asarray(crops)
        if crops.ndim == 2:
            crops = crops[None]
        x = (crops.astype(np.float32) / 255.0)[:, None]
        (p,) = self.session.run(None, {self.input_name: x})
        return np.asarray(p, dtype=np.float32).reshape(-1)


def load_eye_predictor(path: str | Path, num_threads: Optional[int] = None):
    """Pick the right predictor from the file extension."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Eye model not found: {path}")
    if path.suffix.lower() == ".onnx":
        return OnnxEyePredictor(path, num_threads)
    from .eye_model import EyeStatePredictor   # imports PyTorch only when needed
    return EyeStatePredictor(path, num_threads=num_threads)
