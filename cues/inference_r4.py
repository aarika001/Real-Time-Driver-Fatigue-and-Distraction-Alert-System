"""R4 live pipeline: one call per webcam frame -> all eye and mouth cues (Milestone 6).

This is what R8's 15 FPS loop calls::

    from cues.inference_r4 import R4Pipeline
    from cues.predictors import load_eye_predictor

    pipe = R4Pipeline(eye_predictor=load_eye_predictor("runs/eye_cnn/eye_cnn.onnx"))
    pipe.start_calibration()                 # R6's 30-s calibration stage
    ...                                      # call pipe.process_frame(...) for ~30 s
    pipe.finish_calibration()
    res = pipe.process_frame(frame_bgr, timestamp_s)
    features = res.output.to_flat_dict()     # -> R6 (GRU), R7 (reliability), R8 (overlay)

Per frame: MediaPipe landmarks -> both eye crops in ONE batch of 2 through the
CNN -> EAR -> MAR + yawn features. Timings per stage are returned so the
>= 15 FPS claim can be measured, not assumed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from .eye_crop import crop_eye, eye_box
from .landmarks import EAR_IMG_LEFT, EAR_IMG_RIGHT, EYE_IMG_LEFT, EYE_IMG_RIGHT
from .mouth_cue import MouthConfig, MouthCueExtractor
from .schemas import NAN, EyeResult, R4FrameOutput


def eye_aspect_ratio(landmarks_px: np.ndarray, idxs: Sequence[int]) -> float:
    """Six-point EAR (Soukupova & Cech 2016): mean lid distance / eye width."""
    p = landmarks_px[list(idxs), :2].astype(np.float64)
    vert = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horiz = np.linalg.norm(p[0] - p[3])
    return float(vert / (2.0 * horiz + 1e-6))


@dataclass
class FrameResult:
    output: R4FrameOutput
    landmarks_px: Optional[np.ndarray] = None
    eye_boxes: tuple = ()                       # ((x1, y1, x2, y2) left, right) for overlays
    timings_ms: dict = field(default_factory=dict)


class R4Pipeline:
    """Eye + mouth cues for one driver.

    Args:
        eye_predictor: anything with ``predict_crops`` (``cues.predictors``),
            or None to run landmarks + mouth only.
        mouth_cfg: thresholds for the mouth cue.
        landmarker: an object with ``detect(rgb, timestamp_ms)`` returning
            (478, 3) pixels or None; defaults to MediaPipe in video mode.
    """

    def __init__(self, eye_predictor=None, mouth_cfg: Optional[MouthConfig] = None,
                 landmarker=None, landmarker_model_path: Optional[str] = None):
        if landmarker is None:
            from .face_landmarker import FaceLandmarker
            landmarker = FaceLandmarker(landmarker_model_path, mode="video")
        self.landmarker = landmarker
        self.eye_predictor = eye_predictor
        self.mouth = MouthCueExtractor(mouth_cfg)
        self._calib_mars: Optional[list[float]] = None

    # --- calibration hooks for R6 -------------------------------------------------
    @property
    def calibrating(self) -> bool:
        return self._calib_mars is not None

    def start_calibration(self) -> None:
        """Begin collecting neutral-face frames (driver looks ahead, mouth relaxed)."""
        self.mouth.reset()
        self._calib_mars = []

    def finish_calibration(self) -> float:
        """Set the per-driver baseline from the frames since ``start_calibration``."""
        if self._calib_mars is None:
            raise RuntimeError("start_calibration() was not called")
        mars, self._calib_mars = self._calib_mars, None
        return self.mouth.calibrate(mars)

    # --- per frame -------------------------------------------------------------------
    def process_frame(self, frame_bgr: np.ndarray, timestamp_s: float) -> FrameResult:
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pts = self.landmarker.detect(rgb, int(round(timestamp_s * 1000)))
        t1 = time.perf_counter()
        timings = {"landmarks": (t1 - t0) * 1000}

        if pts is None:
            mouth = self.mouth.update(None)
            timings["total"] = (time.perf_counter() - t0) * 1000
            return FrameResult(R4FrameOutput(timestamp_s, False, mouth=mouth), None, (), timings)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        left_pts, right_pts = pts[list(EYE_IMG_LEFT), :2], pts[list(EYE_IMG_RIGHT), :2]
        cl, ql, vl = crop_eye(gray, left_pts)
        cr, qr, vr = crop_eye(gray, right_pts)
        ear = float(np.mean([eye_aspect_ratio(pts, EAR_IMG_LEFT), eye_aspect_ratio(pts, EAR_IMG_RIGHT)]))
        t2 = time.perf_counter()
        timings["crops"] = (t2 - t1) * 1000

        p_l = p_r = NAN
        if self.eye_predictor is not None and (vl or vr):
            p = self.eye_predictor.predict_crops(np.stack([cl, cr]))   # both eyes in one batch
            p_l, p_r = float(p[0]), float(p[1])
        t3 = time.perf_counter()
        timings["eye_cnn"] = (t3 - t2) * 1000

        mouth = self.mouth.update(pts)
        if self._calib_mars is not None and mouth.valid:
            self._calib_mars.append(mouth.mar)
        timings["mouth"] = (time.perf_counter() - t3) * 1000
        timings["total"] = (time.perf_counter() - t0) * 1000

        # valid_* = the crop was inside the frame (quality signals are usable even
        # without the CNN); P(closed) stays NaN when no eye model is loaded.
        eye = EyeResult(p_closed_left=p_l, p_closed_right=p_r, valid_left=vl, valid_right=vr,
                        ear=ear, quality_left=ql if vl else None, quality_right=qr if vr else None)
        boxes = (eye_box(left_pts, w, h), eye_box(right_pts, w, h))
        return FrameResult(R4FrameOutput(timestamp_s, True, eye, mouth), pts, boxes, timings)
