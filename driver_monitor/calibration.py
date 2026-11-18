"""Per-driver calibration (wraps R6's temporal_model/features.py)."""
import os
import sys
import numpy as np

_TM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "temporal_model"))
if _TM_DIR not in sys.path:
    sys.path.insert(0, _TM_DIR)

import features as ft
import config
from interfaces import Features, Percept


class Calibrator:
    def __init__(self, seconds: float = config.CALIB_SECONDS):
        self.seconds = seconds
        self.min_frames = config.CALIB_MIN_FRAMES if seconds >= 5 else 5
        self.reset()

    def reset(self):
        self.t0 = None
        self.ear_samples = []
        self.mar_samples = []
        self.pose_samples = []
        self.det_samples = []
        self.base = None
        self.r6_base = None

    @property
    def done(self) -> bool:
        return self.r6_base is not None

    def progress(self, t: float) -> float:
        if self.done:
            return 1.0
        return 0.0 if self.t0 is None else min((t - self.t0) / self.seconds, 1.0)

    def waiting_for_face(self, t: float) -> bool:
        return (not self.done and self.t0 is not None and t - self.t0 >= self.seconds)

    def add(self, p: Percept, t: float):
        if self.done:
            return
        if self.t0 is None:
            self.t0 = t
        det = 1 if p.face_found else 0
        self.det_samples.append(det)
        self.mar_samples.append(p.mar if p.face_found else 0.0)
        self.pose_samples.append([p.pitch, p.yaw, p.roll] if p.face_found else [0.0, 0.0, 0.0])
        if p.face_found:
            self.ear_samples.append(p.ear)

        n_det = sum(self.det_samples)
        if t - self.t0 >= self.seconds and n_det >= self.min_frames:
            mar_arr = np.asarray(self.mar_samples, dtype=np.float32)
            pose_arr = np.asarray(self.pose_samples, dtype=np.float32)
            det_arr = np.asarray(self.det_samples, dtype=np.uint8)
            self.r6_base = ft.fit_baseline(mar_arr, pose_arr, det_arr, min_frames=self.min_frames)
            self.base = {
                "ear": max(float(np.median(self.ear_samples)), 1e-3),
                "mar": float(self.r6_base.mar_mu),
                "pitch": float(self.r6_base.pose_mu[0]),
                "yaw": float(self.r6_base.pose_mu[1]),
                "roll": float(self.r6_base.pose_mu[2]),
            }

    def normalise(self, p: Percept) -> Features:
        if not p.face_found:
            raw_q = [0.0, 0.0, 1.0, 0.0]
        elif p.raw_quality is not None:
            raw_q = list(p.raw_quality)
        else:
            q = p.quality or {}
            raw_q = [
                float(q.get("brightness", 0.5)),
                float(q.get("contrast", 0.5) * 80.0 / 255.0),
                float(q.get("glare_frac", 0.0)),
                float(q.get("sharpness", 0.5) * 500.0),
            ]

        rec = ft.VideoRec(
            subject="live", fold="live", kss="0", label="live",
            mar=np.array([p.mar if p.face_found else 0.0], dtype=np.float32),
            pose=np.array([[p.pitch, p.yaw, p.roll] if p.face_found else [0.0, 0.0, 0.0]], dtype=np.float32),
            quality=np.array([raw_q], dtype=np.float32),
            det=np.array([1 if p.face_found else 0], dtype=np.uint8),
            eye=np.array([p.p_eye_closed if p.face_found else 0.0], dtype=np.float32),
        )
        gru_row = ft.frame_features(rec, self.r6_base)[0]

        if not p.face_found:
            return Features(face_found=False, gru_row=gru_row)

        b = self.base
        pitch_rel = float(ft.wrap_deg(p.pitch - b["pitch"]))
        yaw_rel = float(ft.wrap_deg(p.yaw - b["yaw"]))
        return Features(
            p_eye_closed=p.p_eye_closed,
            ear_rel=p.ear / b["ear"],
            mar_rel=p.mar - b["mar"],
            yaw_rel=yaw_rel,
            pitch_rel=pitch_rel,
            face_found=True,
            gru_row=gru_row,
        )