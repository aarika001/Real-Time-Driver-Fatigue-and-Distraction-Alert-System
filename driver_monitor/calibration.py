"""Per-driver calibration: first N seconds of normal driving set personal baselines."""
import numpy as np
import config
from interfaces import Features, Percept


class Calibrator:
    def __init__(self, seconds: float = config.CALIB_SECONDS):
        self.seconds = seconds
        self.min_frames = config.CALIB_MIN_FRAMES if seconds >= 5 else 5
        self.reset()

    def reset(self):
        self.t0 = None
        self.samples = []
        self.base = None

    @property
    def done(self) -> bool:
        return self.base is not None

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
        if p.face_found:
            self.samples.append((p.ear, p.mar, p.yaw, p.pitch))
        if t - self.t0 >= self.seconds and len(self.samples) >= self.min_frames:
            m = np.median(np.array(self.samples), axis=0)
            self.base = {"ear": max(float(m[0]), 1e-3), "mar": float(m[1]),
                         "yaw": float(m[2]), "pitch": float(m[3])}

    def normalise(self, p: Percept) -> Features:
        if not p.face_found:
            return Features()
        b = self.base
        return Features(p_eye_closed=p.p_eye_closed, ear_rel=p.ear / b["ear"],
                        mar_rel=p.mar - b["mar"], yaw_rel=p.yaw - b["yaw"],
                        pitch_rel=p.pitch - b["pitch"])
