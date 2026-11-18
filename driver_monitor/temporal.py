"""Temporal model. STUB: rule-based PERCLOS / yawn / look-away over a 5 s window.

R6 replaces update()'s body with the GRU, reading window_array().
Contract: update(features, reliability, t) -> TemporalOut(probs over config.STATES).
"""
from collections import deque
import numpy as np
import config
from interfaces import TemporalOut, FEATURE_NAMES


class TemporalModel:
    def __init__(self):
        self.buf = deque()      # (t, row) with row ordered as FEATURE_NAMES

    def update(self, f, r, t) -> TemporalOut:
        self.buf.append((t, [f.p_eye_closed, f.ear_rel, f.mar_rel, f.yaw_rel, f.pitch_rel,
                             r.eye, r.mouth, r.head]))
        while self.buf and t - self.buf[0][0] > config.WINDOW_SECONDS:
            self.buf.popleft()
        return self._stub()

    def window_array(self) -> np.ndarray:
        """(WINDOW_FRAMES, len(FEATURE_NAMES)) float32 window resampled to a fixed grid.
        This is exactly the tensor the GRU should consume."""
        W = config.WINDOW_FRAMES
        if not self.buf:
            return np.zeros((W, len(FEATURE_NAMES)), np.float32)
        ts = np.array([b[0] for b in self.buf])
        rows = np.array([b[1] for b in self.buf], dtype=np.float32)
        grid = np.linspace(ts[-1] - config.WINDOW_SECONDS, ts[-1], W)
        return rows[np.clip(np.searchsorted(ts, grid), 0, len(ts) - 1)]

    # ------------------------------------------------------------------
    def _stub(self) -> TemporalOut:
        ts = np.array([b[0] for b in self.buf])
        if ts[-1] - ts[0] < config.STUB_WARMUP_S:
            return TemporalOut(np.array([1.0, 0.0, 0.0]), {"warmup": True})
        rows = np.array([b[1] for b in self.buf], dtype=np.float64)
        ear, mar = rows[:, 1], rows[:, 2]
        yaw, pitch = np.abs(rows[:, 3]), np.abs(rows[:, 4])
        r_e, r_m, r_h = rows[:, 5], rows[:, 6], rows[:, 7]

        def wfrac(flag, w):          # reliability-weighted fraction of the window
            return float((flag * w).sum() / (w.sum() + 1e-6))

        closed = np.clip((config.EAR_CLOSED_REL - ear) / 0.35, 0, 1)
        perclos = wfrac(closed, r_e)
        yawn = wfrac((mar > config.MAR_YAWN_DELTA).astype(float), r_m)
        away = wfrac(((yaw > config.YAW_AWAY_DEG) | (pitch > config.PITCH_AWAY_DEG)).astype(float), r_h)

        eye_s = float(np.clip((perclos - 0.15) / 0.25, 0, 1))
        yawn_s = float(np.clip((yawn - 0.3) / 0.3, 0, 1))
        dist_s = float(np.clip((away - 0.3) / 0.4, 0, 1))      # brief mirror checks stay below 0.3

        pf = 1 - (1 - eye_s * r_e.mean()) * (1 - yawn_s * r_m.mean())
        pd = dist_s * r_h.mean()
        probs = np.array([(1 - pf) * (1 - pd), pf, pd])
        probs = probs / probs.sum()
        return TemporalOut(probs, {"perclos": perclos, "yawn": yawn, "away": away})
