"""Temporal model (wraps R6's 4-fold TemporalGRU ensemble + distraction rule)."""
from collections import deque
import glob
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)

_TM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "temporal-model"))
if _TM_DIR not in sys.path:
    sys.path.insert(0, _TM_DIR)

import features as ft
from model import TemporalGRU
import config
from interfaces import TemporalOut

_CKPT_GLOB = os.path.join(_TM_DIR, "runs", "with_eye", "model_Fold*.pt")
_ENSEMBLE = None


def _load_ensemble():
    global _ENSEMBLE
    if _ENSEMBLE is not None:
        return _ENSEMBLE
    models = []
    for path in sorted(glob.glob(_CKPT_GLOB)):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = TemporalGRU(
            n_features=ft.N_FEATURES,
            hidden=int(ckpt.get("hidden", 64)),
            n_classes=int(ckpt.get("n_classes", 2)),
            dropout=0.0,
        )
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models.append((m, float(ckpt.get("temperature", 1.0))))
    if not models: print(f"[WARNING] No GRU checkpoints found at {_CKPT_GLOB}")
    _ENSEMBLE = models
    return _ENSEMBLE


class TemporalModel:
    def __init__(self):
        self.buf = deque()      # (t, gru_row_13, yaw_rel, pitch_rel, r_head)
        self.ensemble = _load_ensemble()
        self._last_gru_t = -1.0
        self._last_pf = 0.0

    def update(self, f, r, t) -> TemporalOut:
        if f.gru_row is not None:
            row = f.gru_row.astype(np.float32, copy=True)
        else:
            row = np.zeros(ft.N_FEATURES, dtype=np.float32)
            row[0] = f.p_eye_closed
            row[ft.REL_START + 4] = 1.0 if f.face_found else 0.0
            row[ft.FLAG_START:] = 1.0

        # Reliability-aware cue masking (matches R6's cue_dropout training contract):
        # hide an unreliable cue group only if at least one other cue group is reliable.
        rels = (r.eye, r.mouth, r.head)
        if max(rels) >= config.R_MIN:
            for g, rv in enumerate(rels):
                if rv < config.R_MIN:
                    for c in ft.GROUP_CUE_IDX[ft.GROUP_NAMES[g]]:
                        row[c] = 0.0
                    row[ft.FLAG_START + g] = 0.0

        self.buf.append((t, row, abs(f.yaw_rel), abs(f.pitch_rel), r.head))
        while self.buf and t - self.buf[0][0] > config.WINDOW_SECONDS:
            self.buf.popleft()

        ts = np.array([b[0] for b in self.buf])
        if ts[-1] - ts[0] < config.STUB_WARMUP_S:
            return TemporalOut(np.array([1.0, 0.0, 0.0], dtype=np.float32), {"warmup": True})

        # 1. Fatigue probability from R6's 4-fold GRU ensemble (4 Hz refresh + EMA)
        if self.ensemble and (t - self._last_gru_t >= 0.25):
            win = self.window_array()
            x = torch.from_numpy(win[None])                     # (1, 75, 13)
            with torch.inference_mode():
                probs_2 = sum(F.softmax(m(x) / temp, dim=-1)[0] for m, temp in self.ensemble) / len(self.ensemble)
            raw_pf = float(probs_2[1].item())
            self._last_pf = 0.25 * raw_pf + 0.75 * self._last_pf if self._last_gru_t >= 0 else raw_pf
            self._last_gru_t = t
        pf = self._last_pf

        # 2. Distraction probability from sustained head deviation (brief mirror checks stay < 0.3)
        yaws = np.array([b[2] for b in self.buf], dtype=np.float64)
        pitches = np.array([b[3] for b in self.buf], dtype=np.float64)
        r_h = np.array([b[4] for b in self.buf], dtype=np.float64)
        away_flag = ((yaws > config.YAW_AWAY_DEG) | (pitches > config.PITCH_AWAY_DEG)).astype(np.float64)
        away = float((away_flag * r_h).sum() / (r_h.sum() + 1e-6))
        dist_s = float(np.clip((away - 0.3) / 0.4, 0.0, 1.0))
        pd = dist_s * float(r_h.mean())

        # Combine into 3-class [alert, fatigue, distraction] for R7's utility policy
        pa = (1.0 - pf) * (1.0 - pd)
        probs = np.array([pa, pf, pd], dtype=np.float32)
        probs = probs / max(float(probs.sum()), 1e-6)
        return TemporalOut(probs, {"p_fatigue_gru": pf, "away": away, "n_folds": len(self.ensemble)})

    def window_array(self) -> np.ndarray:
        """(75, 13) float32 window resampled onto a 15 FPS grid."""
        W = ft.WINDOW
        if not self.buf:
            return np.zeros((W, ft.N_FEATURES), np.float32)
        ts = np.array([b[0] for b in self.buf])
        rows = np.stack([b[1] for b in self.buf]).astype(np.float32)
        grid = np.linspace(ts[-1] - config.WINDOW_SECONDS, ts[-1], W)
        return rows[np.clip(np.searchsorted(ts, grid), 0, len(ts) - 1)]