"""Mouth cue for R4: mouth aspect ratio (MAR), per-driver calibration and
yawn-vs-talking features.

Nothing here is trained. MAR is geometry; the yawn rule is a transparent
threshold-and-duration rule whose thresholds are tuned on YawDD
(``scripts/r4/eval_yawdd.py``). R6's GRU receives the per-frame features and
learns the finer temporal patterns.

Two MAR variants
----------------
``mar`` (outer lip, 2-D)
    Identical to R3's ``compute_mar`` in RLDD preprocessing, so it matches the
    values stored in ``processed_rldd/*.npz``. Includes lip thickness, so a
    closed mouth is roughly 0.2-0.4, not 0.
``mar_inner`` (inner lip, 3-D)
    Measures the actual opening. Uses MediaPipe's z coordinate, so when the
    head turns (yaw) the mouth width does not shrink artificially and the
    value stays stable. 2-D MAR inflates as the head turns because the width
    is foreshortened.

Why normalise per driver
------------------------
Resting MAR differs between people (lip shape, beard, camera height). Dividing
by each driver's own neutral MAR from the 30-s calibration (``mar_norm``) puts
everyone on the same scale: ~1.0 at rest, ~2-3 during a yawn.

Yawn vs talking
---------------
A yawn opens the mouth wide and holds it (typically 2-6 s, one smooth event).
Talking opens it less, for short bursts, several times per second. So:
    open_run_s  long  + osc_hz low   -> yawn
    open_run_s  short + osc_hz high  -> talking
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .landmarks import MOUTH_INNER, MOUTH_OUTER
from .schemas import NAN, MouthResult

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class MouthConfig:
    """Thresholds for the mouth cue. Ratios are relative to the driver's neutral MAR."""

    fps: float = 15.0               # frame rate the features are computed at
    min_calib_frames: int = 30      # minimum valid frames needed to calibrate (2 s)
    open_ratio: float = 1.6         # mar_norm >= this counts as "wide open"
    talk_ratio: float = 1.15        # upward crossings of this count as one open/close cycle
    yawn_peak_ratio: float = 2.0    # a yawn must reach at least this mar_norm
    min_yawn_s: float = 1.5         # ... and stay wide open at least this long
    osc_window_s: float = 2.0       # window for the oscillation rate
    max_gap_s: float = 0.5          # face-lost gap tolerated inside a yawn (hand over mouth)

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "MouthConfig":
        """Build from a dict (e.g. the ``mouth:`` section of configs/r4.yaml).
        Unknown keys raise, so a typo in the YAML is caught instead of ignored."""
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"Unknown mouth config keys: {sorted(unknown)}")
        cfg = cls(**d)
        if cfg.yawn_peak_ratio < cfg.open_ratio:
            raise ValueError("yawn_peak_ratio must be >= open_ratio")
        if cfg.talk_ratio >= cfg.open_ratio:
            raise ValueError("talk_ratio must be < open_ratio")
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# MAR geometry
# ---------------------------------------------------------------------------
def _mar(points: np.ndarray, idxs: Sequence[int]) -> float:
    """Mean of three vertical lip distances divided by mouth width.

    ``idxs`` order: left corner, right corner, then three (upper, lower) pairs.
    """
    p = points[list(idxs)].astype(np.float64)
    horiz = np.linalg.norm(p[0] - p[1])
    vert = (np.linalg.norm(p[2] - p[3]) + np.linalg.norm(p[4] - p[5])
            + np.linalg.norm(p[6] - p[7]))
    return float(vert / (3.0 * horiz + _EPS))


def mar_outer(landmarks_px: np.ndarray) -> float:
    """Outer-lip 2-D MAR, identical to R3's RLDD ``compute_mar``."""
    return _mar(landmarks_px[:, :2], MOUTH_OUTER)


def mar_inner(landmarks_px: np.ndarray) -> float:
    """Inner-lip 3-D MAR (robust to head yaw)."""
    return _mar(landmarks_px[:, :3], MOUTH_INNER)


def robust_baseline(values: Sequence[float], min_count: int = 30) -> float:
    """Median of the finite values; the per-driver neutral MAR.

    The median ignores the odd smile or word during calibration, which a mean
    would not. Raises if fewer than ``min_count`` valid frames are available.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < min_count:
        raise ValueError(f"Need >= {min_count} valid MAR frames to calibrate, got {v.size}")
    base = float(np.median(v))
    if base <= _EPS:
        raise ValueError("Neutral MAR is ~0; check that the outer-lip MAR is being used")
    return base


# ---------------------------------------------------------------------------
# Temporal yawn features
# ---------------------------------------------------------------------------
class YawnTracker:
    """Streaming yawn-vs-talking features from a ``mar_norm`` sequence.

    Call ``step`` once per frame at ``cfg.fps``; pass NaN for face-lost frames.
    """

    def __init__(self, cfg: MouthConfig):
        self.cfg = cfg
        self._max_gap = int(round(cfg.max_gap_s * cfg.fps))
        self._osc_window = int(round(cfg.osc_window_s * cfg.fps))
        self.reset()

    def reset(self) -> None:
        self._t = -1
        self._run = 0            # frames in the current wide-open run
        self._peak = 0.0         # highest mar_norm in the current run
        self._gap = 0            # consecutive face-lost frames
        self._above_talk = False
        self._crossings: deque[int] = deque()

    def step(self, mar_norm: float) -> tuple[float, float, float]:
        """Returns (open_run_s, osc_hz, yawn_flag) for this frame."""
        self._t += 1
        if not np.isfinite(mar_norm):
            self._gap += 1
            if self._gap > self._max_gap:
                self._run, self._peak = 0, 0.0
            elif self._run > 0:
                self._run += 1   # a yawn in progress behind a hand keeps counting
        else:
            self._gap = 0
            above = mar_norm >= self.cfg.talk_ratio
            if above and not self._above_talk:
                self._crossings.append(self._t)
            self._above_talk = above
            if mar_norm >= self.cfg.open_ratio:
                self._run += 1
                self._peak = max(self._peak, float(mar_norm))
            else:
                self._run, self._peak = 0, 0.0

        while self._crossings and self._t - self._crossings[0] >= self._osc_window:
            self._crossings.popleft()

        open_run_s = self._run / self.cfg.fps
        osc_hz = len(self._crossings) / self.cfg.osc_window_s
        is_yawn = open_run_s >= self.cfg.min_yawn_s and self._peak >= self.cfg.yawn_peak_ratio
        return open_run_s, osc_hz, 1.0 if is_yawn else 0.0


def yawn_features(mar: np.ndarray, neutral_mar: float, cfg: MouthConfig) -> dict[str, np.ndarray]:
    """Offline version of the streaming features for a whole sequence.

    Args:
        mar: (N,) outer MAR per frame at ``cfg.fps``; NaN where the face was lost.
        neutral_mar: the driver's calibrated neutral MAR.
        cfg: thresholds.

    Returns:
        dict of (N,) float32 arrays: mar_norm, open_run_s, osc_hz, yawn_flag.
    """
    mar = np.asarray(mar, dtype=np.float64)
    norm = mar / neutral_mar
    tracker = YawnTracker(cfg)
    out = np.array([tracker.step(x) for x in norm], dtype=np.float32).reshape(-1, 3)
    return {"mar_norm": norm.astype(np.float32), "open_run_s": out[:, 0],
            "osc_hz": out[:, 1], "yawn_flag": out[:, 2]}


def yawn_events(yawn_flag: np.ndarray, open_run_s: np.ndarray,
                fps: float) -> list[tuple[float, float, float]]:
    """Group per-frame flags into events.

    Returns a list of (start_s, end_s, detect_s): start of the wide-open run,
    end of the flagged period, and the moment the rule first fired
    (detect_s - start_s is the detection delay).
    """
    events = []
    on = False
    start = detect = 0.0
    for i, f in enumerate(np.asarray(yawn_flag)):
        if f >= 0.5 and not on:
            on = True
            detect = i / fps
            start = max(0.0, detect - float(open_run_s[i]) + 1.0 / fps)
        elif f < 0.5 and on:
            on = False
            events.append((start, i / fps, detect))
    if on:
        events.append((start, len(yawn_flag) / fps, detect))
    return events


# ---------------------------------------------------------------------------
# Live / per-frame extractor
# ---------------------------------------------------------------------------
class MouthCueExtractor:
    """Per-frame mouth cue for the live pipeline (R8) and calibration (R6).

    Usage::

        mouth = MouthCueExtractor(MouthConfig())
        # calibration stage (R6): collect ~30 s of neutral frames
        mars = [mouth.update(lm).mar for lm in calibration_frames]
        mouth.calibrate(mars)
        # then every frame
        result = mouth.update(landmarks_px_or_None)
    """

    def __init__(self, cfg: Optional[MouthConfig] = None):
        self.cfg = cfg or MouthConfig()
        self.neutral_mar: Optional[float] = None
        self._tracker = YawnTracker(self.cfg)

    @property
    def calibrated(self) -> bool:
        return self.neutral_mar is not None

    def calibrate(self, neutral_mar_values: Sequence[float]) -> float:
        """Set the per-driver neutral MAR (median of the given values)."""
        self.neutral_mar = robust_baseline(neutral_mar_values, self.cfg.min_calib_frames)
        self._tracker.reset()
        return self.neutral_mar

    def reset(self) -> None:
        """Forget calibration and temporal state (new driver)."""
        self.neutral_mar = None
        self._tracker.reset()

    def update(self, landmarks_px: Optional[np.ndarray]) -> MouthResult:
        if landmarks_px is None:
            if not self.calibrated:
                return MouthResult()
            run, osc, flag = self._tracker.step(NAN)
            return MouthResult(valid=False, open_run_s=run, osc_hz=osc, yawn_flag=flag)

        m, mi = mar_outer(landmarks_px), mar_inner(landmarks_px)
        if not self.calibrated:
            return MouthResult(mar=m, mar_inner=mi, valid=True)
        norm = m / self.neutral_mar
        run, osc, flag = self._tracker.step(norm)
        return MouthResult(mar=m, mar_inner=mi, mar_norm=norm, valid=True,
                           open_run_s=run, osc_hz=osc, yawn_flag=flag)


# ---------------------------------------------------------------------------
# Bridge to R3's processed RLDD archives
# ---------------------------------------------------------------------------
def load_rldd_mar(npz_path: str | Path) -> np.ndarray:
    """Outer MAR from a ``processed_rldd`` archive with face-lost frames set to NaN
    (R3 stores 0 for them, which would look like a closed mouth)."""
    with np.load(npz_path) as z:
        mar = z["mar"].astype(np.float64)
        mar[z["face_detected"] == 0] = np.nan
    return mar
