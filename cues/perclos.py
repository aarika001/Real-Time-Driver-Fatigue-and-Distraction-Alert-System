"""PERCLOS and Baseline B2 (CNN eye state + PERCLOS rule) for R4 evaluation.

PERCLOS (Dinges & Grace, 1998) is the share of time the eyes are closed over
a sliding window (classically: at least 80% closed, over ~1 minute). Our CNN
gives P(closed) per frame; a frame counts as closed when P(closed) >= 0.5.
That is an approximation of the "80% closed" definition, stated as such in
the report.

Baseline B2 from the proposal = this PERCLOS with a fixed alert threshold.
Our full system (R6 GRU + R7 policy) is compared against it.
"""
from __future__ import annotations

import numpy as np


def perclos(p_closed: np.ndarray, fps: float, window_s: float = 60.0,
            closed_threshold: float = 0.5, min_valid_frac: float = 0.5) -> np.ndarray:
    """Running PERCLOS per frame over the previous ``window_s`` seconds.

    NaN frames (face or eyes not visible) are ignored; if fewer than
    ``min_valid_frac`` of the window is valid the result is NaN, so B2 does not
    raise alerts from a handful of frames.
    """
    p = np.asarray(p_closed, dtype=np.float64)
    valid = np.isfinite(p)
    closed = (np.where(valid, p, 0.0) >= closed_threshold) & valid
    w = max(1, int(round(window_s * fps)))
    c_closed = np.concatenate([[0], np.cumsum(closed)])
    c_valid = np.concatenate([[0], np.cumsum(valid)])
    i = np.arange(1, len(p) + 1)
    lo = np.maximum(0, i - w)
    n_valid = c_valid[i] - c_valid[lo]
    n_closed = c_closed[i] - c_closed[lo]
    window_len = i - lo
    out = np.full(len(p), np.nan)
    ok = (n_valid > 0) & (n_valid >= min_valid_frac * window_len)
    out[ok] = n_closed[ok] / n_valid[ok]
    return out


def b2_alerts(perclos_series: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    """Baseline B2 rule: alert (1.0) whenever PERCLOS exceeds a fixed threshold."""
    s = np.asarray(perclos_series, dtype=np.float64)
    return np.where(np.isfinite(s), (s > threshold).astype(float), 0.0)
