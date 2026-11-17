"""Reliability estimator. STUB: heuristic from quality signals.

R7 replaces this with per-cue logistic regression.
Contract: estimate(percept) -> Reliability(eye, mouth, head), each in [0, 1].
"""
from interfaces import Percept, Reliability


def _trap(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    if x < b:
        return (x - a) / (b - a)
    return 1.0 if x <= c else (d - x) / (d - c)


def _clip01(x):
    return max(0.0, min(1.0, x))


def estimate(p: Percept) -> Reliability:
    if not p.face_found:
        return Reliability(0.0, 0.0, 0.0)
    q = p.quality
    light = _trap(q.get("brightness", 0.5), 0.08, 0.20, 0.85, 0.98)
    no_glare = 1.0 - _clip01(q.get("glare_frac", 0.0) / 0.15)
    sharp = 0.5 + 0.5 * _clip01(q.get("sharpness", 0.5) / 0.3)
    steady = 1.0 - _clip01((q.get("jitter", 0.0) - 0.3) / 0.5)
    steady_f = 0.5 + 0.5 * steady
    angle = max(abs(p.yaw), abs(p.pitch))
    angle_f = 1.0 - _clip01((angle - 45.0) / 30.0)
    return Reliability(
        eye=light * no_glare * sharp * steady_f,
        mouth=light * sharp * steady_f,
        head=angle_f * steady_f * light ** 0.5,
    )
