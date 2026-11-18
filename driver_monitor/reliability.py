"""Reliability estimator adapter (wraps R7's CueReliabilityEstimator)."""
from R7_Jayesh_Contribution.reliability.estimator import CueReliabilityEstimator
from R7_Jayesh_Contribution.reliability.quality_features import QualitySignals
from interfaces import Percept, Reliability

estimator = CueReliabilityEstimator()


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
    signals = QualitySignals(
        timestamp_s=p.t,
        face_detected=True,
        eye_brightness=q.get("brightness", 0.5) * 255.0,
        eye_contrast=q.get("contrast", 0.5) * 80.0,
        eye_glare_frac=q.get("glare_frac", 0.0),
        eye_sharpness=q.get("sharpness", 0.5) * 500.0,
        head_angle_z=max(abs(p.yaw), abs(p.pitch)) / 15.0,
        landmark_jitter=q.get("jitter", 0.0),
        detector_confidence=q.get("det_conf", 1.0),
    )

    res = estimator.predict_all(signals)

    # Physical illumination gate (R7's mouth/head vectors omit brightness,
    # and R7's untrained fallback weights stay > 0.85 at brightness = 0).
    light = _trap(q.get("brightness", 0.5), 0.08, 0.20, 0.85, 0.98)
    no_glare = 1.0 - _clip01(q.get("glare_frac", 0.0) / 0.15)
    eye_gate = light * no_glare if estimator.models["eye"] is None else light

    return Reliability(
        eye=float(res["eye"] * eye_gate),
        mouth=float(res["mouth"] * light),
        head=float(res["head_pose"] * light),
    )