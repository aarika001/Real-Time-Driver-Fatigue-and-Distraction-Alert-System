"""R4 output contract: what the eye and mouth cues hand to R3, R6, R7 and R8.

One ``R4FrameOutput`` is produced per processed frame (15 FPS). Downstream code
should normally use ``to_flat_dict()``, whose keys are listed in
``FRAME_FEATURE_KEYS`` in a fixed order (R6 can stack them into the GRU input).

Missing values are NaN, never 0. A MAR of 0 would look like a closed mouth and
a P(closed) of 0 like a wide-open eye; NaN forces consumers to check
``face_detected`` / validity instead of silently learning from fake values.
Note: R3's processed RLDD archives store 0 for lost frames, so mask those with
their ``face_detected`` array.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np

from .eye_crop import EyeQuality

NAN = float("nan")

# Fixed, ordered flat schema. Do not reorder: R6's GRU depends on it.
FRAME_FEATURE_KEYS: tuple[str, ...] = (
    "face_detected",     # 1.0 / 0.0
    "p_closed_left",     # P(eye closed), image-left eye, [0, 1]
    "p_closed_right",    # P(eye closed), image-right eye, [0, 1]
    "p_closed",          # mean over valid eyes
    "ear",               # mean six-point eye aspect ratio (for baselines / R7)
    "mar",               # outer-lip MAR, same formula as R3's RLDD archives
    "mar_inner",         # inner-lip MAR (actual opening)
    "mar_norm",          # mar / driver's calibrated neutral MAR (NaN before calibration)
    "mouth_open_run_s",  # seconds the mouth has been continuously wide open
    "mouth_osc_hz",      # open/close cycles per second over the last 2 s (talking is high)
    "yawn_flag",         # 1.0 while the rule-based yawn detector fires, else 0.0
    "eye_brightness",    # quality signals, mean over valid eyes (R3-compatible)
    "eye_contrast",
    "eye_glare_frac",
    "eye_sharpness",
)


@dataclass
class EyeResult:
    p_closed_left: float = NAN
    p_closed_right: float = NAN
    valid_left: bool = False
    valid_right: bool = False
    ear: float = NAN
    quality_left: Optional[EyeQuality] = None
    quality_right: Optional[EyeQuality] = None

    @property
    def p_closed(self) -> float:
        vals = [p for p, ok in ((self.p_closed_left, self.valid_left),
                                (self.p_closed_right, self.valid_right)) if ok]
        return float(np.mean(vals)) if vals else NAN

    def mean_quality(self) -> list[float]:
        qs = [q.as_list() for q, ok in ((self.quality_left, self.valid_left),
                                        (self.quality_right, self.valid_right))
              if ok and q is not None]
        return np.mean(qs, axis=0).tolist() if qs else [NAN] * 4


@dataclass
class MouthResult:
    mar: float = NAN
    mar_inner: float = NAN
    mar_norm: float = NAN
    valid: bool = False
    # Temporal yawn features (NaN until the driver is calibrated).
    open_run_s: float = NAN
    osc_hz: float = NAN
    yawn_flag: float = NAN


@dataclass
class R4FrameOutput:
    timestamp_s: float
    face_detected: bool
    eye: EyeResult = field(default_factory=EyeResult)
    mouth: MouthResult = field(default_factory=MouthResult)

    def to_flat_dict(self) -> dict[str, float]:
        """Flat {key: float} with keys exactly ``FRAME_FEATURE_KEYS``."""
        b, c, g, s = self.eye.mean_quality()
        out = {
            "face_detected": 1.0 if self.face_detected else 0.0,
            "p_closed_left": self.eye.p_closed_left if self.eye.valid_left else NAN,
            "p_closed_right": self.eye.p_closed_right if self.eye.valid_right else NAN,
            "p_closed": self.eye.p_closed,
            "ear": self.eye.ear,
            "mar": self.mouth.mar if self.mouth.valid else NAN,
            "mar_inner": self.mouth.mar_inner if self.mouth.valid else NAN,
            "mar_norm": self.mouth.mar_norm if self.mouth.valid else NAN,
            # Temporal features are kept even on face-lost frames: they carry
            # state (e.g. a yawn in progress while a hand covers the mouth).
            "mouth_open_run_s": self.mouth.open_run_s,
            "mouth_osc_hz": self.mouth.osc_hz,
            "yawn_flag": self.mouth.yawn_flag,
            "eye_brightness": b, "eye_contrast": c,
            "eye_glare_frac": g, "eye_sharpness": s,
        }
        assert tuple(out) == FRAME_FEATURE_KEYS
        return {k: float(v) for k, v in out.items()}

    def to_vector(self) -> np.ndarray:
        """(len(FRAME_FEATURE_KEYS),) float32 vector in schema order."""
        return np.array(list(self.to_flat_dict().values()), dtype=np.float32)


def is_missing(x: float) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


# ---------------------------------------------------------------------------
# Interfaces the concrete classes (Milestones 2, 4, 6) must implement.
# They are typing.Protocols: signatures only, checked by type checkers.
# ---------------------------------------------------------------------------
class EyeStatePredictorAPI(Protocol):
    def predict_crops(self, crops: np.ndarray) -> np.ndarray:
        """crops: (B, 64, 64) uint8, CLAHE-normalised (see eye_crop.crop_eye).
        Returns (B,) float32 calibrated P(eye closed)."""
        ...


class MouthCueExtractorAPI(Protocol):
    def update(self, landmarks_px: Optional[np.ndarray]) -> MouthResult:
        """Process one frame (call at a steady 15 FPS).
        landmarks_px: (478, 3) from landmarks.landmarks_to_array, or None if
        no face was found. Returns MAR values and temporal yawn features."""
        ...

    def calibrate(self, neutral_mar_values: Sequence[float]) -> float:
        """Set the per-driver neutral MAR from ~30 s of calibration frames
        (called by R6's calibration stage). Returns the baseline used."""
        ...
