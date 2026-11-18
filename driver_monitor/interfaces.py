"""Data contracts between modules. Keep these stable: teammates plug in against them."""
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

FEATURE_NAMES = ("p_eye_closed", "ear_rel", "mar_rel", "yaw_rel", "pitch_rel",
                 "r_eye", "r_mouth", "r_head")


@dataclass
class Percept:
    """Raw per-frame output of perception (R4/R5 own the content)."""
    t: float
    face_found: bool
    p_eye_closed: float = 0.0
    ear: float = 0.0
    mar: float = 0.0
    yaw: float = 0.0            # degrees, raw solvePnP
    pitch: float = 0.0
    quality: dict = field(default_factory=dict)
    landmarks_px: Optional[np.ndarray] = None
    eye_crops: Optional[Tuple] = None
    roll: float = 0.0
    # Raw R3/R4 eye quality [brightness, contrast(std/255), glare_frac, laplacian_var]
    raw_quality: Optional[Tuple[float, float, float, float]] = None


@dataclass
class Features:
    """Per-driver-calibrated cues (calibration module output)."""
    p_eye_closed: float = 0.0
    ear_rel: float = 1.0
    mar_rel: float = 0.0
    yaw_rel: float = 0.0
    pitch_rel: float = 0.0
    face_found: bool = True
    gru_row: Optional[np.ndarray] = None        # (13,) float32 from temporal_model/features.py


@dataclass
class Reliability:
    """Probability each cue is currently trustworthy, in [0, 1] (R7 owns it)."""
    eye: float = 0.0
    mouth: float = 0.0
    head: float = 0.0

    def best(self) -> float:
        return max(self.eye, self.mouth, self.head)


@dataclass
class TemporalOut:
    """P(alert, fatigue, distraction) in config.STATES order (R6 owns it)."""
    probs: np.ndarray
    debug: dict = field(default_factory=dict)


@dataclass
class Decision:
    """Output of the utility policy (R7 owns it)."""
    action_idx: int = 0
    action: str = "none"
    dominant: str = "-"
    degraded: bool = False
    eu: dict = field(default_factory=dict)
    n_recent_alerts: int = 0