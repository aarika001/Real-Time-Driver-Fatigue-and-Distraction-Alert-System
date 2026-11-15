"""Eye-region cropping, contrast normalisation and quality signals.

This reproduces R3's RLDD preprocessing (``RLDD-preprocess/preprocess-rldd.py``,
``extract_eye_crop_and_quality``) exactly, so that crops seen by the eye CNN
during training, offline evaluation and the live webcam loop are identical.
``tests/r4/test_contract.py`` checks this parity.

Pipeline for one eye:
    landmark box -> square, padded by 35% -> clip to frame -> quality signals
    on the raw crop -> resize to 64x64 (INTER_AREA) -> CLAHE(2.0, 4x4)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CROP_SIZE: int = 64
PAD_RATIO: float = 0.35
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID: tuple[int, int] = (4, 4)
GLARE_THRESHOLD: int = 245

_CLAHE = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)


@dataclass(frozen=True)
class EyeQuality:
    """Raw-crop quality signals consumed by R7's reliability estimator.

    brightness: mean intensity / 255, in [0, 1].
    contrast: intensity std / 255, in [0, 0.5].
    glare_frac: fraction of pixels >= 245 (saturated), in [0, 1].
    sharpness: variance of the Laplacian (unbounded; higher = sharper).
    """

    brightness: float
    contrast: float
    glare_frac: float
    sharpness: float

    def as_list(self) -> list[float]:
        return [self.brightness, self.contrast, self.glare_frac, self.sharpness]


# Values R3 writes when a crop is empty; kept identical for compatibility.
EMPTY_QUALITY = EyeQuality(brightness=0.0, contrast=0.0, glare_frac=1.0, sharpness=0.0)


def normalize_crop(gray_crop: np.ndarray) -> np.ndarray:
    """Resize any grayscale eye crop to 64x64 and apply CLAHE.

    Use this for dataset crops that are already cut out (e.g. MRL Eye), so they
    go through the same normalisation as crops cut from video frames.
    """
    if gray_crop.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale crop, got shape {gray_crop.shape}")
    resized = cv2.resize(gray_crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
    return _CLAHE.apply(resized)


def eye_box(eye_pts_xy: np.ndarray, frame_w: int, frame_h: int,
            pad_ratio: float = PAD_RATIO) -> tuple[int, int, int, int]:
    """Square, padded bounding box (x1, y1, x2, y2) around eye landmarks, clipped to the frame."""
    x_min, y_min = np.min(eye_pts_xy, axis=0)
    x_max, y_max = np.max(eye_pts_xy, axis=0)
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    half = max(x_max - x_min, y_max - y_min) * (0.5 + pad_ratio)
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(frame_w, int(cx + half))
    y2 = min(frame_h, int(cy + half))
    return x1, y1, x2, y2


def crop_eye(gray_frame: np.ndarray, eye_pts_xy: np.ndarray,
             pad_ratio: float = PAD_RATIO) -> tuple[np.ndarray, EyeQuality, bool]:
    """Cut one eye out of a grayscale frame.

    Args:
        gray_frame: (H, W) uint8 grayscale frame.
        eye_pts_xy: (K, 2) pixel coordinates of that eye's landmarks
            (rows of ``landmarks_to_array(...)[EYE_IMG_LEFT, :2]``).
        pad_ratio: padding around the landmark box (0.35 = R3's setting).

    Returns:
        crop: (64, 64) uint8, CLAHE-normalised (all zeros if the box is empty).
        quality: ``EyeQuality`` computed on the raw crop before normalisation.
        valid: False if the box fell outside the frame (crop is then zeros).
    """
    h, w = gray_frame.shape[:2]
    x1, y1, x2, y2 = eye_box(eye_pts_xy, w, h, pad_ratio)
    raw = gray_frame[y1:y2, x1:x2]
    if raw.size == 0:
        return np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8), EMPTY_QUALITY, False

    quality = EyeQuality(
        brightness=float(np.mean(raw) / 255.0),
        contrast=float(np.std(raw) / 255.0),
        glare_frac=float(np.mean(raw >= GLARE_THRESHOLD)),
        sharpness=float(cv2.Laplacian(raw, cv2.CV_64F).var()),
    )
    return normalize_crop(raw), quality, True
