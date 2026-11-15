"""Domain-gap augmentations for eye crops (R4, Milestone 3).

Each transform imitates a real in-cabin degradation from the proposal (L1):
    gamma / brightness   -> night, dim cabin, IR vs RGB exposure differences
    sensor noise         -> low-light webcam noise
    blur (gaussian/motion) -> defocus, head motion
    glare patches        -> spectacle and lens reflections (saturated blobs)
    occlusion            -> hair, frames, hand partly covering the eye
    affine + flip        -> different crop framing between datasets, left/right eye

Order matters: degradations happen in the camera, BEFORE our CLAHE contrast
normalisation. So these functions run on the raw resized 64x64 crop and
CLAHE is applied afterwards (see ``eye_dataset.EyeCropDataset``).

Pure NumPy/OpenCV, so it runs anywhere and is unit-tested without PyTorch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Optional

import cv2
import numpy as np


@dataclass
class AugmentConfig:
    p_affine: float = 0.7     # scale/rotate/shift
    max_rotate_deg: float = 15.0
    scale_range: tuple = (0.85, 1.15)
    max_shift_px: float = 4.0
    p_flip: float = 0.5
    p_gamma: float = 0.6
    gamma_range: tuple = (0.5, 2.0)
    p_contrast: float = 0.4
    p_noise: float = 0.3
    noise_std_range: tuple = (2.0, 12.0)
    p_blur: float = 0.3
    p_glare: float = 0.3
    p_occlusion: float = 0.25

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "AugmentConfig":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"Unknown augment config keys: {sorted(unknown)}")
        for k in ("scale_range", "gamma_range", "noise_std_range"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def random_affine(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    h, w = img.shape
    angle = rng.uniform(-cfg.max_rotate_deg, cfg.max_rotate_deg)
    scale = rng.uniform(*cfg.scale_range)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[:, 2] += rng.uniform(-cfg.max_shift_px, cfg.max_shift_px, size=2)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def random_gamma(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    # sample log-uniformly so darkening and brightening are equally likely
    g = float(np.exp(rng.uniform(np.log(cfg.gamma_range[0]), np.log(cfg.gamma_range[1]))))
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def random_contrast(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    alpha = rng.uniform(0.6, 1.4)     # contrast
    beta = rng.uniform(-30, 30)       # brightness
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def random_noise(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    std = rng.uniform(*cfg.noise_std_range)
    return np.clip(img.astype(np.float32) + rng.normal(0, std, img.shape), 0, 255).astype(np.uint8)


def random_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        k = int(rng.choice([3, 5]))
        return cv2.GaussianBlur(img, (k, k), 0)
    # motion blur: a short line kernel at a random angle
    k = int(rng.choice([3, 5, 7]))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def random_glare(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """1-3 soft saturated ellipses, like reflections on spectacles."""
    h, w = img.shape
    mask = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(1, 4))):
        center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        axes = (int(rng.integers(2, max(3, w // 5))), int(rng.integers(2, max(3, h // 6))))
        cv2.ellipse(mask, center, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    level = rng.uniform(230, 255)
    out = img.astype(np.float32) * (1 - mask) + level * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def random_occlusion(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Erase one rectangle (5-22% of the area) with a flat or noisy fill.

    Kept below ~a quarter of the crop so the eye usually stays partly visible;
    covering it completely would make the label meaningless (label noise).
    """
    h, w = img.shape
    area = rng.uniform(0.05, 0.22) * h * w
    aspect = np.exp(rng.uniform(np.log(0.4), np.log(2.5)))
    eh = int(min(h, max(2, round(np.sqrt(area / aspect)))))
    ew = int(min(w, max(2, round(np.sqrt(area * aspect)))))
    y, x = int(rng.integers(0, h - eh + 1)), int(rng.integers(0, w - ew + 1))
    out = img.copy()
    if rng.random() < 0.5:
        out[y:y + eh, x:x + ew] = int(rng.integers(0, 256))
    else:
        out[y:y + eh, x:x + ew] = rng.integers(0, 256, size=(eh, ew), dtype=np.uint8)
    return out


def augment_eye(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Apply the augmentation chain to one raw (pre-CLAHE) uint8 grayscale crop."""
    if img.dtype != np.uint8 or img.ndim != 2:
        raise ValueError("augment_eye expects a 2-D uint8 image")
    out = img
    if rng.random() < cfg.p_affine:
        out = random_affine(out, rng, cfg)
    if rng.random() < cfg.p_flip:
        out = np.ascontiguousarray(out[:, ::-1])
    if rng.random() < cfg.p_gamma:
        out = random_gamma(out, rng, cfg)
    if rng.random() < cfg.p_contrast:
        out = random_contrast(out, rng)
    if rng.random() < cfg.p_blur:
        out = random_blur(out, rng)
    if rng.random() < cfg.p_glare:
        out = random_glare(out, rng)
    if rng.random() < cfg.p_noise:
        out = random_noise(out, rng, cfg)
    if rng.random() < cfg.p_occlusion:
        out = random_occlusion(out, rng)
    return out
