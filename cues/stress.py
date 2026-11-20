"""Graded synthetic stress tests on eye crops (R4 evaluation, Section 3.3).

Each factor has severity levels 0 (clean) to 4 (severe) and is applied to the
raw crop BEFORE CLAHE, exactly where the real degradation would happen.
Unlike training augmentation these are deterministic for a given (seed,
image), so every model and baseline sees the same degraded images.

They operate on crops only; R2's generators degrade whole frames (which also
affects landmark detection). Results here isolate the eye CNN itself.
"""
from __future__ import annotations

import cv2
import numpy as np

LEVELS = (0, 1, 2, 3, 4)
FACTORS = ("low_light", "glare", "blur", "occlusion", "noise")


def degrade(img: np.ndarray, factor: str, level: int, rng: np.random.Generator) -> np.ndarray:
    """Apply one factor at one severity level to a raw uint8 grayscale crop."""
    if level == 0:
        return img
    if not 1 <= level <= 4:
        raise ValueError("level must be 0-4")
    h, w = img.shape
    x = img.astype(np.float32)
    if factor == "low_light":
        # darker (gamma) + sensor noise, as in a dim cabin at night
        gamma = (1.0, 1.6, 2.2, 2.8, 3.5)[level]
        x = 255.0 * (x / 255.0) ** gamma
        x = x + rng.normal(0, 2.0 * level, x.shape)
    elif factor == "glare":
        mask = np.zeros((h, w), np.float32)
        n_blobs, radius = level, int(3 + 2.5 * level)
        for _ in range(n_blobs):
            c = (int(rng.integers(w // 4, 3 * w // 4)), int(rng.integers(h // 4, 3 * h // 4)))
            cv2.circle(mask, c, radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        x = x * (1 - mask) + 250.0 * mask
    elif factor == "blur":
        k = (1, 3, 5, 7, 9)[level]
        x = cv2.GaussianBlur(x, (k, k), 0)
    elif factor == "occlusion":
        frac = (0.0, 0.1, 0.2, 0.3, 0.4)[level]           # fraction of the crop width covered
        ew = max(1, int(round(frac * w)))
        x0 = int(rng.integers(0, w - ew + 1))
        x[:, x0:x0 + ew] = float(rng.integers(0, 60))       # dark vertical bar, e.g. a hand or frame
    elif factor == "noise":
        x = x + rng.normal(0, (0, 6, 12, 20, 30)[level], x.shape)
    else:
        raise ValueError(f"unknown factor {factor!r}; choose from {FACTORS}")
    return np.clip(x, 0, 255).astype(np.uint8)


def degrade_batch(raw: np.ndarray, factor: str, level: int, seed: int = 0) -> np.ndarray:
    """Degrade every crop in (N, H, W) deterministically (per-image seeds)."""
    return np.stack([degrade(im, factor, level, np.random.default_rng((seed, i))) for i, im in enumerate(raw)])
