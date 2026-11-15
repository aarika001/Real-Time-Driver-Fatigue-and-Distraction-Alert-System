"""Deterministic image-level synthetic stress generators for proposal v4.

The generators operate on images so they can be used after extracting frames
from an approved course dataset. They do not collect or record new footage.
Requires Pillow and NumPy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


SEVERITIES = {
    "low_light": {
        "mild": {"gamma": 1.35, "noise_std": 2.0},
        "moderate": {"gamma": 1.9, "noise_std": 5.0},
        "severe": {"gamma": 2.7, "noise_std": 9.0},
    },
    "glare": {
        # radius_fraction and core_fraction are relative to the face-box width.
        # The core is saturated white; the halo uses the given opacity.
        "mild": {"opacity": 0.55, "radius_fraction": 0.15, "core_fraction": 0.30},
        "moderate": {"opacity": 0.80, "radius_fraction": 0.22, "core_fraction": 0.45},
        "severe": {"opacity": 0.95, "radius_fraction": 0.30, "core_fraction": 0.60},
    },
    "off_axis": {"mild": {"angle_deg": 10.0}, "moderate": {"angle_deg": 20.0}, "severe": {"angle_deg": 30.0}},
    "occlusion": {
        "mild": {"area_fraction": 0.10},
        "moderate": {"area_fraction": 0.25},
        "severe": {"area_fraction": 0.40},
    },
    "face_lost": {"mild": {}, "moderate": {}, "severe": {}},
}


def _severity(stress_type: str, severity: str) -> dict:
    try:
        return dict(SEVERITIES[stress_type][severity])
    except KeyError as exc:
        valid = ", ".join(f"{kind}:{level}" for kind, levels in SEVERITIES.items() for level in levels)
        raise ValueError(f"Unknown stress/severity. Valid combinations: {valid}") from exc


def _box_or_default(box: Sequence[int] | None, width: int, height: int) -> tuple[int, int, int, int]:
    if box is None:
        return (width // 4, height // 8, width // 2, height * 3 // 4)
    if len(box) != 4:
        raise ValueError("A box must be x,y,w,h")
    x, y, w, h = map(int, box)
    if w <= 0 or h <= 0:
        raise ValueError("Box width and height must be positive")
    return x, y, w, h


def _clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    left, top = max(0, x), max(0, y)
    right, bottom = min(width, x + w), min(height, y + h)
    if right <= left or bottom <= top:
        raise ValueError("Box does not intersect the image")
    return left, top, right - left, bottom - top


def low_light(image: Image.Image, severity: str, seed: int = 42) -> Image.Image:
    params = _severity("low_light", severity)
    rng = np.random.default_rng(seed)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    arr = np.power(arr, params["gamma"])
    arr += rng.normal(0.0, params["noise_std"] / 255.0, arr.shape)
    return Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255.0), "RGB")


def glare(image: Image.Image, severity: str, region: Sequence[int] | None = None, seed: int = 42) -> Image.Image:
    """Saturated glare blob placed on the eye band of the face box (lens flare or sun on the eyes).

    region is the face box x, y, w, h. The blob centre is drawn at random (seeded) between 30 and
    70 percent of the face width and 30 to 45 percent of the face height, which covers the eyes.
    """
    params = _severity("glare", severity)
    rng = np.random.default_rng(seed)
    out = image.convert("RGBA")
    width, height = out.size
    x, y, w, h = _clip_box(_box_or_default(region, width, height), width, height)
    radius = max(3, int(w * params["radius_fraction"]))
    core = max(2, int(radius * params["core_fraction"]))
    cx = int(x + w * rng.uniform(0.30, 0.70))
    cy = int(y + h * rng.uniform(0.30, 0.45))
    halo = Image.new("RGBA", out.size, (255, 255, 255, 0))
    ImageDraw.Draw(halo).ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                                 fill=(255, 255, 255, int(255 * params["opacity"])))
    halo = halo.filter(ImageFilter.GaussianBlur(max(1, radius // 3)))
    out = Image.alpha_composite(out, halo)
    centre = Image.new("RGBA", out.size, (255, 255, 255, 0))
    ImageDraw.Draw(centre).ellipse((cx - core, cy - core, cx + core, cy + core), fill=(255, 255, 255, 255))
    centre = centre.filter(ImageFilter.GaussianBlur(max(1, core // 6)))
    return Image.alpha_composite(out, centre).convert("RGB")


def _perspective_coefficients(source: Sequence[tuple[float, float]], target: Sequence[tuple[float, float]]) -> tuple[float, ...]:
    matrix = []
    values = []
    for (x, y), (u, v) in zip(source, target):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        values.append(u)
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        values.append(v)
    return tuple(np.linalg.solve(np.asarray(matrix, dtype=float), np.asarray(values, dtype=float)))


def off_axis(image: Image.Image, severity: str, direction: int = 1) -> Image.Image:
    params = _severity("off_axis", severity)
    width, height = image.size
    angle = float(params["angle_deg"]) * (1 if direction >= 0 else -1)
    shift = math.tan(math.radians(angle)) * width * 0.25
    source = [(0, 0), (width, 0), (width, height), (0, height)]
    target = [(shift, 0), (width + shift, 0), (width - shift, height), (-shift, height)]
    coefficients = _perspective_coefficients(target, source)
    return image.convert("RGB").transform(image.size, Image.Transform.PERSPECTIVE, coefficients, Image.Resampling.BICUBIC)


def occlusion(image: Image.Image, severity: str, kind: str = "mask", face_box: Sequence[int] | None = None) -> Image.Image:
    params = _severity("occlusion", severity)
    out = image.convert("RGB")
    width, height = out.size
    x, y, w, h = _clip_box(_box_or_default(face_box, width, height), width, height)
    draw = ImageDraw.Draw(out)
    patch_h = max(1, int(h * params["area_fraction"]))
    if kind == "mask":
        top = y + int(h * 0.55)
        draw.rectangle((x, top, x + w, min(y + h, top + patch_h)), fill=(80, 80, 80))
    elif kind == "cap_brim":
        top = y + int(h * 0.10)
        draw.rectangle((x, top, x + w, min(y + h, top + patch_h)), fill=(30, 30, 30))
    elif kind == "hand_patch":
        left = x + int(w * 0.45)
        top = y + int(h * 0.25)
        patch_w = max(1, int(w * params["area_fraction"]))
        draw.rounded_rectangle((left, top, min(x + w, left + patch_w), min(y + h, top + patch_h)), radius=8, fill=(150, 105, 80))
    else:
        raise ValueError("kind must be mask, cap_brim, or hand_patch")
    return out


def face_lost(image: Image.Image, face_box: Sequence[int] | None = None) -> Image.Image:
    out = image.convert("RGB")
    width, height = out.size
    x, y, w, h = _clip_box(_box_or_default(face_box, width, height), width, height)
    draw = ImageDraw.Draw(out)
    draw.rectangle((x, y, x + w, y + h), fill=(0, 0, 0))
    return out


def apply_stress(image: Image.Image, stress_type: str, severity: str, seed: int = 42, face_box: Sequence[int] | None = None, direction: int = 1, occlusion_kind: str = "mask") -> Image.Image:
    if stress_type == "low_light":
        return low_light(image, severity, seed)
    if stress_type == "glare":
        return glare(image, severity, face_box, seed)
    if stress_type == "off_axis":
        return off_axis(image, severity, direction)
    if stress_type == "occlusion":
        return occlusion(image, severity, occlusion_kind, face_box)
    if stress_type == "face_lost":
        return face_lost(image, face_box)
    raise ValueError(f"Unknown stress_type: {stress_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply one proposal-v4 synthetic stress test to an image")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stress", required=True, choices=sorted(SEVERITIES))
    parser.add_argument("--severity", required=True, choices=("mild", "moderate", "severe"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--face-box", nargs=4, type=int, default=None, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--occlusion-kind", choices=("mask", "cap_brim", "hand_patch"), default="mask")
    args = parser.parse_args()
    image = Image.open(args.input)
    output = apply_stress(image, args.stress, args.severity, args.seed, args.face_box, args.direction, args.occlusion_kind)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    print(json.dumps({"source": str(args.input), "output": str(args.output), "stress": args.stress, "severity": args.severity, "seed": args.seed}))


if __name__ == "__main__":
    main()
