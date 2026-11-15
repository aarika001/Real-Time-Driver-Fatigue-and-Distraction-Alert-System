"""Milestone 1 contract tests for the R4 cues package.

Run from the repo root:  python -m pytest tests/r4 -v
"""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cues import landmarks as L
from cues.eye_crop import CROP_SIZE, EMPTY_QUALITY, crop_eye, normalize_crop
from cues.schemas import FRAME_FEATURE_KEYS, EyeResult, MouthResult, R4FrameOutput

REPO_ROOT = Path(__file__).resolve().parents[2]
R3_SCRIPT = REPO_ROOT / "RLDD-preprocess" / "preprocess-rldd.py"


def _fake_landmarks(seed: int = 0, w: int = 640, h: int = 480):
    """478 normalised landmarks with eyes/mouth placed at plausible positions."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.3, 0.7, size=(L.NUM_LANDMARKS, 3)).astype(np.float32)
    pts[:, 2] = rng.uniform(-0.05, 0.05, size=L.NUM_LANDMARKS)
    for idxs, (cx, cy) in ((L.EYE_IMG_LEFT, (0.40, 0.40)), (L.EYE_IMG_RIGHT, (0.60, 0.40))):
        pts[list(idxs), 0] = cx + rng.uniform(-0.03, 0.03, len(idxs))
        pts[list(idxs), 1] = cy + rng.uniform(-0.01, 0.01, len(idxs))
    return [SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in pts]


def test_indices_valid_and_consistent():
    for name in ("EYE_IMG_LEFT", "EYE_IMG_RIGHT", "EAR_IMG_LEFT", "EAR_IMG_RIGHT",
                 "MOUTH_OUTER", "MOUTH_INNER", "INTEROCULAR"):
        idxs = getattr(L, name)
        assert all(0 <= i < L.NUM_LANDMARKS for i in idxs), name
        assert len(set(idxs)) == len(idxs), f"duplicate index in {name}"
    # EAR points are a subset of the cropping contour
    assert set(L.EAR_IMG_LEFT) <= set(L.EYE_IMG_LEFT)
    assert set(L.EAR_IMG_RIGHT) <= set(L.EYE_IMG_RIGHT)


def test_landmarks_to_array_scales_to_pixels():
    lms = _fake_landmarks()
    arr = L.landmarks_to_array(lms, 640, 480)
    assert arr.shape == (478, 3) and arr.dtype == np.float32
    assert math.isclose(arr[0, 0], lms[0].x * 640, rel_tol=1e-5)
    assert math.isclose(arr[0, 1], lms[0].y * 480, rel_tol=1e-5)
    with pytest.raises(ValueError):
        L.landmarks_to_array(lms[:100], 640, 480)


def test_crop_eye_shapes_and_quality_ranges():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, size=(480, 640), dtype=np.uint8)
    arr = L.landmarks_to_array(_fake_landmarks(), 640, 480)
    crop, q, valid = crop_eye(frame, arr[list(L.EYE_IMG_LEFT), :2])
    assert valid and crop.shape == (CROP_SIZE, CROP_SIZE) and crop.dtype == np.uint8
    assert 0 <= q.brightness <= 1 and 0 <= q.contrast <= 0.5 and 0 <= q.glare_frac <= 1
    assert q.sharpness >= 0


def test_crop_eye_outside_frame_is_invalid():
    frame = np.zeros((100, 100), dtype=np.uint8)
    pts = np.array([[500.0, 500.0], [520.0, 505.0]])
    crop, q, valid = crop_eye(frame, pts)
    assert not valid and q == EMPTY_QUALITY and not crop.any()


def test_normalize_crop_accepts_any_size():
    for shape in ((30, 50), (86, 86), (120, 90)):
        out = normalize_crop(np.full(shape, 128, dtype=np.uint8))
        assert out.shape == (CROP_SIZE, CROP_SIZE)


def test_schema_keys_order_and_nan_for_missing():
    out = R4FrameOutput(timestamp_s=0.0, face_detected=False)
    d = out.to_flat_dict()
    assert tuple(d) == FRAME_FEATURE_KEYS
    assert d["face_detected"] == 0.0
    assert all(math.isnan(v) for k, v in d.items() if k != "face_detected")
    assert out.to_vector().shape == (len(FRAME_FEATURE_KEYS),)


def test_p_closed_averages_only_valid_eyes():
    eye = EyeResult(p_closed_left=0.9, p_closed_right=0.1, valid_left=True, valid_right=False)
    mouth = MouthResult(mar=0.3, mar_inner=0.1, valid=True)  # not calibrated yet
    d = R4FrameOutput(0.0, True, eye=eye, mouth=mouth).to_flat_dict()
    assert d["p_closed"] == pytest.approx(0.9)
    assert math.isnan(d["p_closed_right"])
    assert d["mar"] == pytest.approx(0.3) and math.isnan(d["mar_norm"])


def _load_r3_module():
    pytest.importorskip("mediapipe")
    pytest.importorskip("pandas")
    if "tqdm" not in sys.modules and importlib.util.find_spec("tqdm") is None:
        sys.modules["tqdm"] = types.SimpleNamespace(tqdm=lambda it, **kw: it)  # R3 only uses it for progress bars
    if not R3_SCRIPT.exists():
        pytest.skip("R3 preprocessing script not present")
    spec = importlib.util.spec_from_file_location("r3_preprocess", R3_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parity_with_r3_rldd_preprocessing():
    """Our crops/quality/indices must equal what R3 stored in processed_rldd."""
    r3 = _load_r3_module()
    assert tuple(r3.LEFT_EYE_IDXS) == L.EYE_IMG_LEFT
    assert tuple(r3.RIGHT_EYE_IDXS) == L.EYE_IMG_RIGHT
    assert tuple(r3.MOUTH_IDXS) == L.MOUTH_OUTER

    rng = np.random.default_rng(2)
    frame = rng.integers(0, 256, size=(480, 640), dtype=np.uint8)
    for seed in range(5):
        lms = _fake_landmarks(seed)
        arr = L.landmarks_to_array(lms, 640, 480)
        for idxs in (L.EYE_IMG_LEFT, L.EYE_IMG_RIGHT):
            r3_crop, r3_q = r3.extract_eye_crop_and_quality(frame, lms, list(idxs), 640, 480)
            crop, q, _ = crop_eye(frame, arr[list(idxs), :2])
            np.testing.assert_array_equal(crop, r3_crop)
            np.testing.assert_allclose(q.as_list(), r3_q, rtol=1e-6)
