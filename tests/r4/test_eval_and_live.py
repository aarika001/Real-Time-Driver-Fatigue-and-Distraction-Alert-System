"""Milestones 5-6 tests: PERCLOS / Baseline B2, stress tests, evaluation script,
the live R4 pipeline and the webcam script (headless, on a synthetic video).

A fake eye predictor stands in for the trained CNN, so these run without
PyTorch; the ONNX export test runs only where torch + onnxruntime exist.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from cues import landmarks as L
from cues.inference_r4 import R4Pipeline, eye_aspect_ratio
from cues.perclos import b2_alerts, perclos
from cues.schemas import FRAME_FEATURE_KEYS
from cues.stress import FACTORS, degrade, degrade_batch
from tests.r4.test_eye_pipeline import fake_eye, make_fake_mrl
from tests.r4.test_mouth_cue import mouth_landmarks

REPO_ROOT = Path(__file__).resolve().parents[2]


class FakePredictor:
    """Low-texture crops (our synthetic closed eyes) -> high P(closed)."""

    def __init__(self):
        self.calls = []

    def predict_crops(self, crops):
        crops = np.asarray(crops)
        self.calls.append(len(crops))
        std = crops.reshape(len(crops), -1).std(axis=1)
        return (1.0 / (1.0 + np.exp((std - 45.0) / 6.0))).astype(np.float32)


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "r4" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fake_predictor_separates_synthetic_eyes():
    from cues.eye_crop import normalize_crop
    rng = np.random.default_rng(0)
    crops = np.stack([normalize_crop(fake_eye(c, rng, 64)) for c in (True, False, True, False)])
    p = FakePredictor().predict_crops(crops)
    assert p[0] > 0.5 > p[1] and p[2] > 0.5 > p[3]


# --- PERCLOS / B2 -------------------------------------------------------------
def test_perclos_running_fraction_and_nan_handling():
    fps = 10
    p = np.r_[np.zeros(50), np.ones(50)]              # 5 s open then 5 s closed
    pc = perclos(p, fps, window_s=5)
    assert pc[49] == 0.0 and pc[-1] == 1.0 and pc[74] == pytest.approx(0.5, abs=0.05)
    p2 = np.full(100, np.nan)
    p2[:10] = 1.0
    assert np.isnan(perclos(p2, fps, window_s=5)[-1])   # too few valid frames -> no value
    assert b2_alerts(np.array([0.1, 0.2, np.nan]), 0.15).tolist() == [0.0, 1.0, 0.0]


# --- stress tests ---------------------------------------------------------------
def test_stress_levels_are_deterministic_and_graded():
    img = fake_eye(False, np.random.default_rng(0), 64)
    for f in FACTORS:
        a = degrade_batch(img[None], f, 3, seed=1)
        np.testing.assert_array_equal(a, degrade_batch(img[None], f, 3, seed=1))
        assert np.array_equal(degrade(img, f, 0, np.random.default_rng(0)), img)
    dark = [degrade(img, "low_light", lv, np.random.default_rng(0)).mean() for lv in range(5)]
    assert dark == sorted(dark, reverse=True)
    with pytest.raises(ValueError):
        degrade(img, "fog", 1, np.random.default_rng(0))


# --- evaluation script ------------------------------------------------------------
def _fake_rldd(root: Path, n_subjects: int = 6, frames: int = 300) -> Path:
    rng = np.random.default_rng(0)
    from cues.eye_crop import normalize_crop
    open_c = normalize_crop(fake_eye(False, rng, 64))
    closed_c = normalize_crop(fake_eye(True, rng, 64))
    for s in range(1, n_subjects + 1):
        d = root / "Fold1" / f"subject_{s:02d}"
        d.mkdir(parents=True)
        for kss, label, frac_closed in (("0", "alert", 0.05), ("10", "fatigue", 0.4)):
            closed = rng.random(frames) < frac_closed + 0.02 * s
            crops = np.where(closed[:, None, None], closed_c, open_c).astype(np.uint8)
            det = np.ones(frames, np.uint8)
            det[:10] = 0
            np.savez_compressed(d / f"kss_{kss}_{label}.npz", left_eye_crops=crops, right_eye_crops=crops,
                                face_detected=det, mar=np.zeros(frames, np.float32))
    return root


def test_eval_script_end_to_end_with_fake_predictor(tmp_path):
    base = make_fake_mrl(tmp_path / "mrl", n_subjects=10, per_subject=16)
    from cues.eye_dataset import build_mrl_index, split_by_subject
    df = build_mrl_index(base)
    df["split"] = split_by_subject(df, 2, 3, 0)
    df["path"] = "/kaggle/input/somewhere/" + df["path"].str.split("/").str[-1]   # paths from another machine
    df.to_csv(tmp_path / "splits.csv", index=False)
    rldd = _fake_rldd(tmp_path / "rldd")
    out = tmp_path / "eval"
    ev = _script("eval_eye_cnn")
    rc = ev.main(["--splits", str(tmp_path / "splits.csv"), "--mrl_root", str(base), "--rldd_dir", str(rldd),
                  "--rldd_stride", "2", "--stress_images", "40", "--out_dir", str(out)], predictor=FakePredictor())
    assert rc == 0
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "MRL Eye, unseen test people" in text and "Stress tests" in text and "Baseline B2" in text
    summ = pd.read_csv(out / "mrl_test_summary.csv")
    assert summ.iloc[0]["accuracy"] > 0.9
    sweep = pd.read_csv(out / "stress_sweep.csv")
    assert set(sweep["factor"]) == set(FACTORS) and len(sweep) == 25
    pv = pd.read_csv(out / "rldd_per_video.csv")
    assert (pv.groupby("label")["mean_perclos"].mean()["fatigue"]
            > pv.groupby("label")["mean_perclos"].mean()["alert"])
    b2 = ev.b2_video_level(pv)
    assert b2["auroc"] == 1.0 and b2["within_person_fatigue_higher"] == 1.0


# --- live pipeline -------------------------------------------------------------------
class StubLandmarker:
    """Returns a synthetic face: eyes from fixed boxes, mouth from mouth_landmarks."""

    def __init__(self, mouth_open_seq):
        self.seq = list(mouth_open_seq)
        self.i = 0

    def detect(self, rgb, ts_ms):
        open_px = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        if open_px is None:
            return None
        pts = mouth_landmarks(open_px, cx=160, cy=170)
        for idxs, (cx, cy) in ((L.EYE_IMG_LEFT, (110, 90)), (L.EYE_IMG_RIGHT, (210, 90))):
            ring = [(-20, 0), (20, 0), (-8, -7), (0, -8), (8, -7), (-8, 7), (0, 8), (8, 7)]
            for k, i in enumerate(idxs):
                pts[i, :2] = (cx + ring[k][0], cy + ring[k][1])
        return pts


def _face_frame(closed: bool) -> np.ndarray:
    frame = np.full((240, 320, 3), 120, np.uint8)
    rng = np.random.default_rng(0)
    for cx in (110, 210):
        eye = cv2.resize(fake_eye(closed, rng, 86), (68, 68))      # fills the 68-px crop box
        frame[56:124, cx - 34:cx + 34] = eye[..., None]
    return frame


def test_pipeline_outputs_calibration_and_batching():
    seq = [0.0] * 40 + [40.0] * 30 + [None]
    pred = FakePredictor()
    pipe = R4Pipeline(eye_predictor=pred, landmarker=StubLandmarker(seq))
    pipe.start_calibration()
    for k in range(40):
        res = pipe.process_frame(_face_frame(False), k / 15)
    neutral = pipe.finish_calibration()
    assert neutral > 0 and not pipe.calibrating
    d = res.output.to_flat_dict()
    assert tuple(d) == FRAME_FEATURE_KEYS and d["face_detected"] == 1.0
    assert d["p_closed"] < 0.5 and np.isfinite(d["ear"]) and np.isfinite(d["eye_glare_frac"])
    assert set(res.timings_ms) >= {"landmarks", "crops", "eye_cnn", "mouth", "total"}
    assert all(n == 2 for n in pred.calls)                      # left + right eye in one batch
    for k in range(30):
        res = pipe.process_frame(_face_frame(True), (40 + k) / 15)
    d = res.output.to_flat_dict()
    assert d["p_closed"] > 0.5 and d["yawn_flag"] == 1.0 and d["mar_norm"] > 2
    lost = pipe.process_frame(_face_frame(True), 71 / 15).output.to_flat_dict()
    assert lost["face_detected"] == 0.0 and np.isnan(lost["p_closed"])


def test_pipeline_without_eye_model_still_gives_quality_signals():
    pipe = R4Pipeline(eye_predictor=None, landmarker=StubLandmarker([0.0]))
    d = pipe.process_frame(_face_frame(False), 0.0).output.to_flat_dict()
    assert np.isnan(d["p_closed"]) and np.isfinite(d["eye_brightness"]) and np.isfinite(d["mar"])


def test_eye_aspect_ratio_open_vs_closed():
    pts = np.zeros((L.NUM_LANDMARKS, 3), np.float32)
    p1, p2, p3, p4, p5, p6 = L.EAR_IMG_LEFT
    pts[[p1, p4], :2] = [(0, 0), (30, 0)]
    pts[[p2, p3], :2] = [(10, -6), (20, -6)]
    pts[[p6, p5], :2] = [(10, 6), (20, 6)]
    open_ear = eye_aspect_ratio(pts, L.EAR_IMG_LEFT)
    pts[[p2, p3, p6, p5], 1] = [-1, -1, 1, 1]
    assert open_ear == pytest.approx(0.4) and eye_aspect_ratio(pts, L.EAR_IMG_LEFT) < 0.1


def test_webcam_script_headless_on_video(tmp_path):
    pytest.importorskip("mediapipe")
    if not (REPO_ROOT / "face_landmarker.task").exists():
        pytest.skip("face_landmarker.task missing")
    path = tmp_path / "clip.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (320, 240))
    if not vw.isOpened():
        pytest.skip("cannot write video here")
    for i in range(20):
        vw.write(_face_frame(i % 2 == 0))
    vw.release()
    report = tmp_path / "fps.json"
    rc = _script("webcam_demo").main(["--video", str(path), "--no_window", "--benchmark_frames", "15",
                                      "--report", str(report)])
    assert rc == 0
    rep = json.loads(report.read_text())
    assert rep["frames"] == 15 and rep["landmarks_ms_mean"] > 0 and rep["eye_model"] == "none"


def test_onnx_export_matches_pytorch(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from cues.eye_model import EyeModelConfig, EyeNet, EyeStatePredictor, save_checkpoint
    from cues.predictors import load_eye_predictor
    model = EyeNet(EyeModelConfig(pretrained=False)).eval()
    model.temperature.fill_(1.7)
    save_checkpoint(tmp_path / "m.pt", model)
    assert _script("export_onnx").main(["--checkpoint", str(tmp_path / "m.pt"), "--out", str(tmp_path / "m.onnx")]) == 0
    crops = np.random.default_rng(0).integers(0, 256, (4, 64, 64), dtype=np.uint8)
    a = EyeStatePredictor(tmp_path / "m.pt").predict_crops(crops)
    b = load_eye_predictor(tmp_path / "m.onnx").predict_crops(crops)
    np.testing.assert_allclose(a, b, atol=1e-4)
