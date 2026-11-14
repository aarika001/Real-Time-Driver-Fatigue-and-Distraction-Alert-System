"""Milestone 2 tests: MAR geometry, calibration, yawn features, YawDD scoring
and the two YawDD scripts end to end (on synthetic data).

Run from the repo root:  python -m pytest tests/r4 -v
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cues import landmarks as L
from cues.mouth_cue import (MouthConfig, MouthCueExtractor, YawnTracker, load_rldd_mar, mar_inner,
                            mar_outer, robust_baseline, yawn_events, yawn_features)
from cues.schemas import FRAME_FEATURE_KEYS, R4FrameOutput
from cues.yawdd import ground_truth_for, load_annotations, parse_yawdd_path, score_events

REPO_ROOT = Path(__file__).resolve().parents[2]
FPS = 15.0


def mouth_landmarks(open_px: float, width_px: float = 60.0, yaw_deg: float = 0.0,
                    cx: float = 320.0, cy: float = 300.0) -> np.ndarray:
    """(478, 3) array with only the lip points filled in, optionally rotated about the vertical axis."""
    pts = np.zeros((L.NUM_LANDMARKS, 3), dtype=np.float32)
    half = width_px / 2
    lip = 6.0   # lip thickness, so the outer MAR is > 0 when closed
    xs = (-half * 0.5, 0.0, half * 0.5)
    o0, o1 = L.MOUTH_OUTER[:2]
    i0, i1 = L.MOUTH_INNER[:2]
    for idx, x in ((o0, -half), (o1, half), (i0, -half * 0.9), (i1, half * 0.9)):
        pts[idx] = (x, 0.0, 0.0)
    for k, x in enumerate(xs):
        up_o, lo_o = L.MOUTH_OUTER[2 + 2 * k], L.MOUTH_OUTER[3 + 2 * k]
        up_i, lo_i = L.MOUTH_INNER[2 + 2 * k], L.MOUTH_INNER[3 + 2 * k]
        pts[up_i] = (x, -open_px / 2, 0.0)
        pts[lo_i] = (x, open_px / 2, 0.0)
        pts[up_o] = (x, -open_px / 2 - lip, 0.0)
        pts[lo_o] = (x, open_px / 2 + lip, 0.0)
    a = math.radians(yaw_deg)
    rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]], dtype=np.float32)
    pts = pts @ rot.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return pts


# --- geometry ----------------------------------------------------------------
def test_mar_increases_with_opening():
    closed, yawn = mouth_landmarks(0.0), mouth_landmarks(40.0)
    assert mar_inner(closed) < 0.01
    assert mar_outer(closed) == pytest.approx(12 / 60, rel=1e-3)   # lip thickness only
    assert mar_inner(yawn) > 0.6 and mar_outer(yawn) > mar_outer(closed)


def test_inner_3d_mar_is_robust_to_head_yaw_but_2d_is_not():
    front, turned = mouth_landmarks(20.0), mouth_landmarks(20.0, yaw_deg=40.0)
    assert mar_inner(turned) == pytest.approx(mar_inner(front), rel=1e-4)
    assert mar_outer(turned) > mar_outer(front) * 1.2   # 2-D inflates when width is foreshortened


def test_outer_mar_matches_r3_rldd_formula():
    pytest.importorskip("mediapipe")
    pytest.importorskip("pandas")
    from tests.r4.test_contract import _load_r3_module   # reuse loader (handles missing tqdm)
    r3 = _load_r3_module()
    pts = mouth_landmarks(17.0, yaw_deg=10.0)
    w, h = 640, 480
    lms = [SimpleNamespace(x=float(p[0] / w), y=float(p[1] / h), z=0.0) for p in pts]
    arr = L.landmarks_to_array(lms, w, h)
    assert mar_outer(arr) == pytest.approx(r3.compute_mar(lms, w, h), rel=1e-5)


# --- calibration ----------------------------------------------------------------
def test_robust_baseline_ignores_nan_and_outliers():
    vals = [0.30] * 40 + [np.nan] * 10 + [0.9, 1.0]   # a smile does not move the median
    assert robust_baseline(vals) == pytest.approx(0.30)
    with pytest.raises(ValueError):
        robust_baseline([0.3] * 5 + [np.nan] * 50)


def test_config_validation_and_unknown_keys():
    assert MouthConfig.from_dict({"open_ratio": 1.8}).open_ratio == 1.8
    with pytest.raises(KeyError):
        MouthConfig.from_dict({"open_ration": 1.8})
    with pytest.raises(ValueError):
        MouthConfig.from_dict({"open_ratio": 2.5, "yawn_peak_ratio": 2.0})


def test_repo_yaml_config_loads():
    from cues.config import load_config
    cfg = MouthConfig.from_dict(load_config(REPO_ROOT / "configs" / "r4.yaml")["mouth"])
    assert cfg.fps == 15.0


# --- temporal features --------------------------------------------------------
def _signal(*segments):
    return np.concatenate([np.full(int(s * FPS), v, dtype=float) for v, s in segments])


def test_yawn_detected_once_with_correct_timing():
    cfg = MouthConfig()
    norm = _signal((1.0, 2.0), (2.6, 3.0), (1.0, 2.0))
    f = yawn_features(norm * 0.3, 0.3, cfg)
    ev = yawn_events(f["yawn_flag"], f["open_run_s"], FPS)
    assert len(ev) == 1
    start, end, detect = ev[0]
    assert start == pytest.approx(2.0, abs=0.1)
    assert detect == pytest.approx(2.0 + cfg.min_yawn_s, abs=0.1)
    assert end == pytest.approx(5.0, abs=0.1)


def test_talking_is_not_a_yawn_and_has_high_oscillation():
    cfg = MouthConfig()
    t = np.arange(int(10 * FPS)) / FPS
    norm = 1.0 + 0.5 * (np.sin(2 * np.pi * 3.0 * t) > 0)   # 3 Hz open/close, peak 1.5
    f = yawn_features(norm * 0.3, 0.3, cfg)
    assert f["yawn_flag"].max() == 0.0
    assert np.median(f["osc_hz"][int(3 * FPS):]) == pytest.approx(3.0, abs=0.6)


def test_short_hand_occlusion_does_not_split_a_yawn_but_long_one_does():
    cfg = MouthConfig()
    short = _signal((1.0, 1), (2.6, 1.5), (np.nan, 0.3), (2.6, 1.5), (1.0, 1))
    long_ = _signal((1.0, 1), (2.6, 1.6), (np.nan, 1.5), (2.6, 1.6), (1.0, 1))
    for norm, n_expected in ((short, 1), (long_, 2)):
        f = yawn_features(norm * 0.3, 0.3, cfg)
        assert len(yawn_events(f["yawn_flag"], f["open_run_s"], FPS)) == n_expected


def test_streaming_tracker_matches_batch():
    cfg = MouthConfig()
    rng = np.random.default_rng(0)
    norm = np.clip(1 + rng.normal(0, 0.6, 400), 0.5, 3.0)
    norm[rng.random(400) < 0.05] = np.nan
    batch = yawn_features(norm, 1.0, cfg)
    tr = YawnTracker(cfg)
    stream = np.array([tr.step(x) for x in norm])
    np.testing.assert_allclose(stream[:, 2], batch["yawn_flag"])
    np.testing.assert_allclose(stream[:, 1], batch["osc_hz"], rtol=1e-6)


# --- live extractor + schema ------------------------------------------------
def test_extractor_flow_before_and_after_calibration():
    mouth = MouthCueExtractor()
    r = mouth.update(mouth_landmarks(0.0))
    assert r.valid and math.isnan(r.mar_norm) and math.isnan(r.yawn_flag)
    assert mouth.update(None).valid is False

    mouth.calibrate([mouth.update(mouth_landmarks(0.0)).mar for _ in range(40)])
    flags = [mouth.update(mouth_landmarks(40.0)).yawn_flag for _ in range(int(3 * FPS))]
    assert flags[0] == 0.0 and flags[-1] == 1.0
    r = mouth.update(mouth_landmarks(40.0))
    d = R4FrameOutput(0.0, True, mouth=r).to_flat_dict()
    assert tuple(d) == FRAME_FEATURE_KEYS and d["yawn_flag"] == 1.0 and d["mar_norm"] > 2


def test_load_rldd_mar_masks_lost_frames(tmp_path):
    p = tmp_path / "kss_0_alert.npz"
    np.savez_compressed(p, mar=np.array([0.3, 0.0, 0.35], np.float32),
                        face_detected=np.array([1, 0, 1], np.uint8))
    m = load_rldd_mar(p)
    assert m[0] == pytest.approx(0.3) and np.isnan(m[1])


# --- YawDD helpers ----------------------------------------------------------------
def test_parse_yawdd_names():
    a = parse_yawdd_path("Mirror/Female_mirror/12-FemaleGlasses-Yawning.avi")
    assert (a.action, a.eyewear, a.camera, a.subject_id) == ("yawning", "glasses", "mirror",
                                                             "mirror/12-FemaleGlasses")
    b = parse_yawdd_path("Dash/Male/3-MaleSunGlasses.avi")
    assert (b.action, b.eyewear, b.camera) == ("mixed", "sunglasses", "dash")
    assert parse_yawdd_path("x/5-MaleNoGlasses-Talking.avi").eyewear == "none"


def test_score_events_hits_false_alarms_and_delay():
    gt = [(2.0, 6.0), (20.0, 24.0)]
    det = [(2.1, 6.0, 3.5), (10.0, 12.0, 11.5)]
    s = score_events(det, gt, tolerance_s=1.0)
    assert (s.n_gt, s.n_hit, s.n_false) == (2, 1, 1)
    assert s.delays_s == (1.5,)


def test_annotations_and_ground_truth_rules(tmp_path):
    csv = tmp_path / "a.csv"
    pd.DataFrame({"video_id": ["1-MaleNoGlasses-Yawning"], "start_s": [2.0], "end_s": [6.0]}).to_csv(csv, index=False)
    ann = load_annotations(csv)
    assert ground_truth_for(parse_yawdd_path("1-MaleNoGlasses-Yawning.avi"), ann) == [(2.0, 6.0)]
    assert ground_truth_for(parse_yawdd_path("1-MaleNoGlasses-Talking.avi"), ann) == []
    assert ground_truth_for(parse_yawdd_path("2-MaleNoGlasses-Yawning.avi"), ann) is None


# --- scripts end to end ------------------------------------------------------------
def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "r4" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_feature_dir(root: Path, n_people: int = 6) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    rows, ann = [], []
    for p in range(1, n_people + 1):
        neutral = 0.25 + 0.03 * p
        for action in ("Normal", "Talking", "Yawning"):
            vid = f"{p}-MaleNoGlasses-{action}"
            n = int(30 * FPS)
            norm = 1 + rng.normal(0, 0.03, n)
            if action == "Talking":
                t = np.arange(n) / FPS
                norm += 0.4 * (np.sin(2 * np.pi * 3 * t) > 0)
            if action == "Yawning":
                for s in (5.0, 18.0):
                    norm[int(s * FPS):int((s + 4) * FPS)] = 2.7
                    ann.append({"video_id": vid, "start_s": s, "end_s": s + 4})
            np.savez_compressed(root / f"{vid}.npz", mar=(norm * neutral).astype(np.float32),
                                mar_inner=np.zeros(n, np.float32), face_detected=np.ones(n, np.uint8),
                                t_s=(np.arange(n) / FPS).astype(np.float32), fps=np.float32(FPS),
                                source_fps=np.float32(30))
            info = parse_yawdd_path(f"Mirror/{vid}.avi")
            rows.append({**info.__dict__, "npz": f"{vid}.npz", "n_frames": n, "face_detection_rate": 1.0})
    pd.DataFrame(rows).to_csv(root / "index.csv", index=False)
    pd.DataFrame(ann).to_csv(root / "yawns.csv", index=False)
    return root


def test_eval_yawdd_end_to_end(tmp_path):
    feats = _fake_feature_dir(tmp_path / "feats")
    out = tmp_path / "out"
    rc = _load_script("eval_yawdd").main(["--features_dir", str(feats), "--annotations",
                                          str(feats / "yawns.csv"), "--out_dir", str(out), "--sweep"])
    assert rc == 0
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "held-out people" in summary and "Calibration modes" in summary
    per_video = pd.read_csv(out / "per_video_subject.csv")
    yawning = per_video[per_video["action"] == "yawning"]
    assert yawning["n_hit"].sum() == yawning["n_gt"].sum() == 2 * 6
    assert per_video[per_video["action"] == "talking"]["n_false"].sum() == 0


def test_extract_yawdd_runs_mediapipe_on_a_video(tmp_path):
    pytest.importorskip("mediapipe")
    import cv2
    if not (REPO_ROOT / "face_landmarker.task").exists():
        pytest.skip("face_landmarker.task not in repo root")
    vdir = tmp_path / "YawDD" / "Mirror"
    vdir.mkdir(parents=True)
    path = vdir / "1-MaleNoGlasses-Normal.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (160, 120))
    if not vw.isOpened():
        pytest.skip("OpenCV cannot write test video here")
    for i in range(60):   # 2 s at 30 FPS, no face
        vw.write(np.full((120, 160, 3), (i * 4) % 255, np.uint8))
    vw.release()
    out = tmp_path / "feats"
    assert _load_script("extract_yawdd_mar").main(["--videos_dir", str(tmp_path / "YawDD"),
                                                   "--out_dir", str(out)]) == 0
    with np.load(out / "1-MaleNoGlasses-Normal.npz") as z:
        assert len(z["mar"]) == 30            # 2 s resampled to 15 FPS
        assert z["face_detected"].sum() == 0 and np.isnan(z["mar"]).all()
    idx = pd.read_csv(out / "index.csv")
    assert idx.loc[0, "camera"] == "mirror" and idx.loc[0, "action"] == "normal"
