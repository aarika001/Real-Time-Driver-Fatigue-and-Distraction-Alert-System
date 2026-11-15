"""Milestones 3-4 tests: augmentation, MRL indexing, subject splits, dataset,
metrics, and (when PyTorch is installed) the model, predictor and a smoke
training run on a synthetic MRL-style folder.

Run from the repo root:  python -m pytest tests/r4 -v
"""
from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from cues.augment import AugmentConfig, augment_eye, random_glare
from cues.eye_dataset import (EyeCropDataset, assert_no_subject_leak, build_mrl_index, group_kfold_assign,
                              load_csv_index, load_raw_crops, parse_mrl_name, split_by_subject)
from cues.eye_metrics import (binary_metrics, bootstrap_ci_over_subjects, expected_calibration_error,
                              summarize_predictions, worst_condition)

REPO_ROOT = Path(__file__).resolve().parents[2]


def fake_eye(closed: bool, rng: np.random.Generator, size: int = 86) -> np.ndarray:
    img = np.full((size, size), int(rng.integers(90, 140)), np.uint8)
    c = size // 2
    if closed:
        cv2.line(img, (c - 30, c), (c + 30, c), 30, 3)
    else:
        cv2.ellipse(img, (c, c), (30, 16), 0, 0, 360, 220, -1)
        cv2.circle(img, (c, c), 10, 20, -1)
    return img


def make_fake_mrl(root: Path, n_subjects: int = 8, per_subject: int = 24, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    base = root / "mrlEyes_2018_01"
    for s in range(1, n_subjects + 1):
        d = base / f"s{s:04d}"
        d.mkdir(parents=True, exist_ok=True)
        for k in range(per_subject):
            state = k % 2                     # 1 = open, 0 = closed
            glasses = int(s % 3 == 0)
            refl, light = k % 3, (k // 3) % 2
            name = f"s{s:04d}_{k + 1:05d}_{s % 2}_{glasses}_{state}_{refl}_{light}_01.png"
            cv2.imwrite(str(d / name), fake_eye(closed=state == 0, rng=rng))
    return base


# --- augmentation --------------------------------------------------------------
def test_augment_keeps_shape_dtype_and_is_reproducible():
    img = fake_eye(False, np.random.default_rng(0), 64)
    cfg = AugmentConfig()
    a = augment_eye(img, np.random.default_rng(7), cfg)
    b = augment_eye(img, np.random.default_rng(7), cfg)
    assert a.shape == img.shape and a.dtype == np.uint8
    np.testing.assert_array_equal(a, b)
    outs = [augment_eye(img, np.random.default_rng(i), cfg) for i in range(20)]
    assert len({o.tobytes() for o in outs}) > 15           # actually random


def test_glare_adds_saturated_pixels():
    img = np.full((64, 64), 100, np.uint8)
    out = random_glare(img, np.random.default_rng(3))
    assert (out >= 220).sum() > (img >= 220).sum()


def test_augment_config_rejects_typos():
    with pytest.raises(KeyError):
        AugmentConfig.from_dict({"p_glar": 0.5})
    assert AugmentConfig.from_dict({"gamma_range": [0.7, 1.5]}).gamma_range == (0.7, 1.5)


# --- MRL index and splits -------------------------------------------------------
def test_parse_mrl_name_maps_labels():
    m = parse_mrl_name("mrlEyes_2018_01/s0012/s0012_01234_1_1_0_2_0_02.png")
    assert m["subject_id"] == "mrl/s0012" and m["label_closed"] == 1
    assert (m["glasses"], m["reflection"], m["lighting"], m["sensor"]) == ("yes", "high", "bad", "ids")
    assert parse_mrl_name("random.png") is None


def test_build_index_from_folder_and_zip_agree(tmp_path):
    base = make_fake_mrl(tmp_path / "data", n_subjects=3, per_subject=6)
    from_dir = build_mrl_index(base)
    zpath = tmp_path / "mrl.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for p in base.rglob("*.png"):
            z.write(p, p.relative_to(base.parent).as_posix())
    from_zip = build_mrl_index(zpath)
    assert len(from_dir) == len(from_zip) == 18
    assert from_dir["label_closed"].sum() == from_zip["label_closed"].sum() == 9
    raw_d, raw_z = load_raw_crops(from_dir, verbose=False), load_raw_crops(from_zip, verbose=False)
    assert raw_d.shape == (18, 64, 64)
    np.testing.assert_array_equal(raw_d, raw_z)


def test_split_by_subject_has_no_leak_and_expected_sizes(tmp_path):
    df = build_mrl_index(make_fake_mrl(tmp_path, n_subjects=37, per_subject=2))
    split = split_by_subject(df, n_val=5, n_test=7, seed=42)
    per = df.assign(split=split).groupby("split")["subject_id"].nunique()
    assert (per["test"], per["val"], per["train"]) == (7, 5, 25)
    assert_no_subject_leak(df.assign(split=split))
    again = split_by_subject(df, n_val=5, n_test=7, seed=42)
    assert (split == again).all()                         # deterministic


def test_leak_detector_catches_a_leak():
    df = pd.DataFrame({"subject_id": ["a", "a", "b"], "split": ["train", "test", "val"]})
    with pytest.raises(AssertionError):
        assert_no_subject_leak(df)


def test_group_kfold_assigns_each_person_to_one_fold(tmp_path):
    df = build_mrl_index(make_fake_mrl(tmp_path, n_subjects=10, per_subject=2))
    folds = group_kfold_assign(df, k=5)
    assert df.assign(f=folds).groupby("subject_id")["f"].nunique().max() == 1
    assert sorted(folds.unique()) == [0, 1, 2, 3, 4]


def test_csv_index_for_extra_datasets(tmp_path):
    img = tmp_path / "crops" / "a.png"
    img.parent.mkdir()
    cv2.imwrite(str(img), fake_eye(True, np.random.default_rng(0), 50))
    csv = tmp_path / "nthu.csv"
    pd.DataFrame({"path": ["crops/a.png"], "subject_id": ["001"], "label_closed": [1],
                  "glasses": ["yes"]}).to_csv(csv, index=False)
    df = load_csv_index(csv, dataset_name="nthu")
    assert df.loc[0, "subject_id"] == "nthu/001" and df.loc[0, "lighting"] == "unknown"
    assert load_raw_crops(df, verbose=False).shape == (1, 64, 64)


# --- dataset --------------------------------------------------------------------
def test_dataset_items_are_normalised_and_epoch_dependent():
    rng = np.random.default_rng(0)
    raw = np.stack([fake_eye(i % 2 == 0, rng, 64) for i in range(4)])
    ds = EyeCropDataset(raw, [1, 0, 1, 0], augment=AugmentConfig(), seed=1)
    x, y, i = ds[0]
    assert x.shape == (1, 64, 64) and x.dtype == np.float32 and 0 <= x.min() and x.max() <= 1
    assert y == 1.0 and i == 0
    ds.set_epoch(1)
    assert not np.array_equal(ds[0][0], x)                # new augmentation each epoch
    plain = EyeCropDataset(raw, [1, 0, 1, 0])
    np.testing.assert_array_equal(plain[2][0], plain[2][0])  # no augmentation -> deterministic


# --- metrics --------------------------------------------------------------------
def test_binary_metrics_and_ece():
    y = np.array([1, 1, 0, 0])
    m = binary_metrics(np.array([0.9, 0.8, 0.2, 0.6]), y)
    assert m["accuracy"] == 0.75 and m["recall_closed"] == 1.0 and m["auroc"] == 1.0
    assert expected_calibration_error(np.array([1.0, 1.0, 0.0, 0.0]), y) == pytest.approx(0.0)


def test_summary_flags_small_groups_and_finds_worst(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for s in range(12):
        for k in range(20):
            y = k % 2
            glasses = "yes" if s < 6 else "no"
            noise = 0.45 if glasses == "yes" else 0.1
            rows.append({"subject_id": f"p{s}", "label_closed": y, "glasses": glasses,
                         "sensor": "ids" if s < 2 else "realsense",
                         "p_closed": float(np.clip(y + rng.normal(0, noise) * (1 - 2 * y), 0, 1))})
    df = pd.DataFrame(rows)
    summ = summarize_predictions(df, ["glasses", "sensor"], n_boot=100)
    assert summ.iloc[0]["group"] == "overall"
    assert summ.set_index("group").loc["sensor=ids", "flag"] == "fewer than 5 people"
    assert worst_condition(summ, "accuracy")["group"] == "glasses=yes"
    lo, hi = bootstrap_ci_over_subjects(df["p_closed"].to_numpy(), df["label_closed"].to_numpy(),
                                        df["subject_id"], n_boot=100)
    assert lo <= summ.iloc[0]["accuracy"] <= hi


# --- PyTorch parts (run on Kaggle / your laptop; skipped where torch is missing) ---
def test_model_forward_predictor_and_temperature(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from cues.eye_model import EyeModelConfig, EyeNet, EyeStatePredictor, fit_temperature, save_checkpoint
    model = EyeNet(EyeModelConfig(pretrained=False))
    out = model(torch.rand(3, 1, 64, 64))
    assert out.shape == (3,)
    with pytest.raises(ValueError):
        EyeNet(EyeModelConfig(pretrained=True, width_mult=0.5))
    model.temperature.fill_(2.0)
    save_checkpoint(tmp_path / "m.pt", model, {"note": np.float32(1.5)})
    pred = EyeStatePredictor(tmp_path / "m.pt")
    p = pred.predict_crops(np.zeros((2, 64, 64), np.uint8))
    assert p.shape == (2,) and p.dtype == np.float32 and ((0 <= p) & (p <= 1)).all()
    assert pred.temperature == pytest.approx(2.0)
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 5000)                                 # perfectly calibrated logits
    y = (rng.random(5000) < 1 / (1 + np.exp(-z))).astype(int)
    assert fit_temperature(z * 4, y) == pytest.approx(4.0, rel=0.2)    # undoes 4x over-confidence


def test_train_script_smoke_run(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    base = make_fake_mrl(tmp_path / "data", n_subjects=8, per_subject=24)
    spec = importlib.util.spec_from_file_location("train_eye_cnn", REPO_ROOT / "scripts" / "r4" / "train_eye_cnn.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod      # dataclasses need the module registered (Python 3.13)
    spec.loader.exec_module(mod)
    out = tmp_path / "run"
    assert mod.main(["--mrl_root", str(base), "--out_dir", str(out), "--smoke", "--cpu", "--num_workers", "0"]) == 0
    for f in ("best.pt", "last.pt", "history.csv", "splits.csv", "test_predictions.csv", "test_report.md",
              "test_metrics.json"):
        assert (out / f).exists(), f
    splits = pd.read_csv(out / "splits.csv")
    assert_no_subject_leak(splits)
