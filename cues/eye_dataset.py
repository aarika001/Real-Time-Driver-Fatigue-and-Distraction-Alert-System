"""Eye-crop datasets and subject-independent splits (R4, Milestone 3).

Sources
-------
MRL Eye (Fusek, 2018): 84,898 infrared eye crops from 37 people. Every label
is in the file name, e.g. ``s0012_01234_0_1_0_2_1_01.png``::

    subject _ image _ gender _ glasses _ eye_state _ reflections _ lighting _ sensor
    eye_state: 0 = closed, 1 = open     reflections: 0 none, 1 low, 2 high
    glasses:   0 = no, 1 = yes          lighting:    0 bad, 1 good

Other labelled crops (NTHU-DDD, DMD, your own RGB crops) are added through a
CSV index with columns ``path, subject_id, label_closed`` (+ optional tag
columns such as ``glasses``, ``lighting``); paths are relative to the CSV.

Pipeline per image
------------------
Load grayscale -> resize to 64x64 (INTER_AREA), cached in RAM as uint8 ->
[training only] domain-gap augmentation (``augment.py``) -> CLAHE
(``eye_crop.normalize_crop``) -> float32 in [0, 1], shape (1, 64, 64).
The CLAHE step is the same one used on live webcam crops and on R3's RLDD
archives, so training and deployment see identically normalised inputs.

Splits are by PERSON (subject_id), never by image: the same person's eyes in
train and test would let the network memorise faces and inflate the scores.
"""
from __future__ import annotations

import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import pandas as pd

from .augment import AugmentConfig, augment_eye
from .eye_crop import CROP_SIZE, _CLAHE

MRL_NAME_RE = re.compile(r"s(\d{4})_(\d+)_(\d)_(\d)_(\d)_(\d)_(\d)_(\d+)\.png$", re.IGNORECASE)
REFLECTION_NAMES = {0: "none", 1: "low", 2: "high"}
LIGHTING_NAMES = {0: "bad", 1: "good"}
SENSOR_NAMES = {1: "realsense", 2: "ids", 3: "aptina"}
TAG_COLUMNS = ("glasses", "reflection", "lighting", "sensor")


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------
def parse_mrl_name(name: str) -> Optional[dict]:
    """Labels from an MRL file name, or None if it does not match."""
    m = MRL_NAME_RE.search(name.replace("\\", "/").split("/")[-1])
    if not m:
        return None
    subj, img, gender, glasses, state, refl, light, sensor = m.groups()
    return {
        "subject_id": f"mrl/s{subj}",
        "image_id": int(img),
        "label_closed": 1 - int(state),          # we predict P(closed): 1 = closed
        "glasses": "yes" if glasses == "1" else "no",
        "reflection": REFLECTION_NAMES[int(refl)],
        "lighting": LIGHTING_NAMES[int(light)],
        "sensor": SENSOR_NAMES.get(int(sensor), f"s{sensor}"),
        "gender": "female" if gender == "1" else "male",
    }


def build_mrl_index(root: str | Path) -> pd.DataFrame:
    """Index MRL Eye from the extracted folder or directly from the .zip."""
    root = Path(root)
    rows = []
    if root.is_file() and root.suffix.lower() == ".zip":
        with zipfile.ZipFile(root) as z:
            members = [n for n in z.namelist() if n.lower().endswith(".png")]
        for n in members:
            meta = parse_mrl_name(n)
            if meta:
                rows.append({"path": n, "zip_path": str(root), **meta})
    elif root.is_dir():
        for p in root.rglob("*.png"):
            meta = parse_mrl_name(p.name)
            if meta:
                rows.append({"path": str(p), "zip_path": "", **meta})
    else:
        raise FileNotFoundError(f"MRL root not found (folder or .zip expected): {root}")
    if not rows:
        raise ValueError(f"No MRL-style images found under {root}")
    df = pd.DataFrame(rows)
    df.insert(0, "dataset", "mrl")
    return df.sort_values(["subject_id", "image_id"]).reset_index(drop=True)


def load_csv_index(csv_path: str | Path, dataset_name: Optional[str] = None) -> pd.DataFrame:
    """Index for other labelled crops. Required columns: path, subject_id, label_closed."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, dtype={"subject_id": str})   # keep "001" as "001"
    missing = {"path", "subject_id", "label_closed"} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns {sorted(missing)}")
    if not set(df["label_closed"].unique()) <= {0, 1}:
        raise ValueError("label_closed must be 0 (open) or 1 (closed)")
    df["path"] = [str((csv_path.parent / p).resolve()) if not Path(p).is_absolute() else p for p in df["path"]]
    df["zip_path"] = ""
    df["dataset"] = dataset_name or df.get("dataset", csv_path.stem)
    df["subject_id"] = df["dataset"].astype(str) + "/" + df["subject_id"].astype(str)
    for col in TAG_COLUMNS:
        if col not in df.columns:
            df[col] = "unknown"
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Subject-independent splits
# ---------------------------------------------------------------------------
def split_by_subject(df: pd.DataFrame, n_val: int, n_test: int, seed: int = 42) -> pd.Series:
    """Assign whole people to train / val / test.

    Within each dataset, people are shuffled with ``seed`` and the first
    ``n_test`` go to test, the next ``n_val`` to val (scaled down for small
    datasets so every dataset keeps most people for training).
    """
    split = pd.Series("train", index=df.index, dtype=object)
    rng = np.random.default_rng(seed)
    for _, part in df.groupby("dataset"):
        subjects = np.array(sorted(part["subject_id"].unique()))
        rng.shuffle(subjects)
        n = len(subjects)
        k_test = min(n_test, max(1, n // 5)) if n >= 3 else 0
        k_val = min(n_val, max(1, n // 6)) if n >= 3 else 0
        test_s, val_s = set(subjects[:k_test]), set(subjects[k_test:k_test + k_val])
        split[part.index[part["subject_id"].isin(test_s)]] = "test"
        split[part.index[part["subject_id"].isin(val_s)]] = "val"
    assert_no_subject_leak(df.assign(split=split))
    return split


def group_kfold_assign(df: pd.DataFrame, k: int = 5, seed: int = 42) -> pd.Series:
    """Fold id (0..k-1) per row, with each person in exactly one fold (for R3's k-fold)."""
    subjects = np.array(sorted(df["subject_id"].unique()))
    np.random.default_rng(seed).shuffle(subjects)
    fold_of = {s: i % k for i, s in enumerate(subjects)}
    return df["subject_id"].map(fold_of).astype(int)


def assert_no_subject_leak(df: pd.DataFrame, col: str = "split") -> None:
    """Raise if any person appears in more than one split."""
    per_subject = df.groupby("subject_id")[col].nunique()
    leaked = per_subject[per_subject > 1]
    if len(leaked):
        raise AssertionError(f"Subject leakage across '{col}': {list(leaked.index[:5])}")


# ---------------------------------------------------------------------------
# Image loading (into RAM, before augmentation and CLAHE)
# ---------------------------------------------------------------------------
def _decode_resize(buf: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("could not decode image")
    if img.shape != (CROP_SIZE, CROP_SIZE):
        img = cv2.resize(img, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
    return img


def load_raw_crops(df: pd.DataFrame, num_threads: int = 8, verbose: bool = True) -> np.ndarray:
    """(N, 64, 64) uint8 raw (not yet CLAHE'd) crops in index order."""
    out = np.zeros((len(df), CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    from_zip = df["zip_path"].fillna("").astype(str) != ""
    for zip_path, part in df[from_zip].groupby("zip_path"):
        with zipfile.ZipFile(zip_path) as z:
            for i, (pos, member) in enumerate(zip(part.index, part["path"])):
                out[pos] = _decode_resize(z.read(member))
                if verbose and i % 20000 == 0:
                    print(f"  loaded {i}/{len(part)} from {Path(zip_path).name}")
    files = df[~from_zip]
    if len(files):
        def read(p: str) -> np.ndarray:
            with open(p, "rb") as f:
                return _decode_resize(f.read())
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            for pos, img in zip(files.index, ex.map(read, files["path"], chunksize=256)):
                out[pos] = img
        if verbose:
            print(f"  loaded {len(files)} image files")
    return out


# ---------------------------------------------------------------------------
# Dataset (works with torch.utils.data.DataLoader without importing torch here)
# ---------------------------------------------------------------------------
class EyeCropDataset:
    """Map-style dataset returning (image (1,64,64) float32 in [0,1], label float32, index).

    Args:
        raw: (N, 64, 64) uint8 raw crops from ``load_raw_crops``.
        labels: (N,) 0/1 with 1 = closed.
        augment: AugmentConfig to enable training augmentation, or None.
        seed: base seed; each (epoch, index) pair gets its own random stream so
            results are reproducible with any number of DataLoader workers.
    """

    def __init__(self, raw: np.ndarray, labels: Sequence[int],
                 augment: Optional[AugmentConfig] = None, seed: int = 0):
        if len(raw) != len(labels):
            raise ValueError("raw and labels have different lengths")
        self.raw = raw
        self.labels = np.asarray(labels, dtype=np.float32)
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        img = self.raw[i]
        if self.augment is not None:
            rng = np.random.default_rng((self.seed, self.epoch, int(i)))
            img = augment_eye(img, rng, self.augment)
        img = _CLAHE.apply(np.ascontiguousarray(img))
        x = (img.astype(np.float32) / 255.0)[None]
        return x, self.labels[i], i
