"""Feature building, per-driver calibration, windowing and subject-safe splits
for the UTA-RLDD temporal model.

Pure numpy / pandas (no torch) so it can be tested and reused by the live demo.

Per-frame feature layout (N_FEATURES = 13)
------------------------------------------
cues         [0:5]   eye_closed_prob | mar_z | pitch_z | yaw_z | roll_z
reliability  [5:10]  brightness | contrast | glare_frac | log-sharpness | face_detected
availability [10:13] flag for each cue group: eye | mouth | head   (1 = cue present)

* mar_z / pitch_z / yaw_z / roll_z are expressed relative to the driver's own
  30-second baseline (per-driver calibration).  Angle differences are wrapped to
  (-180, 180] because solvePnP pitch often sits near +/-180 degrees and would
  otherwise jump between the two ends.
* eye_closed_prob is optional.  The preprocessed .npz files only hold eye *crops*;
  the eye-state CNN (R4) is a separate model.  If an .npz contains an
  ``eye_closed_prob`` array of shape (N,) it is used; otherwise the eye group is
  absent (zeros, availability flag 0).
* The reliability channels are the raw quality signals from the preprocessor.  They
  are a stand-in for the learned reliability scores of the reliability estimator
  (R7); replace them with its outputs when available.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- constants
FPS = 15
WINDOW = 75            # 5 s at 15 FPS
STRIDE = 15            # 1 s
CALIB_FRAMES = 450     # 30 s at 15 FPS

LABEL_MODES: Dict[str, Dict[str, int]] = {
    # alert vs fatigue; KSS-5 "low_vigilant" videos are dropped from train/eval
    "binary": {"alert": 0, "fatigue": 1},
    # low_vigilant counted as fatigue (early drowsiness)
    "binary_lv_fatigue": {"alert": 0, "low_vigilant": 1, "fatigue": 1},
    # keep all three RLDD classes
    "three": {"alert": 0, "low_vigilant": 1, "fatigue": 2},
}
CLASS_NAMES: Dict[str, List[str]] = {
    "binary": ["alert", "fatigue"],
    "binary_lv_fatigue": ["alert", "fatigue"],
    "three": ["alert", "low_vigilant", "fatigue"],
}

CUE_NAMES = ["eye_closed", "mar", "pitch", "yaw", "roll"]
REL_NAMES = ["brightness", "contrast", "glare", "sharpness", "face_detected"]
GROUP_NAMES = ["eye", "mouth", "head"]
GROUP_CUE_IDX = {"eye": [0], "mouth": [1], "head": [2, 3, 4]}

N_CUES, N_REL, N_GROUPS = 5, 5, 3
CUE_START, REL_START, FLAG_START = 0, N_CUES, N_CUES + N_REL
N_FEATURES = N_CUES + N_REL + N_GROUPS  # 13

MIN_MAR_SD = 0.01      # floors so a very still driver does not blow up the z-scores
MIN_POSE_SD = 1.0      # degrees
Z_CLIP = 8.0
SHARP_SCALE = 8.0      # log1p(Laplacian variance) / 8  -> roughly [0, 1]


# ----------------------------------------------------------------------------- records
@dataclass
class VideoRec:
    subject: str            # unique group id "<fold>_<subject_id>" - the unit we never split
    fold: str
    kss: str
    label: str
    mar: np.ndarray         # (N,)
    pose: np.ndarray        # (N, 3) pitch, yaw, roll in degrees
    quality: np.ndarray     # (N, 4) brightness, contrast, glare, sharpness
    det: np.ndarray         # (N,) uint8 face_detected
    eye: Optional[np.ndarray] = None  # (N,) P(eye closed) if available
    name: str = ""

    @property
    def n(self) -> int:
        return len(self.det)


@dataclass
class Baseline:
    mar_mu: float
    mar_sd: float
    pose_mu: np.ndarray   # (3,) circular mean, degrees
    pose_sd: np.ndarray   # (3,) degrees


# ----------------------------------------------------------------------------- loading
def _resolve_npz(root: str, row: pd.Series) -> str:
    """The index stores the path as written on the preprocessing machine.  If it no
    longer resolves (e.g. data copied to Colab), rebuild it from the folder layout."""
    p = str(row.get("npz_path", ""))
    if p and os.path.isfile(p):
        return p
    rebuilt = os.path.join(
        root, str(row["fold_id"]), f"subject_{row['subject_id']}",
        f"kss_{row['raw_kss']}_{row['label']}.npz",
    )
    return rebuilt


def eye_sidecar_path(npz_path: str) -> str:
    """Per-frame P(eye closed) written by add_eye_probs.py, stored next to the .npz."""
    return os.path.splitext(npz_path)[0] + ".eye.npy"


def load_videos(root: str, label_mode: str = "binary") -> List[VideoRec]:
    """Load every processed video listed in ``rldd_metadata_index.csv``.

    Only the small per-frame arrays are read; the eye-crop arrays inside the .npz are
    never touched (numpy loads .npz members lazily)."""
    if label_mode not in LABEL_MODES:
        raise ValueError(f"label_mode must be one of {list(LABEL_MODES)}")
    csv = os.path.join(root, "rldd_metadata_index.csv")
    if not os.path.isfile(csv):
        raise FileNotFoundError(f"{csv} not found - run RLDD-preprocess/preprocess-rldd.py first")
    meta = pd.read_csv(csv, dtype={"subject_id": str, "raw_kss": str, "fold_id": str, "label": str})
    keep = set(LABEL_MODES[label_mode]) | {"alert"}   # alert video is always needed for calibration
    recs: List[VideoRec] = []
    for _, row in meta.iterrows():
        if row["label"] not in keep:
            continue
        path = _resolve_npz(root, row)
        if not os.path.isfile(path):
            print(f"[warn] missing {path}; skipped")
            continue
        with np.load(path) as z:
            eye = np.nan_to_num(z["eye_closed_prob"].astype(np.float32), nan=0.0) if "eye_closed_prob" in z.files else None
            side = eye_sidecar_path(path)
            if os.path.isfile(side):                       # sidecar wins over an in-file array
                eye = np.nan_to_num(np.load(side).astype(np.float32), nan=0.0)   # NaN = no face (R4 convention)
            if eye is not None and len(eye) != len(z["face_detected"]):
                raise ValueError(f"{path}: eye_closed_prob has {len(eye)} frames but the video has "
                                 f"{len(z['face_detected'])} - re-run add_eye_probs.py")
            recs.append(VideoRec(
                subject=f"{row['fold_id']}_{row['subject_id']}",
                fold=str(row["fold_id"]),
                kss=str(row["raw_kss"]),
                label=str(row["label"]),
                mar=z["mar"].astype(np.float32),
                pose=z["head_pose"].astype(np.float32),
                quality=z["quality"].astype(np.float32),
                det=z["face_detected"].astype(np.uint8),
                eye=eye,
                name=os.path.basename(path),
            ))
    if not recs:
        raise RuntimeError("no videos loaded")
    return recs


# ----------------------------------------------------------------------------- calibration
def wrap_deg(x):
    """Wrap an angle (difference) in degrees to [-180, 180)."""
    return (np.asarray(x, dtype=np.float32) + 180.0) % 360.0 - 180.0


def fit_baseline(mar: np.ndarray, pose: np.ndarray, det: np.ndarray, min_frames: int = 30) -> Baseline:
    """Personal baseline from normal (alert) frames where the face was found."""
    m = det.astype(bool)
    if m.sum() < min_frames:
        raise ValueError(f"only {int(m.sum())} detected frames for calibration (need {min_frames})")
    mar_d, pose_d = mar[m], pose[m]
    ang = np.deg2rad(pose_d.astype(np.float64))
    mu = np.rad2deg(np.arctan2(np.sin(ang).mean(0), np.cos(ang).mean(0)))
    dev = wrap_deg(pose_d - mu)
    return Baseline(
        mar_mu=float(mar_d.mean()),
        mar_sd=float(max(mar_d.std(), MIN_MAR_SD)),
        pose_mu=mu.astype(np.float32),
        pose_sd=np.maximum(dev.std(0), MIN_POSE_SD).astype(np.float32),
    )


def calibration_frames(alert: VideoRec, n_calib: int = CALIB_FRAMES):
    """First 30 s of the subject's alert video (README hand-off note for R6)."""
    s = slice(0, n_calib)
    return alert.mar[s], alert.pose[s], alert.det[s]


def build_subject_calibration(recs: List[VideoRec], n_calib: int = CALIB_FRAMES):
    """subject -> (mar, pose, det) calibration frames.  Subjects without a usable alert
    video are dropped (returned in `dropped`)."""
    calib, dropped = {}, []
    alerts = {r.subject: r for r in recs if r.label == "alert"}
    for subj in sorted({r.subject for r in recs}):
        a = alerts.get(subj)
        if a is None:
            dropped.append((subj, "no alert video"))
            continue
        mar, pose, det = calibration_frames(a, n_calib)
        if det.sum() < 30:
            dropped.append((subj, "face rarely detected in first 30 s of alert video"))
            continue
        calib[subj] = (mar, pose, det)
    return calib, dropped


def baselines_per_subject(calib) -> Dict[str, Baseline]:
    return {s: fit_baseline(*v) for s, v in calib.items()}


def baseline_pooled(calib, subjects) -> Baseline:
    """One shared baseline from the given (training) subjects only - used for the
    'no per-driver calibration' ablation without leaking test subjects."""
    mar = np.concatenate([calib[s][0] for s in subjects])
    pose = np.concatenate([calib[s][1] for s in subjects])
    det = np.concatenate([calib[s][2] for s in subjects])
    return fit_baseline(mar, pose, det)


# ----------------------------------------------------------------------------- per-frame features
def frame_features(rec: VideoRec, base: Baseline, use_reliability: bool = True) -> np.ndarray:
    """(N, N_FEATURES) float32 matrix for one video."""
    n = rec.n
    det = rec.det.astype(bool)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    if rec.eye is not None:
        X[:, 0] = rec.eye
        X[:, FLAG_START + 0] = 1.0
    X[:, 1] = (rec.mar - base.mar_mu) / base.mar_sd
    X[:, 2:5] = wrap_deg(rec.pose - base.pose_mu) / base.pose_sd
    X[:, CUE_START:CUE_START + N_CUES] = np.clip(X[:, CUE_START:CUE_START + N_CUES], -Z_CLIP, Z_CLIP)
    X[~det, CUE_START:CUE_START + N_CUES] = 0.0          # no face -> no cue (not "z = 0")
    X[:, FLAG_START + 1] = 1.0
    X[:, FLAG_START + 2] = 1.0

    if use_reliability:
        q = rec.quality
        X[:, REL_START + 0] = q[:, 0]
        X[:, REL_START + 1] = q[:, 1]
        X[:, REL_START + 2] = q[:, 2]
        X[:, REL_START + 3] = np.clip(np.log1p(np.maximum(q[:, 3], 0.0)) / SHARP_SCALE, 0.0, 1.5)
        X[:, REL_START + 4] = det.astype(np.float32)
    return X


def present_groups(recs: List[VideoRec]) -> List[int]:
    """Indices (into GROUP_NAMES) of cue groups that exist in this data."""
    groups = [1, 2]
    if any(r.eye is not None for r in recs):
        groups = [0] + groups
    return groups


# ----------------------------------------------------------------------------- windows
class WindowSet:
    """All 5-second windows of a set of videos.

    Frames of every video are stored once in a single big array; windows are just
    (global start index, label).  A window never spans two videos or two subjects.
    Windows start after the 30 s calibration period (``skip``)."""

    def __init__(self, feats: List[np.ndarray], dets: List[np.ndarray], recs: List[VideoRec],
                 label_map: Dict[str, int], win: int = WINDOW, stride: int = STRIDE,
                 skip: int = CALIB_FRAMES, min_face_rate: float = 0.2):
        assert len(feats) == len(recs) == len(dets)
        self.win = win
        self.recs = recs
        lens = np.array([len(f) for f in feats])
        offs = np.concatenate([[0], np.cumsum(lens)])
        self.big = np.concatenate(feats).astype(np.float32) if feats else np.zeros((0, N_FEATURES), np.float32)
        det_big = np.concatenate(dets).astype(np.float32) if dets else np.zeros(0, np.float32)
        cs = np.concatenate([[0.0], np.cumsum(det_big)])

        starts, vid, y = [], [], []
        for v, rec in enumerate(recs):
            if rec.label not in label_map:
                continue
            first = offs[v] + skip
            last = offs[v] + lens[v] - win          # inclusive last start
            if last < first:
                continue
            s = np.arange(first, last + 1, stride)
            rate = (cs[s + win] - cs[s]) / win
            s = s[rate >= min_face_rate]
            starts.append(s)
            vid.append(np.full(len(s), v))
            y.append(np.full(len(s), label_map[rec.label]))
        self.start = np.concatenate(starts) if starts else np.zeros(0, np.int64)
        self.vid = np.concatenate(vid) if vid else np.zeros(0, np.int64)
        self.y = np.concatenate(y).astype(np.int64) if y else np.zeros(0, np.int64)
        self.offs = offs
        self.subject = np.array([recs[v].subject for v in self.vid]) if len(self.vid) else np.array([])

    def __len__(self) -> int:
        return len(self.y)

    def batch(self, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g = self.start[idx]
        X = self.big[g[:, None] + np.arange(self.win)[None, :]]     # (B, win, F)
        return X, self.y[idx]

    def local_start(self, idx: np.ndarray) -> np.ndarray:
        """Frame index of each window's first frame inside its own video."""
        return self.start[idx] - self.offs[self.vid[idx]]


# ----------------------------------------------------------------------------- subject-safe splits
def assign_buckets(recs: List[VideoRec], mode: str = "fold", n_splits: int = 5, seed: int = 0) -> Dict[str, str]:
    """subject -> bucket.  mode='fold': the official UTA-RLDD folds.  mode='subject':
    subjects shuffled into ``n_splits`` buckets.  A subject is always in one bucket."""
    subjects = sorted({r.subject for r in recs})
    if mode == "fold":
        by = {}
        for r in recs:
            by.setdefault(r.subject, set()).add(r.fold)
        bad = [s for s, f in by.items() if len(f) != 1]
        if bad:
            raise ValueError(f"subjects appear in several folds: {bad[:5]}")
        return {s: next(iter(f)) for s, f in by.items()}
    if mode == "subject":
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(subjects))
        return {subjects[i]: f"S{k % n_splits}" for k, i in enumerate(order)}
    raise ValueError("mode must be 'fold' or 'subject'")


def rotating_splits(buckets: Dict[str, str]) -> Iterator[dict]:
    """Leave-one-bucket-out with a separate validation bucket.

    For bucket i: test = bucket i, val = bucket i+1 (early stopping + temperature
    scaling), train = all other buckets.  Test subjects are never used for anything
    except the final evaluation.  Every subject is tested exactly once."""
    uniq = sorted(set(buckets.values()))
    if len(uniq) < 3:
        raise ValueError(f"need >= 3 buckets (train/val/test), got {uniq}")
    for i, test_b in enumerate(uniq):
        val_b = uniq[(i + 1) % len(uniq)]
        sel = lambda f: sorted(s for s, b in buckets.items() if f(b))
        split = {
            "name": test_b,
            "test": sel(lambda b: b == test_b),
            "val": sel(lambda b: b == val_b),
            "train": sel(lambda b: b not in (test_b, val_b)),
        }
        assert not (set(split["train"]) & set(split["val"]))
        assert not (set(split["train"]) & set(split["test"]))
        assert not (set(split["val"]) & set(split["test"]))
        yield split
