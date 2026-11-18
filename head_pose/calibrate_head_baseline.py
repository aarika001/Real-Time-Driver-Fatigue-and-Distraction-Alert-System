"""
calibrate_head_baseline.py
==========================
R5 — Head Pose & Off-Axis Tests (Aaditya)
Team NeuroNova · Real-Time Driver Fatigue and Distraction Alert System

Offline CLI that walks the processed_rldd directory tree produced by R3's
preprocess-rldd.py, reads each subject's alert baseline sequence
(kss_0_alert.npz), and computes per-driver neutral-head-pose statistics.

Output
------
  head_baselines.csv    Per-subject mean and std for pitch/yaw/roll over the
                        first 30 s (450 frames at 15 FPS) of their alert
                        recording.  Fed to stress_test_offaxis.py and to R6's
                        per-driver calibration module.

Usage
-----
  python calibrate_head_baseline.py \\
      --processed_dir ./processed_rldd \\
      --output_path   ./head_baselines.csv

  # Optionally override calibration window length:
  python calibrate_head_baseline.py \\
      --processed_dir ./processed_rldd \\
      --output_path   ./head_baselines.csv \\
      --calibration_seconds 20

Population fallback
-------------------
  If a subject has fewer than MIN_CALIBRATION_FRAMES clean frames in their
  alert window (landmark occlusion etc.), their baseline is replaced by the
  across-subject mean computed from subjects that passed the threshold.
  Flagged subjects are printed as warnings and marked with fallback_used=True.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Local import — head_pose_core must be on the Python path.
# When running from the head-pose/ directory this works as-is.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from head_pose_core import compute_head_baseline, MIN_CALIBRATION_FRAMES

TARGET_FPS = 15  # must match R3's preprocessing rate


def find_alert_npz_files(processed_dir: str) -> list[dict]:
    """Walk the processed_rldd tree and collect all kss_0_alert.npz paths.

    Expected layout (from R3):
        processed_rldd/
        └── FoldX/
            └── subject_YY/
                ├── kss_0_alert.npz
                ├── kss_5_low_vigilant.npz
                └── kss_10_fatigue.npz

    Returns
    -------
    list of dicts with keys: subject_id, fold_id, npz_path
    """
    records = []
    for dirpath, _, filenames in os.walk(processed_dir):
        for fname in filenames:
            if fname != "kss_0_alert.npz":
                continue
            full_path = os.path.join(dirpath, fname)
            parts = Path(full_path).parts

            # Extract fold_id (e.g. "Fold1") and subject_id (e.g. "31")
            fold_id = next(
                (p.split("_")[0] for p in parts if p.startswith("Fold")), "Unknown"
            )
            subject_id = next(
                (p.replace("subject_", "") for p in parts if p.startswith("subject_")),
                "unknown",
            )
            records.append(
                {
                    "subject_id": subject_id,
                    "fold_id": fold_id,
                    "npz_path": full_path,
                }
            )
    return records


def compute_baselines(records: list[dict], calibration_frames: int) -> pd.DataFrame:
    """Compute per-subject baselines; apply population fallback for thin subjects.

    Parameters
    ----------
    records : list of dicts
        From ``find_alert_npz_files``.
    calibration_frames : int
        Number of leading frames to treat as calibration window.

    Returns
    -------
    pd.DataFrame with one row per subject.
    """
    rows = []
    for rec in tqdm(records, desc="Computing head-pose baselines"):
        try:
            data = np.load(rec["npz_path"])
        except Exception as exc:
            print(f"[WARN] Could not load {rec['npz_path']}: {exc}", file=sys.stderr)
            continue

        head_pose      = data["head_pose"]       # (N, 3)
        face_detected  = data["face_detected"]   # (N,)

        baseline = compute_head_baseline(head_pose, face_detected, calibration_frames)
        row = {**rec, **baseline}
        row.pop("npz_path")  # not needed in the output CSV
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No kss_0_alert.npz files found — check --processed_dir path."
        )

    # ------------------------------------------------------------------
    # Population fallback: replace thin-subject baselines with the mean
    # computed across subjects that *did* pass the frame-count threshold.
    # ------------------------------------------------------------------
    good_mask = ~df["fallback_used"]
    if good_mask.sum() == 0:
        print(
            "[WARN] No subject has enough clean calibration frames. "
            "All baselines will use hard-coded defaults.",
            file=sys.stderr,
        )
        return df

    pop_means = {
        col: df.loc[good_mask, col].mean()
        for col in ["mean_pitch", "mean_yaw", "mean_roll",
                    "std_pitch",  "std_yaw",  "std_roll"]
    }

    for col, val in pop_means.items():
        df.loc[df["fallback_used"], col] = val

    fallback_subjects = df.loc[df["fallback_used"], "subject_id"].tolist()
    if fallback_subjects:
        print(
            f"[WARN] {len(fallback_subjects)} subjects used population fallback "
            f"(< {MIN_CALIBRATION_FRAMES} clean frames): "
            + ", ".join(fallback_subjects),
            file=sys.stderr,
        )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="R5: Compute per-driver head-pose calibration baselines from R3 .npz archives.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        required=True,
        help="Path to the root processed_rldd directory (output of R3's preprocess-rldd.py).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./head_baselines.csv",
        help="Destination path for the output CSV file.",
    )
    parser.add_argument(
        "--calibration_seconds",
        type=int,
        default=30,
        help="Length of the calibration window in seconds (1 s = 15 frames at TARGET_FPS=15).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.processed_dir):
        raise FileNotFoundError(
            f"processed_dir not found: {args.processed_dir}\n"
            "Run R3's preprocess-rldd.py first."
        )

    calibration_frames = args.calibration_seconds * TARGET_FPS
    print(
        f"Calibration window: {args.calibration_seconds} s "
        f"= {calibration_frames} frames at {TARGET_FPS} FPS"
    )

    records = find_alert_npz_files(args.processed_dir)
    if not records:
        raise FileNotFoundError(
            f"No kss_0_alert.npz files found under {args.processed_dir}."
        )
    print(f"Found {len(records)} alert sequences across subjects.")

    df = compute_baselines(records, calibration_frames)

    # Reorder columns for readability
    col_order = [
        "fold_id", "subject_id",
        "mean_pitch", "mean_yaw", "mean_roll",
        "std_pitch",  "std_yaw",  "std_roll",
        "n_clean_frames", "fallback_used",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    output_path = args.output_path
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(
        f"\nBaselines saved to: {output_path}\n"
        f"  Subjects processed : {len(df)}\n"
        f"  Fallback used      : {df['fallback_used'].sum()}\n"
        f"\nPopulation summary (degrees):\n"
        f"  pitch  mean={df['mean_pitch'].mean():.2f}°  std_mean={df['std_pitch'].mean():.2f}°\n"
        f"  yaw    mean={df['mean_yaw'].mean():.2f}°   std_mean={df['std_yaw'].mean():.2f}°\n"
        f"  roll   mean={df['mean_roll'].mean():.2f}°  std_mean={df['std_roll'].mean():.2f}°"
    )


if __name__ == "__main__":
    main()
