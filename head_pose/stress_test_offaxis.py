"""
stress_test_offaxis.py
======================
R5 — Head Pose & Off-Axis Tests (Aaditya)
Team NeuroNova · Real-Time Driver Fatigue and Distraction Alert System

Generates synthetic off-axis perspective-warped frames from subjects' stored
eye-crop and head-pose data, then measures how much the head-pose estimation
and landmark detection degrade as the simulated camera angle increases.

Why synthetic warps instead of real off-axis video?
----------------------------------------------------
  UTA-RLDD videos were self-recorded on subjects' own phones/webcams with
  uncontrolled camera placement; we cannot control off-axis angle in the data.
  Instead we apply a projective homography that approximates what the scene
  would look like if the camera were rotated by ±yaw° or ±pitch° relative to
  the driver.  This is the same approach used in prior driver-monitoring work
  (see project proposal, Section 3.1, synthetic stress tests).

Pipeline per frame
------------------
  raw frame → perspective_warp(yaw_deg, pitch_deg)
            → MediaPipe FaceLandmarker
            → extract_head_pose()
            → compare vs. reference pose (from unwarped frame)

Outputs
-------
  <output_dir>/offaxis_metrics.csv
      Per-subject, per-angle: face_det_rate, pose_mae, distraction_fp_rate.

  <output_dir>/offaxis_summary.png
      Line plots of detection rate and pose MAE vs. angle, ±1 SE across subjects.

Usage
-----
  python stress_test_offaxis.py \\
      --processed_dir  ./processed_rldd \\
      --baselines_csv  ./head_baselines.csv \\
      --output_dir     ./stress_results_offaxis \\
      --held_out_folds Fold4 Fold5

  # Quick smoke test on a single fold with fewer frames per video:
  python stress_test_offaxis.py \\
      --processed_dir  ./processed_rldd \\
      --baselines_csv  ./head_baselines.csv \\
      --output_dir     ./stress_results_offaxis \\
      --held_out_folds Fold5 \\
      --max_frames_per_video 150

Important: this script requires a live webcam-style inference loop using
MediaPipe FaceLandmarker because we need fresh landmark detection on the
*warped* frames (not the pre-extracted landmarks from R3, which were
computed from the original frames).  The face_landmarker.task model must
be present in the directory specified by --model_path.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from head_pose_core import extract_head_pose, head_angle_from_neutral, compute_head_deviation

# ---------------------------------------------------------------------------
# Sweep grid: yaw and pitch angles to test (degrees)
# Zero is always included as the control / identity warp.
# ---------------------------------------------------------------------------
YAW_SWEEP_DEG:   list[float] = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0]
PITCH_SWEEP_DEG: list[float] = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0]

# Distraction threshold: |Δyaw| > this many personal std-devs → distraction flag
DISTRACTION_ZSCORE_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# Perspective-warp helpers
# ---------------------------------------------------------------------------

def perspective_warp_frame(
    frame: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
) -> np.ndarray:
    """Apply a projective homography that simulates camera rotation.

    The warp maps the four corners of the input frame to new positions that
    approximate the view from a camera rotated by ``yaw_deg`` horizontally
    and ``pitch_deg`` vertically.  The displacement of each corner is
    proportional to its distance from the vanishing point and to the sine of
    the rotation angle.

    Parameters
    ----------
    frame : np.ndarray, shape (H, W, C) or (H, W)
        Input BGR or grayscale frame.
    yaw_deg : float
        Horizontal rotation of the simulated camera in degrees.
        Positive = camera rotates right → face appears shifted left in frame.
    pitch_deg : float
        Vertical rotation in degrees.
        Positive = camera tilts up → face appears lower in frame.

    Returns
    -------
    np.ndarray
        Warped frame of the same spatial size as the input.
        Pixels outside the original frame boundary are filled with black.
    """
    H, W = frame.shape[:2]

    yaw_rad   = np.deg2rad(yaw_deg)
    pitch_rad = np.deg2rad(pitch_deg)

    # Horizontal shift fraction per column (perspective foreshortening)
    dx = np.sin(yaw_rad) * W * 0.5
    dy = np.sin(pitch_rad) * H * 0.5

    # Source corners: top-left, top-right, bottom-right, bottom-left
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])

    # Destination corners: apply perspective shift
    # For a rightward yaw the left side of the frame contracts and the right
    # side expands (barrel-like foreshortening).
    dst = np.float32([
        [dx,       dy      ],   # top-left
        [W - dx,   dy      ],   # top-right
        [W - dx,   H - dy  ],   # bottom-right
        [dx,       H - dy  ],   # bottom-left
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (W, H), borderValue=0)


# ---------------------------------------------------------------------------
# Per-video stress evaluation
# ---------------------------------------------------------------------------

def evaluate_video_at_angle(
    alert_npz_path: str,
    baseline: dict,
    yaw_deg: float,
    pitch_deg: float,
    landmarker: mp_vision.FaceLandmarker,
    max_frames: Optional[int],
    skip_calibration_frames: int = 450,
) -> dict:
    """Warp frames from a subject's alert sequence and measure degradation.

    Parameters
    ----------
    alert_npz_path :
        Path to the subject's ``kss_0_alert.npz`` produced by R3.
    baseline : dict
        Personal head-pose baseline from ``head_baselines.csv``.
    yaw_deg, pitch_deg : float
        Warp angles to apply.
    landmarker :
        Shared MediaPipe FaceLandmarker instance.
    max_frames : int | None
        Cap on frames to process (None = all).  Use for smoke tests.
    skip_calibration_frames : int
        Skip the first N frames so the calibration window itself is not
        part of the test population (avoid data leakage).

    Returns
    -------
    dict with keys: n_frames, face_det_rate, pose_mae, distraction_fp_rate
    """
    data = np.load(alert_npz_path)
    head_pose_ref  = data["head_pose"][skip_calibration_frames:]     # (M, 3)
    face_det_ref   = data["face_detected"][skip_calibration_frames:] # (M,)
    left_crops     = data["left_eye_crops"][skip_calibration_frames:]
    right_crops    = data["right_eye_crops"][skip_calibration_frames:]

    if max_frames is not None:
        head_pose_ref = head_pose_ref[:max_frames]
        face_det_ref  = face_det_ref[:max_frames]
        left_crops    = left_crops[:max_frames]
        right_crops   = right_crops[:max_frames]

    n_frames = len(face_det_ref)
    if n_frames == 0:
        return {
            "n_frames": 0,
            "face_det_rate": 0.0,
            "pose_mae": float("nan"),
            "distraction_fp_rate": float("nan"),
        }

    n_detected = 0
    pose_errors = []
    distraction_flags = []

    for i in range(n_frames):
        # Skip frames where R3 already lost the face — the warp can only
        # make detection harder, so ground truth is "face not visible."
        if not face_det_ref[i]:
            continue

        # Reconstruct a synthetic "frame" from the saved eye crops.
        # We tile the two 64×64 crops side-by-side into a 64×128 RGB image
        # and then apply the perspective warp to that miniature frame.
        # NOTE: eye crops are small (64×64) so pose estimation from them
        # will be noisier than from the full frame; the *relative* change
        # in detection rate and pose error across angles is still meaningful.
        eye_tile = np.stack(
            [left_crops[i], right_crops[i]], axis=1
        ).reshape(64, 128)                              # (64, 128) gray
        rgb_tile = cv2.cvtColor(eye_tile, cv2.COLOR_GRAY2RGB)

        warped = perspective_warp_frame(rgb_tile, yaw_deg, pitch_deg)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=warped)
        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            continue

        n_detected += 1
        h, w = warped.shape[:2]
        lms = result.face_landmarks[0]
        pitch_w, yaw_w, roll_w = extract_head_pose(lms, w, h)

        # Reference pose (from R3 — extracted on the full un-warped frame)
        pitch_ref, yaw_ref, roll_ref = head_pose_ref[i]

        # Mean absolute error across the three axes
        mae = float(
            (abs(pitch_w - pitch_ref) + abs(yaw_w - yaw_ref) + abs(roll_w - roll_ref)) / 3.0
        )
        pose_errors.append(mae)

        # Distraction false-positive: alert frame flagged as distracted
        delta_pitch, delta_yaw, _ = compute_head_deviation(
            pitch_w, yaw_w, roll_w, baseline
        )
        angle_from_neutral = head_angle_from_neutral(delta_pitch, delta_yaw)
        distraction_flags.append(int(angle_from_neutral > DISTRACTION_ZSCORE_THRESHOLD))

    n_valid_ref = int(face_det_ref.sum())
    face_det_rate       = n_detected / n_valid_ref if n_valid_ref > 0 else 0.0
    pose_mae            = float(np.mean(pose_errors)) if pose_errors else float("nan")
    distraction_fp_rate = float(np.mean(distraction_flags)) if distraction_flags else float("nan")

    return {
        "n_frames":            n_frames,
        "face_det_rate":       round(face_det_rate, 4),
        "pose_mae":            round(pose_mae, 4) if not np.isnan(pose_mae) else float("nan"),
        "distraction_fp_rate": round(distraction_fp_rate, 4) if not np.isnan(distraction_fp_rate) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Summary plots
# ---------------------------------------------------------------------------

def save_summary_plots(metrics_df: pd.DataFrame, output_dir: str) -> None:
    """Generate line plots of detection rate and pose MAE vs. warp angle."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping summary plots.", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Off-Axis Stress Test: Head Pose Degradation vs. Warp Angle", fontsize=13)

    for ax, metric, ylabel, title in [
        (axes[0], "face_det_rate",       "Face Detection Rate",        "Detection Rate vs. Yaw Warp"),
        (axes[1], "pose_mae",            "Pose MAE (degrees)",         "Pose MAE vs. Yaw Warp"),
    ]:
        grouped = (
            metrics_df[metrics_df["pitch_deg"] == 0.0]
            .groupby("yaw_deg")[metric]
        )
        means  = grouped.mean()
        sems   = grouped.sem()

        ax.plot(means.index, means.values, marker="o", label="Yaw sweep (pitch=0°)")
        ax.fill_between(
            means.index,
            means.values - sems.values,
            means.values + sems.values,
            alpha=0.25,
            label="±1 SE",
        )
        ax.set_xlabel("Simulated Yaw Offset (degrees)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "offaxis_summary.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Summary plot saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="R5: Synthetic off-axis perspective-warp stress tests for head-pose estimation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        required=True,
        help="Root of R3's processed_rldd output directory.",
    )
    parser.add_argument(
        "--baselines_csv",
        type=str,
        required=True,
        help="Path to head_baselines.csv produced by calibrate_head_baseline.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./stress_results_offaxis",
        help="Directory for offaxis_metrics.csv and offaxis_summary.png.",
    )
    parser.add_argument(
        "--held_out_folds",
        type=str,
        nargs="+",
        default=["Fold5"],
        help="Fold IDs to include in the stress test (held-out subjects only).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../face_landmarker.task",
        help="Path to the MediaPipe face_landmarker.task binary.",
    )
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=None,
        help="Cap frames per video for a quick smoke test (None = all).",
    )
    parser.add_argument(
        "--calibration_seconds",
        type=int,
        default=30,
        help="Frames to skip at the start of each video (calibration window).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not os.path.isdir(args.processed_dir):
        raise FileNotFoundError(f"processed_dir not found: {args.processed_dir}")
    if not os.path.isfile(args.baselines_csv):
        raise FileNotFoundError(
            f"baselines_csv not found: {args.baselines_csv}\n"
            "Run calibrate_head_baseline.py first."
        )
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(
            f"MediaPipe model not found: {args.model_path}\n"
            "Download face_landmarker.task from the MediaPipe releases page."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    skip_frames = args.calibration_seconds * 15  # 15 FPS

    # ------------------------------------------------------------------
    # Load baselines
    # ------------------------------------------------------------------
    baselines_df = pd.read_csv(args.baselines_csv)
    baselines_df["subject_id"] = baselines_df["subject_id"].astype(str)

    # ------------------------------------------------------------------
    # Discover held-out alert sequences
    # ------------------------------------------------------------------
    all_records: list[dict] = []
    for dirpath, _, filenames in os.walk(args.processed_dir):
        for fname in filenames:
            if fname != "kss_0_alert.npz":
                continue
            full_path = os.path.join(dirpath, fname)
            parts = Path(full_path).parts
            fold_id = next(
                (p.split("_")[0] for p in parts if p.startswith("Fold")), "Unknown"
            )
            if fold_id not in args.held_out_folds:
                continue
            subject_id = next(
                (p.replace("subject_", "") for p in parts if p.startswith("subject_")),
                "unknown",
            )
            all_records.append(
                {"subject_id": subject_id, "fold_id": fold_id, "npz_path": full_path}
            )

    if not all_records:
        raise FileNotFoundError(
            f"No kss_0_alert.npz files found in held-out folds: {args.held_out_folds}"
        )

    print(
        f"Stress testing {len(all_records)} subjects from folds: {args.held_out_folds}\n"
        f"Yaw sweep:   {YAW_SWEEP_DEG}\n"
        f"Pitch sweep: {PITCH_SWEEP_DEG}"
    )

    # ------------------------------------------------------------------
    # Build MediaPipe landmarker
    # ------------------------------------------------------------------
    base_options = mp_python.BaseOptions(model_asset_path=args.model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )

    results: list[dict] = []
    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        for rec in tqdm(all_records, desc="Subjects"):
            subject_id = rec["subject_id"]
            fold_id    = rec["fold_id"]

            # Fetch this subject's personal baseline
            row = baselines_df[baselines_df["subject_id"] == subject_id]
            if row.empty:
                print(
                    f"[WARN] No baseline for subject {subject_id} — skipping.",
                    file=sys.stderr,
                )
                continue
            baseline = row.iloc[0].to_dict()

            # Sweep over yaw angles (pitch fixed at 0)
            for yaw_deg in tqdm(YAW_SWEEP_DEG, desc=f"  yaw sweep (subj {subject_id})", leave=False):
                metrics = evaluate_video_at_angle(
                    rec["npz_path"],
                    baseline,
                    yaw_deg=yaw_deg,
                    pitch_deg=0.0,
                    landmarker=landmarker,
                    max_frames=args.max_frames_per_video,
                    skip_calibration_frames=skip_frames,
                )
                results.append({
                    "subject_id": subject_id,
                    "fold_id":    fold_id,
                    "yaw_deg":    yaw_deg,
                    "pitch_deg":  0.0,
                    **metrics,
                })

            # Sweep over pitch angles (yaw fixed at 0)
            for pitch_deg in tqdm(PITCH_SWEEP_DEG[1:], desc=f"  pitch sweep (subj {subject_id})", leave=False):
                metrics = evaluate_video_at_angle(
                    rec["npz_path"],
                    baseline,
                    yaw_deg=0.0,
                    pitch_deg=pitch_deg,
                    landmarker=landmarker,
                    max_frames=args.max_frames_per_video,
                    skip_calibration_frames=skip_frames,
                )
                results.append({
                    "subject_id": subject_id,
                    "fold_id":    fold_id,
                    "yaw_deg":    0.0,
                    "pitch_deg":  pitch_deg,
                    **metrics,
                })

    metrics_df = pd.DataFrame(results)
    out_csv = os.path.join(args.output_dir, "offaxis_metrics.csv")
    metrics_df.to_csv(out_csv, index=False)
    print(f"\nMetrics saved to: {out_csv}")

    # Print a quick summary table
    summary = (
        metrics_df[metrics_df["pitch_deg"] == 0.0]
        .groupby("yaw_deg")[["face_det_rate", "pose_mae", "distraction_fp_rate"]]
        .mean()
        .round(4)
    )
    print("\nYaw sweep summary (pitch=0°, mean across subjects):")
    print(summary.to_string())

    save_summary_plots(metrics_df, args.output_dir)


if __name__ == "__main__":
    main()
