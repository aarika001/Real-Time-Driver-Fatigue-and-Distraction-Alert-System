# Head Pose & Off-Axis Tests

**Module owner:** R5 (Aaditya)  
**Project:** Real-Time Driver Fatigue and Distraction Alert System (Team NeuroNova)

This module delivers three things:

1. **`head_pose_core.py`** — the single source of truth for head-pose extraction and feature engineering, shared by both the offline evaluation pipeline and the real-time webcam loop (R8).
2. **`calibrate_head_baseline.py`** — offline CLI that computes per-driver neutral head-pose statistics from R3's processed `.npz` archives.
3. **`stress_test_offaxis.py`** — offline CLI that generates synthetic off-axis perspective-warped frames and measures how detection and pose accuracy degrade as camera angle increases.

---

## 1. Prerequisites & Installation

```bash
pip install opencv-python mediapipe numpy pandas tqdm matplotlib
```

The `face_landmarker.task` model (required by `stress_test_offaxis.py`) must be present. It is already downloaded to the repo root by R3; point `--model_path` there.

```bash
# If you need to re-download it:
wget -O ../face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

---

## 2. `head_pose_core.py` — API Reference

Import from any module in the project:

```python
from head_pose_core import (
    extract_head_pose,
    compute_head_baseline,
    compute_head_deviation,
    head_angle_from_neutral,
    build_head_feature_vector,
    apply_baseline_to_sequence,
)
```

### `extract_head_pose(landmarks, frame_w, frame_h) → (pitch, yaw, roll)`

Runs `cv2.solvePnP` + `cv2.RQDecomp3x3` on the 6-point canonical head model.  
**Called by R8's webcam loop** on every live frame. Runs in < 2 ms.

```python
# Inside R8's live loop:
result = landmarker.detect(mp_image)
if result.face_landmarks:
    pitch, yaw, roll = extract_head_pose(result.face_landmarks[0], w, h)
```

### `compute_head_baseline(head_pose_array, face_detected_array, calibration_frames=450) → dict`

Computes mean and std over the calibration window of a subject's alert `.npz`.

```python
data = np.load("kss_0_alert.npz")
baseline = compute_head_baseline(data["head_pose"], data["face_detected"])
# → {"mean_pitch": ..., "std_yaw": ..., "fallback_used": False, ...}
```

### `compute_head_deviation(pitch, yaw, roll, baseline) → (Δpitch, Δyaw, Δroll)`

Z-scores raw angles relative to the driver's personal baseline.

### `head_angle_from_neutral(Δpitch, Δyaw) → float`

Euclidean magnitude in z-score space. **Consumed by R7's reliability estimator** as a scalar distraction/pose quality signal.

### `build_head_feature_vector(pitch, yaw, roll, baseline) → np.ndarray (6,)`

Returns `[pitch, yaw, roll, Δpitch, Δyaw, Δroll]` as `float32`. **Consumed by R6's GRU** as the head-pose slice of the per-frame feature vector.

### `apply_baseline_to_sequence(head_pose_array, face_detected_array, baseline) → np.ndarray (N, 6)`

Vectorised version for processing an entire `.npz` sequence at once. Undetected frames keep `Δ = 0.0` by convention.

---

## 3. `calibrate_head_baseline.py` — CLI

Reads processed `.npz` archives (output of R3's `preprocess-rldd.py`) and computes per-driver neutral head-pose statistics.

```bash
python calibrate_head_baseline.py \
    --processed_dir ./processed_rldd \
    --output_path   ./head_baselines.csv

# Optional: shorter calibration window for the 10 s / 20 s ablation
python calibrate_head_baseline.py \
    --processed_dir ./processed_rldd \
    --output_path   ./head_baselines_20s.csv \
    --calibration_seconds 20
```

### Output schema — `head_baselines.csv`

| Column | Type | Description |
|--------|------|-------------|
| `fold_id` | str | Dataset fold (`Fold1`–`Fold5`) |
| `subject_id` | str | Participant ID |
| `mean_pitch` | float | Mean pitch over calibration window (°) |
| `mean_yaw` | float | Mean yaw (°) |
| `mean_roll` | float | Mean roll (°) |
| `std_pitch` | float | Std pitch (°), floored at 1° |
| `std_yaw` | float | Std yaw (°), floored at 1° |
| `std_roll` | float | Std roll (°), floored at 1° |
| `n_clean_frames` | int | Frames used after masking undetected faces |
| `fallback_used` | bool | `True` if population mean was substituted |

**Sanity check:** A front-facing driver should have `|mean_yaw| < 20°`. Any subject outside this range likely had a tilted camera — worth noting in the per-condition analysis (R3).

---

## 4. `stress_test_offaxis.py` — CLI

Applies projective perspective warps at graded yaw/pitch angles to subjects' stored eye-crop frames, runs fresh MediaPipe detection, and measures degradation.

> **Important:** Run only on held-out subjects (Fold4, Fold5 or as agreed with R3/R1) to avoid contaminating the training evaluation.

```bash
python stress_test_offaxis.py \
    --processed_dir  ./processed_rldd \
    --baselines_csv  ./head_baselines.csv \
    --output_dir     ./stress_results_offaxis \
    --held_out_folds Fold4 Fold5 \
    --model_path     ../face_landmarker.task

# Quick smoke test (150 frames per video):
python stress_test_offaxis.py \
    --processed_dir  ./processed_rldd \
    --baselines_csv  ./head_baselines.csv \
    --output_dir     ./stress_results_offaxis \
    --held_out_folds Fold5 \
    --model_path     ../face_landmarker.task \
    --max_frames_per_video 150
```

### Warp angle sweep

| Axis | Angles tested |
|------|--------------|
| Yaw (horizontal) | 0°, 5°, 10°, 15°, 20°, 30° |
| Pitch (vertical) | 0°, 5°, 10°, 15°, 20°, 30° |

Both positive and negative directions are symmetric by the warp geometry.

### Output: `offaxis_metrics.csv`

| Column | Description |
|--------|-------------|
| `subject_id` | Participant |
| `fold_id` | Dataset fold |
| `yaw_deg` | Simulated yaw offset applied |
| `pitch_deg` | Simulated pitch offset applied |
| `n_frames` | Frames evaluated |
| `face_det_rate` | Fraction of originally-detected frames where MediaPipe still found a face after warping |
| `pose_mae` | Mean absolute error (°) between warped-frame pose and reference pose |
| `distraction_fp_rate` | Fraction of alert frames incorrectly classified as distracted after warping |

### Output: `offaxis_summary.png`

Line plots of `face_det_rate` and `pose_mae` vs. yaw offset angle, with ±1 SE bands across subjects.

**Interpretation guide:**
- `face_det_rate` dropping below 0.6 at a given angle means the system should trigger "monitoring degraded" at that camera offset.
- `pose_mae` > 15° at a given angle means head-pose cues are unreliable at that offset (should be down-weighted by R7).
- `distraction_fp_rate` rising above 0.1 means the threshold or z-score normalization needs adjusting for off-axis cameras.

---

## 5. Running the Unit Tests

```bash
# From the head-pose/ directory:
pip install pytest
python -m pytest tests/ -v
```

Expected output: **14 tests pass** across `test_head_pose_core.py` and `test_stress_warp.py`.

---

## 6. Interface Contracts with Other Roles

| Role | What they need from R5 | How to get it |
|------|------------------------|---------------|
| **R6 (Sejal)** | 6-element head feature vector per frame | `from head_pose_core import build_head_feature_vector` |
| **R7 (Jayesh)** | Scalar head-angle-from-neutral reliability signal | `from head_pose_core import head_angle_from_neutral` |
| **R8 (Dev)** | Live `extract_head_pose()` function (<2 ms) | `from head_pose_core import extract_head_pose` |
| **R3 (Atharva)** | `head_baselines.csv`, `offaxis_metrics.csv` for evaluation tables | Generated by the two CLI scripts |

---

## 7. Week-by-Week Schedule

| Week | Dates | Deliverable |
|------|-------|-------------|
| 2 *(catch-up)* | 19–25 Oct | `head_pose_core.py` with all functions; unit tests passing |
| 3 | 26 Oct–1 Nov | `calibrate_head_baseline.py`; `head_baselines.csv` shared with R6 |
| 5 | 9–15 Nov | `stress_test_offaxis.py`; initial `offaxis_metrics.csv` and summary plot |
| 6 | 16–22 Nov | Stress results integrated into R3's full k-fold evaluation report |
