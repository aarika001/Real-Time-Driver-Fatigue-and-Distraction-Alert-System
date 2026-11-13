# UTA-RLDD Preprocessing & Feature Extraction Pipeline

**Module owner:** R3 (Data Pipeline & Evaluation)  
**Project:** Real-Time Driver Fatigue and Distraction Alert System (Team NeuroNova)

This module converts raw, variable-FPS videos from the **UTA Real-Life Drowsiness
Dataset (UTA-RLDD)** into standardized 15 FPS temporal feature sequences,
CLAHE-normalized eye crops, and per-frame image quality signals. It preserves
strict subject-level and fold-level isolation to prevent data leakage during
model training and cross-validation.

## 1. Prerequisites & Installation

Ensure Python 3.9+ is installed, then install the required computer vision and
scientific computing dependencies:

```bash
pip install opencv-python mediapipe numpy pandas tqdm
```

Download the official MediaPipe Face Landmarker task bundle (478-point 3D facial
mesh) into your project directory:

```bash
wget -O face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## 2. Expected Input Directory Structure

Pass the root RLDD folder containing the extracted fold archives
(`Fold1_part1`, `Fold1_part2`, ..., `Fold5_part2`) via `--rldd_dir`. The
recursive file walker automatically handles nested extraction folders (such as
`Fold3_part2-005/Fold3_part2/`) and mixed video extensions (`.mp4`, `.mov`,
`.m4v`, `.avi`).

```text
<RLDD_ROOT_DIR>/
├── Fold1_part1/
│   ├── 01/                     # Subject ID (parsed from parent folder name)
│   │   ├── 0.mp4               # KSS 0  -> Alert
│   │   ├── 5.mp4               # KSS 5  -> Low Vigilant
│   │   └── 10.mp4              # KSS 10 -> Fatigue
│   └── ...
├── Fold1_part2/
├── ...
└── Fold3_part2/
    └── 31/
        ├── 0.mp4
        ├── 5.mp4
        └── 10.mp4
```

### Label Mapping Convention

Because UTA-RLDD encodes ground truth at the video level via Karolinska
Sleepiness Scale (KSS) filenames rather than external CSVs, filenames are
mapped as follows:

| Filename | Label | Description |
| --- | --- | --- |
| `0.*` | `alert` | Naturally alert baseline driving/sitting |
| `5.*` | `low_vigilant` | Early drowsiness; optionally excluded via `--exclude_low_vigilant` |
| `10.*` | `fatigue` | Pronounced drowsiness, heavy eye closures, yawning |

## 3. Command-Line Usage

Run `preprocess_rldd.py` by passing the path to your RLDD root directory:

```bash
python preprocess_rldd.py \
  --rldd_dir ~/ai-proj/RLDD \
  --output_dir ./processed_rldd \
  --model_path ./face_landmarker.task
```

### CLI Arguments

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--rldd_dir` | Yes | — | Path to the root RLDD directory containing `Fold1_part1`, `Fold3_part2`, etc. |
| `--output_dir` | No | `./processed_rldd` | Target directory where `.npz` feature archives and `rldd_metadata_index.csv` are saved. |
| `--model_path` | No | `./face_landmarker.task` | Path to the downloaded `face_landmarker.task` binary. |
| `--target_fps` | No | `15` | Target uniform sampling rate (matches the real-time webcam inference budget). |
| `--exclude_low_vigilant` | No | `False` | Flag to skip `5.mp4` (`low_vigilant`) videos when training a strict alert-vs-fatigue split. |

## 4. Pipeline Processing Stages

For each discovered video, the pipeline executes five deterministic steps:

1. **Temporal Downsampling (15 FPS):** Reads the video's native frame rate
   (`CAP_PROP_FPS`) and samples frames at uniform intervals
   (`round(orig_fps / target_fps)`) so all sequences match the live webcam
   inference rate, regardless of whether the participant recorded at 30 FPS or
   60 FPS.
2. **478-Point Facial Landmark Detection:** Runs MediaPipe FaceLandmarker in
   single-face mode on each sampled RGB frame. If a face is not detected (for
   example, because of severe occlusion, an extreme off-axis angle, or
   darkness), a zero-padded frame is logged with `face_detected = 0` so
   downstream modules can track silent-failure and degraded-monitoring rates.
3. **Geometric Cue Extraction:**
   - **Mouth Aspect Ratio (MAR):** Computed from eight inner/outer lip landmarks
     (`[61, 291, 39, 181, 0, 17, 269, 405]`) as the ratio of average vertical
     lip opening to horizontal mouth width.
   - **3D Head Pose Estimation:** Estimates pitch, yaw, and roll in degrees using
     `cv2.solvePnP` and `cv2.RQDecomp3x3` by mapping six canonical 2D facial
     landmarks (nose tip, chin, eye corners, and mouth corners) against a
     standard 3D head model.
4. **Eye-Region Quality Metrics (for R7 Reliability Estimator):** Before
   contrast normalization, raw grayscale eye bounding boxes (padded by 35%) are
   evaluated for four visual degradation signals averaged across both eyes:
   - **Brightness:** Mean pixel intensity normalized to `[0, 1]`.
   - **Contrast:** Pixel intensity standard deviation normalized to `[0, 1]`.
   - **Glare fraction:** Proportion of saturated pixels (`>= 245` intensity),
     capturing lens or spectacle reflections.
   - **Sharpness:** Variance of the 2D Laplacian operator, capturing motion or
     focus blur.
5. **IR-RGB Invariant Eye Cropping (for R4 Eye-State CNN):** Eye regions are
   resized to `64 x 64` and passed through CLAHE (Contrast Limited Adaptive
   Histogram Equalization, `clipLimit=2.0`, `tileGridSize=(4, 4)`) to minimize
   domain shift between grayscale/RGB webcams (UTA-RLDD, YawDD, DMD) and
   infrared datasets (MRL Eye, NTHU-DDD).

## 5. Output Data Schema

### A. Master Index (`rldd_metadata_index.csv`)

A tabular summary of every processed video used for dataset auditing, stratified
condition reporting, and constructing subject-independent splits:

| Column | Type | Description |
| --- | --- | --- |
| `fold_id` | `str` | Official dataset fold (`Fold1` through `Fold5`), parsed from the directory path. |
| `subject_id` | `str` | Unique participant identifier (for example, `31`), parsed from the parent folder name. |
| `raw_kss` | `str` | Original KSS filename stem (`0`, `5`, or `10`). |
| `label` | `str` | Mapped system class (`alert`, `low_vigilant`, or `fatigue`). |
| `video_path` | `str` | Full path to the source video file. |
| `npz_path` | `str` | Full path to the generated compressed `.npz` feature archive. |
| `num_frames_15fps` | `int` | Total number of sampled frames (`N`) in the sequence. |
| `face_detection_rate` | `float` | Fraction of frames (`[0.0, 1.0]`) where MediaPipe successfully localized the face. |

### B. Per-Video Feature Archives

Archives are stored as `FoldX/subject_Y/kss_<raw>_<label>.npz`.

Each compressed NumPy archive stores synchronized arrays of length `N`
(`num_frames_15fps`):

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `mar` | `(N,)` | `float32` | Per-frame Mouth Aspect Ratio. |
| `head_pose` | `(N, 3)` | `float32` | Per-frame head angles `[pitch, yaw, roll]` in degrees. |
| `quality` | `(N, 4)` | `float32` | Per-frame eye-region quality `[brightness, contrast, glare_frac, sharpness]`. |
| `face_detected` | `(N,)` | `uint8` | Binary detection flag (`1 = face detected`, `0 = face lost`). |
| `left_eye_crops` | `(N, 64, 64)` | `uint8` | CLAHE-normalized grayscale crops of the subject's left eye. |
| `right_eye_crops` | `(N, 64, 64)` | `uint8` | CLAHE-normalized grayscale crops of the subject's right eye. |

## 6. Downstream Team Hand-off Notes

- **Per-Driver 30-Second Calibration (R6):** For each subject, extract the
  first 450 frames (`[:450]`, equivalent to 30 s × 15 FPS) from their alert
  (`kss_0_alert.npz`) sequence to compute personal baseline means and standard
  deviations for `mar` and `head_pose`. Express subsequent features relative to
  this baseline.
- **5-Second Temporal Windowing (R3 / R6):** Slice post-calibration frames
  (`[450:]`) into 75-frame windows (5 s at 15 FPS) using a sliding stride (for
  example, 15 frames / 1 second). Never concatenate frames across different
  `.npz` files or `subject_id` values.
- **Subject-Independent & Cross-Dataset Splits (R3):** Group all `.npz` windows
  by `fold_id` (or `subject_id` via `GroupKFold`) so no participant appears in
  both training and test folds. For cross-dataset generalization experiments,
  hold out all Fold1–Fold5 UTA-RLDD archives as an unseen test set against
  models trained on MRL Eye and NTHU-DDD.
