# NeuroNova R6 – Temporal Model & Calibration

This package is a **trainable R6 implementation**, designed around the R3 preprocessing output supplied by the team.

## What it implements

1. **30-second per-driver neutral-pose calibration**
   - Reads the R3 `.npz` files and `rldd_metadata_index.csv`.
   - Uses only recordings labelled `alert` (KSS 0).
   - At 15 FPS, 30 seconds corresponds to 450 sampled frames.
   - Uses only frames where `face_detected == 1`.
   - Writes `driver_baselines.csv` with the exact R5 handoff fields:
     `subject_id, mean_pitch, mean_yaw, std_pitch`
   - Also writes useful audit columns (`std_yaw`, number of frames, seconds, source clips).
   - `mean_pitch` and `mean_yaw` are the neutral pose; R5 computes current-minus-neutral deviations.

2. **R6 temporal model**
   - One-layer GRU with 64 hidden units.
   - 75-frame windows (5 seconds at 15 FPS), configurable stride.
   - Subject-wise train/validation/test splits to avoid frame leakage.
   - Cue dropout during training.
   - Temperature scaling fitted on validation logits.
   - Saves model weights, feature scaler, temperature, label mapping, feature schema, metrics and test predictions.

3. **Inference helper**
   - Loads the saved model package and predicts a class/probability distribution from a 75-frame feature window.

## Important dataset finding: the NTHU ZIP uploaded in this chat

The archive `1 nthu.zip` was inspected. It contains **2,322 JPG files**, all under one folder, all filenames ending in `_drowsy.jpg`, and all carrying subject prefix `001`. It appears to be a single-subject drowsy image collection, not a labelled video/sequence dataset. Therefore:

- It is **not sufficient by itself to train/evaluate the 3-class temporal GRU**.
- It does not provide 30 seconds of normal driving, so it cannot generate a valid neutral-pose baseline.
- It may be useful for an eye/face visual cue model after R4/R3 agree on the crop/label convention.
- Do not treat the ordered filenames as a verified time sequence unless the dataset documentation confirms that interpretation.

The R6 trainer below is intended for R3 `.npz` temporal features, not raw JPGs. You need normal/alert, low-vigilant and fatigue labelled recordings from multiple subjects for a meaningful subject-independent evaluation. Distraction training additionally requires distraction-labelled data; the three labels in the supplied R3 preprocessing script do not include a distraction class.

## Expected R3 processed data

The supplied preprocessing script saves one `.npz` per video with these arrays:

- `mar`: `(frames,)`
- `head_pose`: `(frames, 3)` in `[pitch, yaw, roll]` order
- `quality`: `(frames, 4)` in `[brightness, contrast, glare_fraction, sharpness]` order
- `face_detected`: `(frames,)`
- `left_eye_crops`, `right_eye_crops`: `(frames, 64, 64)`

The metadata CSV contains at least:
`subject_id, label, npz_path`

This package reads those exact fields. It also supports optional per-frame arrays when R4/R7 add them consistently to every file:
- `ear` or `eye_closure` (one scalar per frame)
- `reliability` (one scalar per frame)

If optional arrays are absent, they are not silently fabricated; the model uses the available R3 features. Coordinate the final feature list with R3/R4/R5/R7 before comparing model results across versions.

## Setup (Google Colab or Kaggle)

1. Upload/extract the processed R3 folder (the folder containing `rldd_metadata_index.csv`) to your notebook storage.
2. The supplied MediaPipe asset is included at `assets/face_landmarker.task`. If running the R3 preprocessor, pass `--model_path assets/face_landmarker.task`. It is not needed by the R6 trainer once `.npz` files exist. The original R3 script supplied in the chat is preserved under `r3_reference/` for interface reference.
3. Install:
   ```bash
   pip install -r requirements.txt
   ```
4. Run baseline calibration:
   ```bash
   python -m r6.baseline --metadata processed_rldd/rldd_metadata_index.csv --output outputs
   ```
5. Train:
   ```bash
   python -m r6.train --metadata processed_rldd/rldd_metadata_index.csv --baselines outputs/driver_baselines.csv --output outputs --fps 15 --window 75 --stride 15 --epochs 40
   ```

In Colab/Kaggle, set the paths to the mounted dataset, e.g. `/content/processed_rldd/rldd_metadata_index.csv` or `/kaggle/input/<dataset-folder>/processed_rldd/rldd_metadata_index.csv`.

## Neutral-pose CSV contract for R5

Required columns:
```csv
subject_id,mean_pitch,mean_yaw,std_pitch
001,....,....,...
```

- Angles use the exact convention produced by the supplied R3 `compute_head_pose()` implementation (`cv2.RQDecomp3x3` output).
- Do not mix degrees/radians or change axis conventions between R3, R5 and R6.
- R5 computes:
  `delta_pitch = pitch_current - mean_pitch`
  `delta_yaw = yaw_current - mean_yaw`
- `std_pitch` is the sample standard deviation of valid neutral pitch measurements.
- Baseline generation requires at least 30 seconds of **valid face-detected alert/normal frames** for a subject. Incomplete subjects are reported and omitted by default rather than inventing a baseline.

## Outputs after a successful training run

`outputs/` will contain:
- `driver_baselines.csv`
- `r6_gru.pt`
- `r6_preprocessing.joblib`
- `r6_model_config.json`
- `r6_temperature.json`
- `r6_metrics.json`
- `r6_training_summary.json`
- `r6_test_predictions.csv`
- `r6_confusion_matrix.csv`
- `r6_classification_report.json`

These are the artefacts to share with R5 (baseline CSV), R7/R8 (model and inference contract), and R1 (metrics/report evidence).

## Training validity checks

The trainer refuses to produce a misleading result if:
- fewer than 3 unique subjects are available for group-wise train/validation/test splitting;
- the split leaves a class absent from the training partition;
- no valid windows can be built;
- a subject has no matching baseline.

For a first smoke test, use `--allow-incomplete-baselines` only if you understand that missing-baseline subjects are excluded from the temporal training set. Do not use a random frame split.

## Notes

- A trained checkpoint is not evidence of useful performance unless the dataset has suitable labels, multiple subjects, and a held-out subject test set.
- This package does not train R4's eye CNN, R5's pose estimator, R7's reliability estimator, or R8's application.
- The R3 script's current `head_pose` values are absolute angles. This package adds neutral-relative pitch/yaw using the CSV.
