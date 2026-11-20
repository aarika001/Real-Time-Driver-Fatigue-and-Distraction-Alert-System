# `cues/` — Eye and mouth cue models (R4)

**Owner:** R4 (Dhwaj). Converts MediaPipe landmarks + a grayscale frame into
per-frame eye and mouth cues for R3 (evaluation), R6 (calibration + GRU),
R7 (reliability) and R8 (live app).

## Status

| Milestone | Module | Status |
|---|---|---|
| M1 | `landmarks.py`, `eye_crop.py`, `schemas.py` (interface contract) | done |
| M2 | `mouth_cue.py` (MAR, per-driver baseline, yawn features) + YawDD scripts | done |
| M3 | `eye_dataset.py`, `augment.py`, `eye_metrics.py` (loaders, augmentation, subject splits, metrics) | done |
| M4 | `eye_model.py`, `scripts/r4/train_eye_cnn.py`, Kaggle notebook (MobileNetV2) | done |
| M5 | `scripts/r4/eval_eye_cnn.py`, `perclos.py`, `stress.py` (per-condition, stress, cross-dataset, RLDD + baseline B2) | done |
| M6 | `inference_r4.py`, `predictors.py`, `scripts/r4/export_onnx.py`, `scripts/r4/webcam_demo.py` | done |

## Compatibility with R3's processed RLDD data

`eye_crop.crop_eye` reproduces `extract_eye_crop_and_quality` in
`RLDD-preprocess/preprocess-rldd.py` exactly (64×64, 35% padding, CLAHE 2.0 / 4×4),
and the eye/mouth landmark indices are the same. `tests/r4/test_contract.py`
checks this, so crops in `processed_rldd/*.npz` and live crops are identical.

## Per-frame output (`schemas.R4FrameOutput.to_flat_dict()`)

Keys, in this fixed order (`schemas.FRAME_FEATURE_KEYS`):

| Key | Meaning | Range |
|---|---|---|
| `face_detected` | face found this frame | 0 / 1 |
| `p_closed_left`, `p_closed_right` | CNN P(eye closed), image-left / image-right eye | [0, 1] |
| `p_closed` | mean over valid eyes | [0, 1] |
| `ear` | six-point eye aspect ratio, mean of both eyes | ~0.05–0.4 |
| `mar` | outer-lip MAR (same formula as RLDD archives) | ~0.2–1.0 |
| `mar_inner` | inner-lip MAR (true opening) | ~0–1.0 |
| `mar_norm` | `mar` ÷ driver's neutral MAR from calibration | ~1 at rest |
| `mouth_open_run_s` | seconds the mouth has been continuously wide open | ≥ 0 |
| `mouth_osc_hz` | open/close cycles per second over the last 2 s | talking ~2–4 |
| `yawn_flag` | rule-based yawn detector firing | 0 / 1 |
| `eye_brightness`, `eye_contrast`, `eye_glare_frac`, `eye_sharpness` | raw-crop quality, mean over valid eyes (for R7) | see `EyeQuality` |

**Missing values are NaN, not 0.** Check `face_detected` before using cues.
(The RLDD `.npz` archives store 0 for lost frames, so mask them with their
`face_detected` array.)

"Image-left" eye = the eye on the left of the un-mirrored frame = the driver's
anatomical right eye.

## Mouth cue (M2)

No training: MAR is geometry and the yawn rule is threshold + duration.

```python
from cues.mouth_cue import MouthCueExtractor
mouth = MouthCueExtractor()                    # default thresholds (see configs/r4.yaml)
neutral = [mouth.update(lm).mar for lm in first_30s_landmarks]   # R6 calibration stage
mouth.calibrate(neutral)
result = mouth.update(landmarks_px)            # every frame; pass None if no face
```

* `mar` = outer-lip 2-D MAR (same as RLDD archives); `mar_inner` = inner-lip 3-D MAR (robust to head turns).
* `mar_norm` = `mar` ÷ driver's neutral MAR: ~1 at rest, ~2–3 in a yawn.
* `mouth_open_run_s` long + `mouth_osc_hz` low → yawn; short + high → talking.
* For RLDD archives: `cues.mouth_cue.load_rldd_mar(npz)` + `yawn_features(mar, neutral, cfg)`.

YawDD evaluation:

```bash
python scripts/r4/extract_yawdd_mar.py --videos_dir <YawDD root> --out_dir data/yawdd_mar
python scripts/r4/eval_yawdd.py --features_dir data/yawdd_mar --annotations <R2 yawns.csv> --sweep
```

Annotation CSV columns: `video_id,start_s,end_s` (video file name without extension).

## Eye-state CNN (M3–M4)

* Data: MRL Eye (84,898 IR crops, 37 people; labels in file names) + optional extra labelled
  crops via a CSV with `path, subject_id, label_closed` (+ `glasses`, `reflection`, `lighting`, `sensor`).
* Splits are by person: 25 train / 5 val / 7 test people for MRL (`splits.csv` is saved with every run,
  `group_kfold_assign` gives R3 person-level folds).
* Per image: resize to 64×64 → augmentation (training only) → CLAHE → MobileNetV2.
  Augmentations run before CLAHE because real degradations happen in the camera.
* Training: ImageNet MobileNetV2, 1 head-only epoch, AdamW + warm-up + cosine LR, class-balanced
  BCE, mixed precision on GPU, early stopping on validation AUROC, temperature scaling on the validation
  people so P(closed) is calibrated.

```bash
python scripts/r4/train_eye_cnn.py --mrl_root <MRL folder or zip> --out_dir runs/eye_cnn          # GPU
python scripts/r4/train_eye_cnn.py --mrl_root <MRL folder or zip> --out_dir runs/smoke --smoke    # 1-min check
```

Kaggle: `notebooks/r4_train_eye_kaggle.ipynb` clones this repo and runs tests + training.

```python
from cues.eye_model import EyeStatePredictor
pred = EyeStatePredictor("runs/eye_cnn/best.pt")        # checkpoint is shared via Drive, not git
p_closed = pred.predict_crops(np.stack([left_crop, right_crop]))
```

## Evaluation (M5)

```bash
python scripts/r4/eval_eye_cnn.py --eye_model runs/eye_cnn/eye_cnn.onnx --splits runs/eye_cnn/splits.csv \
    --mrl_root <MRL folder or zip> [--extra_index rgb_crops.csv] [--rldd_dir data/processed_rldd] \
    --out_dir results/r4/eye_eval
```

Writes `report.md`: per-condition results on unseen people, graded stress tests
(low light, glare, blur, occlusion, noise), cross-dataset results, and on UTA-RLDD the
per-video PERCLOS with Baseline B2 (CNN + PERCLOS rule).

## Live pipeline for R8 (M6)

```python
from cues.inference_r4 import R4Pipeline
from cues.predictors import load_eye_predictor
pipe = R4Pipeline(eye_predictor=load_eye_predictor("eye_cnn.onnx"))   # .onnx or .pt
pipe.start_calibration(); ...; pipe.finish_calibration()             # R6 calibration stage
res = pipe.process_frame(frame_bgr, timestamp_s)
features = res.output.to_flat_dict()      # timings in res.timings_ms
```

```bash
python scripts/r4/export_onnx.py --checkpoint runs/eye_cnn/best.pt --out runs/eye_cnn/eye_cnn.onnx
python scripts/r4/webcam_demo.py --eye_model runs/eye_cnn/eye_cnn.onnx     # c = calibrate, q = quit
```

The webcam script prints per-stage latency and FPS and saves them to `results/r4/fps_report.json`.

## DMD evaluation (R2 protocol)

The team evaluates on UTA-RLDD and DMD (R2 decision log). `scripts/r4/eval_dmd.py` runs the full
live pipeline on R2's DMD drowsiness face videos (RGB, never used for training) and reports:
eye closure per frame (`eyes_state/close` vs `open`), yawn recall and false alarms per hour on the
merged yawn intervals (with-hand yawns separately), and paired changes on R2's 456-image stress set.

```bash
python scripts/r4/eval_dmd.py --eye_model runs/eye_cnn/eye_cnn.onnx \
    --videos_dir <NeuroNova_data>/dmd/dmd_video_v1 --events <NeuroNova_data>/dmd/r2_manifests/<events>.csv \
    --stress_root <NeuroNova_data> --stress_manifest R2_Aadi_Contribution/metadata/stress_test_manifest_dmd.csv \
    --out_dir results/r4/dmd_eval
```

Never commit DMD frames, events or stress images (CC BY-NC-ND, see R2 handoff); only the report.

## Run the tests

```bash
python -m pytest tests/r4 -v      # or `python -m pytest` for every role's tests
```
