# Temporal Model (Reliability-Aware GRU) on Preprocessed UTA-RLDD

Stage 6 of the pipeline: a one-layer GRU (64 units) reads a **5-second window**
(75 frames at 15 FPS) of per-driver-normalised cues plus quality signals and outputs
calibrated class probabilities. It is trained and evaluated on the `.npz` features
produced by [`RLDD-preprocess`](../RLDD-preprocess/README.md).

## Quick start

```bash
pip install torch numpy pandas scikit-learn

# 1. real data (output of RLDD-preprocess)
python train.py --data_dir ../processed_rldd --out_dir runs/temporal

# 2. no dataset yet?  fake data in the same format, to check the pipeline runs
python make_synthetic_rldd.py --out_dir /tmp/synthetic_rldd
python train.py --data_dir /tmp/synthetic_rldd --out_dir runs/synthetic --epochs 6

# tests
python test_temporal.py
```

## What it does

| Step | Detail |
| --- | --- |
| Per-driver calibration | Mean/std of MAR and head pose from the first 450 frames (30 s) of the subject's `kss_0_alert.npz`. All cues are z-scored against that baseline. Angle differences are wrapped to ±180° (`solvePnP` pitch often sits near ±180°). |
| Windowing | 75-frame windows, stride 15 (1 s), starting after the calibration period (frame 450). A window never crosses a video or subject. Windows with face found in < 20 % of frames are dropped (`--min_face_rate`). |
| Input per frame (13) | 5 cues (`eye_closed_prob`, `mar_z`, `pitch_z`, `yaw_z`, `roll_z`) + 5 reliability signals (brightness, contrast, glare, log-sharpness, `face_detected`) + 3 cue-group availability flags. Frames with no face have cues set to 0 (not "z = 0"). |
| Cue dropout | During training each cue group (eye / mouth / head) is hidden for a whole window with probability `--cue_dropout` (0.3). The group's values are zeroed and its availability flag set to 0, so the GRU learns to lean on the remaining cues. At least one group always survives. |
| Model | `GRU(13 → 64) → dropout → Linear(64 → classes)`, class-weighted cross-entropy, AdamW, early stopping on validation NLL. Streams frame by frame (`h0` / `return_hidden`) for the live demo. |
| Calibration of probabilities | Temperature scaling fitted on the validation subjects only; reported ECE before and after. |

## Evaluation protocol (no subject leakage)

Subjects are bucketed by the official folds (`Fold1`–`Fold5`; `--cv subject` for random
subject buckets). For each bucket *i*: **test** = *i*, **val** = *i+1* (early stopping and
temperature), **train** = the rest. Every subject is tested exactly once, and the code
asserts that train / val / test subjects are disjoint.

Outputs in `--out_dir`: `metrics.json`, `predictions_windows.csv`,
`predictions_videos.csv`, `model_<Fold>.pt` (weights, temperature, feature layout).

Reported: window accuracy (95 % CI by bootstrapping **subjects**), balanced accuracy,
macro-F1, ROC-AUC (binary), NLL, ECE, per-subject accuracy min / median / max, and
video-level accuracy (mean probability over a video's windows).

## Ablations from the proposal

| Flag | Ablation |
| --- | --- |
| `--no_calibration` | one pooled baseline (fit on the training subjects only) instead of per-driver |
| `--no_reliability` | hide the quality / face-detected channels |
| `--cue_dropout 0` | no cue dropout |

## Label modes (`--label_mode`)

| Mode | Classes |
| --- | --- |
| `binary` (default) | alert vs fatigue; KSS-5 `low_vigilant` videos dropped |
| `binary_lv_fatigue` | alert vs fatigue, `low_vigilant` counted as fatigue |
| `three` | alert / low_vigilant / fatigue |

## Adding R4's eye-state CNN as an input

R4's CNN (MobileNetV2, `cues/eye_model.py`) turns an eye crop into a calibrated
P(eye closed). The processed RLDD `.npz` files already hold the exact 64x64 CLAHE crops
R4's live pipeline produces (R4's `tests/r4/test_contract.py` checks that), so the CNN
only has to be run over them once:

```bash
# temporal-model/ sits in the repo root next to cues/ (branch r4/interface-contract or integration)
python temporal-model/add_eye_probs.py --data_dir data/processed_rldd \
    --eye_model runs/eye_cnn/eye_cnn.onnx          # or runs/eye_cnn/best.pt
python temporal-model/train.py --data_dir data/processed_rldd --out_dir runs/with_eye
```

* `--eye_model` takes R4's `best.pt` checkpoint or the `.onnx` from
  `scripts/r4/export_onnx.py`. Both go through R4's own loader
  (`cues.predictors.load_eye_predictor`), so preprocessing, temperature scaling and the
  sigmoid are exactly what R4 trained. ONNX is usually faster on CPU; `--device cuda`
  works for `.pt`. (The checkpoint is shared via Drive, not git.)
* It writes `kss_<x>_<label>.eye.npy` next to each `.npz`: P(closed) per frame, both eyes
  averaged, NaN on frames with no face (R4's convention). The archives are not modified.
* `train.py` picks the files up automatically; its first log line then lists
  `cue groups: ['eye', 'mouth', 'head']`.
* `--stride N` runs the CNN on every Nth frame and interpolates the rest. Faster, but
  short blinks get smoothed, so keep `--stride 1` for the real run (R4 uses 3 for its
  PERCLOS evaluation, where blinks do not matter).
* If `cues/` is not in the folder above `temporal-model/`, pass `--cues_root`.
* At the end it prints mean P(closed) for alert / low-vigilant / fatigue videos and warns
  if alert is not lower, which is the quick check that the right model was used.

## Known limitations

* **Without the eye step the model has no eye cue** and runs on mouth + head + quality
  only; `train.py` prints a note saying so. RLDD drowsiness is largely about eye closure.
* **Reliability channels are raw quality signals**, a stand-in for the learned reliability
  scores of the reliability estimator (R7). Swap them in `features.frame_features`.
* **No distraction class.** RLDD only has alert / low-vigilant / drowsy; distraction
  labels come from DMD.
* **Video-level labels.** Every window inherits its video's label, so a window inside a
  fatigue video where the driver happens to look alert is still labelled fatigue. Video-level
  accuracy is therefore also reported.
* `make_synthetic_rldd.py` data is invented; scores on it only show the code runs.
