# `cues/` — Eye and mouth cue models (R4)

**Owner:** R4 (Dhwaj). Converts MediaPipe landmarks + a grayscale frame into
per-frame eye and mouth cues for R3 (evaluation), R6 (calibration + GRU),
R7 (reliability) and R8 (live app).

## Status

| Milestone | Module | Status |
|---|---|---|
| M1 | `landmarks.py`, `eye_crop.py`, `schemas.py` (interface contract) | done |
| M2 | `mouth_cue.py` (MAR, per-driver baseline, yawn features) + YawDD scripts | done |
| M3 | `eye_dataset.py` (MRL Eye / NTHU-DDD / DMD loaders, augmentation) | planned |
| M4 | `train_eye_cnn.py` (MobileNetV2) | planned |
| M5 | `eval_eye_cnn.py` (per-condition, cross-dataset, baseline B2) | planned |
| M6 | `inference_r4.py` (≥15 FPS, ONNX, webcam test) | planned |

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

## Run the tests

```bash
python -m pytest tests/r4 -v
```
