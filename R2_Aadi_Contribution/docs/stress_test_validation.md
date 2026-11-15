# Synthetic Stress Tests: DMD Held-Out Set and Validation

Built: 2026-11-15 by Aadi Deshmukh (R2)
Script: `code/build_dmd_stress_set.py` (uses `code/stress_generators.py`)
Manifest: `metadata/stress_test_manifest_dmd.csv` (456 rows)
Measurements: `metadata/stress_validation_summary_dmd.json`. Per-image values are in `stress_validation_dmd.csv` next to the images.
Images: `C:\NeuroNova_data\generated_stress\dmd\` on Aadi's PC (513 MB). Logical team path: `shared Google Drive:/NeuroNova/datasets/generated_stress/dmd/`. Restricted: do not put on GitHub.

## What was built

- **Source:** DMD Kaggle copy, drowsiness face videos (1280 x 720, 29.76 fps).
- **People:** the 4 held-out test subjects only (1, 5, 6, 7), one drowsiness session each. No training or validation subject was used.
- **Source frames:** 6 per person, 24 in total, picked from the native DMD annotations:
  - 2 mid-yawn frames (yawn phases merged)
  - 2 mid-frames of the longest eye-closure intervals
  - 2 mid-frames of the longest eyes-open intervals at least 60 frames away from any yawn
- **Face box:** OpenCV Haar frontal-face detector on the original frame (19 of 24 frames). The other 5 frames (3 eyes closed, 2 yawn) used a fixed centre box, because the detector missed the face. The box source is recorded per row.
- **Stress samples:** 19 per source frame, 456 in total:

| Stress | Levels | Variants | Per frame |
|---|---|---|---|
| Low light (gamma and noise) | mild, moderate, severe | 1 | 3 |
| Glare (saturated blob on the eye band) | mild, moderate, severe | 1 | 3 |
| Off-axis (perspective warp 10, 20, 30 degrees) | mild, moderate, severe | 1 | 3 |
| Partial occlusion (10, 25, 40 percent of face height) | mild, moderate, severe | mask, cap brim, hand patch | 9 |
| Face lost (face box filled black) | single level | 1 | 1 |

- **Seeds:** `42 + 100 x subject + frame index`, recorded per row. Every sample can be regenerated exactly.

## Validation on real frames

The measurements use simple, model-independent checks: OpenCV Haar face and eye detection at the original face box location, mean face brightness (0 to 255), and the share of saturated face pixels. They show whether each stress behaves as intended and gets stronger with severity. **They are not the project model's performance.**

| Condition | Images | Face found | Eyes found | Face brightness | Saturated share |
|---|---|---|---|---|---|
| Original (no stress) | 24 | 0.79 | 0.75 | 91.0 | 0.016 |
| Low light mild | 24 | 0.71 | 0.62 | 67.9 | 0.014 |
| Low light moderate | 24 | 0.71 | 0.42 | 46.3 | 0.010 |
| Low light severe | 24 | 0.54 | 0.00 | 30.5 | 0.006 |
| Glare mild | 24 | 0.33 | 0.33 | 98.4 | 0.020 |
| Glare moderate | 24 | 0.00 | 0.00 | 113.4 | 0.037 |
| Glare severe | 24 | 0.00 | 0.00 | 138.0 | 0.093 |
| Off-axis mild | 24 | 0.79 | 0.75 | 93.2 | 0.022 |
| Off-axis moderate | 24 | 0.71 | 0.71 | 96.7 | 0.035 |
| Off-axis severe | 24 | 0.25 | 0.21 | 100.9 | 0.052 |
| Occlusion mild (3 kinds) | 72 | 0.68 | 0.62 | 88.5 | 0.015 |
| Occlusion moderate | 72 | 0.56 | 0.40 | 85.2 | 0.012 |
| Occlusion severe | 72 | 0.28 | 0.22 | 80.8 | 0.011 |
| Face lost | 24 | 0.00 | 0.00 | 0.0 | 0.000 |

On the originals, the detector found the face in 8 of 8 eyes-open frames, 5 of 8 eyes-closed frames, and 6 of 8 yawn frames. The detector itself struggles with closed eyes and open mouths.

Severity results:
- Low light, glare, off-axis, and occlusion all degrade monotonically with severity.
- Low light removes eye detection at severe level (face brightness 30 of 255).
- Off-axis has little effect up to 20 degrees and a large effect at 30 degrees.
- Cap brim (covers the eyes) has the strongest effect among the occlusion types: face found 0.71, 0.54, 0.00 for mild, moderate, severe. Mask: 0.54, 0.50, 0.42. Hand patch: 0.79, 0.62, 0.42.

## Visual check

Contact sheets for two frames were inspected (`_qa/` next to the images). Low light, off-axis, all three occlusion kinds, and face lost look as intended.

The first version of the glare generator was rejected after this check. It drew a small, faint, non-saturated dot that often landed on the hair: the saturated share stayed at the 0.016 baseline for every level. The generator was changed (2026-11-15) to place the blob on the eye band, with a saturated core and a larger halo. The new parameters:

| Level | Halo opacity | Radius (share of face width) | Saturated core (share of radius) |
|---|---|---|---|
| mild | 0.55 | 0.15 | 0.30 |
| moderate | 0.80 | 0.22 | 0.45 |
| severe | 0.95 | 0.30 | 0.60 |

All images were regenerated with the new generator. The table above reflects the new version.

## Limitations

- These are image-level transforms on still frames, not real camera degradation. Report them separately from natural-condition results, and do not call them equivalent to real low light or glare.
- Glare is detector-hostile even at mild level (face found 0.33). The Haar detector is very sensitive to a bright spot on the eyes. The team should check the effect on the project model before deciding the levels are well spaced.
- 4 people and 24 frames give fewer than five people per cell, so results must be reported as indicative, with the person count stated.
- RLDD stress tests are not built yet. According to the pipeline README, the processed RLDD output holds feature archives and 64 x 64 eye crops, not full frames. The metadata points the original videos to a teammate's personal drive, so whether they are still available is unconfirmed. Whole-frame RLDD stress tests need those videos. Eye-crop-only tests would need to be labelled as such.
- Off-axis warps leave black corners outside the original frame. Models that use the full frame see these corners.

## How the model team should use this

1. Run the trained model on every image listed in `stress_test_manifest_dmd.csv`, and on the 24 `original.png` frames.
2. Compare each stressed output with the same frame's original output (paired comparison).
3. Report results per stress type and severity, and keep them separate from natural-condition results.
4. For video-based models, the same transforms can be applied frame by frame to held-out clips with `apply_stress`. Record any such clips in a new manifest with the same columns.
