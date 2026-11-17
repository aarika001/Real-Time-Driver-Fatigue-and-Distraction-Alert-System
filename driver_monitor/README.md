# Driver Fatigue & Distraction Alert: demo app (R8)

Runs end to end today: real MediaPipe landmarks + EAR/MAR/head pose, with **stub** models behind fixed interfaces.
Research prototype, not a certified safety system.

## Run
```
pip install -r requirements.txt
python main.py                      # webcam; downloads the 4 MB landmark model on first run
python main.py --source clip.mp4 --save-out demo.mp4 --calib-seconds 5
python tests/smoke_test.py          # no camera needed
```
Use `--calib-seconds 5` while developing; the real default is 30 s.

## Keys
1 glare | 2 low light | 3 occlusion | 4 face lost | 5 off-axis | 0 clear | +/- severity
c recalibrate | [ ] miss:FA ratio (3..20) | m mute | l landmarks | h help | q quit

## Swap points (one file each; keep the signature, replace the body)
| Owner | File | Contract |
|---|---|---|
| R4 | `eye_model.py` | `predict_eye_closed(eye_crops, ear) -> P(closed)` (64x64 grayscale crops) |
| R5 | `perception.py` (`head_angles`) | yaw/pitch in degrees; check signs via `YAW_SIGN`/`PITCH_SIGN` in config |
| R6 | `temporal.py` | `update(features, reliability, t) -> TemporalOut(probs[alert,fatigue,distraction])`; `window_array()` already returns the (75, 8) GRU input |
| R6 | `calibration.py` | baselines from first N s |
| R7 | `reliability.py` | `estimate(percept) -> Reliability(eye, mouth, head)` from `percept.quality` |
| R7 | `policy.py` | working EU policy; tune costs in `config.py` |

`percept.quality` keys (all 0..1): brightness, contrast, glare_frac, sharpness, jitter, det_conf.
`FEATURE_NAMES` in `interfaces.py` fixes the GRU input column order.

## Timing evidence
Each run writes `timing_log.csv` (per-stage ms) and prints mean/p95 processing time on exit.
Webcam FPS is capped by the camera, so quote the "max sustainable FPS" from processing time.

## Known limits
- Perception, glare placement and the stress keys were written without a webcam in the build sandbox:
  verify yaw/pitch signs and EAR/MAR ranges on your own face first.
- Stub thresholds (`config.py`) are heuristics, not tuned values.
- Soft alert is visual only (as in the proposal); audio plays for visual+audio and urgent.
