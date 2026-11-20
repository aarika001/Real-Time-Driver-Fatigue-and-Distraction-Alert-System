# Real-Time Driver Fatigue and Distraction Alert System
 
**Team NeuroNova** | CS F407 Artificial Intelligence | BITS Pilani, Hyderabad Campus
**Track:** Application (Applications of Deep Learning, with decision-making under uncertainty from the Agents topic)
**Branch with the final, integrated work:** `integ1`
 
A real-time driver-monitoring system that uses an ordinary webcam to detect **fatigue** and **distraction**, and also estimates **how much it can trust its own observations** before deciding whether to alert the driver. When glare, darkness or an off-axis camera make a cue unreliable, it ignores that cue and says "monitoring degraded" instead of guessing.
 
> **Research prototype, not a certified safety system.** It was built for a university course and tested on public datasets and scripted scenarios, not on long real road trips.
 
---
 
## Table of contents
 
1. [Team](#team)
2. [What the system does](#what-the-system-does)
3. [How it works](#how-it-works)
4. [Results](#results)
5. [Repository layout](#repository-layout)
6. [Setup](#setup)
7. [Run the live demo](#run-the-live-demo)
8. [Run the tests](#run-the-tests)
9. [Reproduce the training and evaluation](#reproduce-the-training-and-evaluation)
10. [Data](#data)
11. [Changes from the proposal](#changes-from-the-proposal)
12. [Known limitations](#known-limitations)
13. [Ethics and privacy](#ethics-and-privacy)
14. [References](#references)
---
 
## Team
 
| Role | Member | Student ID | Area | What they delivered |
| --- | --- | --- | --- | --- |
| R1 | Aarika | 2024A7PS0023H | Project lead, integration | Architecture, repository and code review, merging all modules into one branch |
| R2 | Aadi Deshmukh | 2023A7PS0094H | Data access, stress tests | Dataset access log and terms, labels and subject splits, DMD audit, stress generators, 456-image stress set, hand-off documents |
| R3 | Atharva Shekhar Deshkar | 2023A7PS0121H | Data pipeline, evaluation | UTA-RLDD preprocessing into 15 FPS feature archives, metadata index, splits |
| R4 | Dhwaj Kishore | 2024A1PS1187H | Eye and mouth cues | MobileNetV2 eye-closure CNN, ONNX export, mouth cue and yawn logic, live cue pipeline, evaluation, PERCLOS and DMD harnesses |
| R5 | Aaditya Nandkumar Bhagat | 2023A7PS0131H | Head pose | Head-pose core, per-driver head baseline, off-axis stress test |
| R6 | Sejal Mittal | 2024A7PS0161H | Temporal model, calibration | GRU with cue dropout, temperature scaling, per-driver calibration, four fold checkpoints |
| R7 | Jayesh Singh Mehta | 2024A7PS0058H | Reliability and policy | Reliability estimator, degraded-state tracker, expected-utility policy, annoyance tracker, cost sweep, ablation runner |
| R8 | Dev Tiwari | 2024AAPS0304H | Real-time app | Threaded capture, HUD, audio, live stress modes, timing instrumentation, performance optimisation, smoke test |
 
---
 
## What the system does
 
The proposal named three weaknesses of typical webcam driver monitors. Each one maps to a part of this system.
 
| Limitation of existing monitors | Our answer | Where in the code |
| --- | --- | --- |
| **Silent degradation:** the monitor keeps giving confident answers even when glare, dark light or an off-axis camera ruin a cue | A reliability score for each cue (eye, mouth, head pose), cue masking, and a "monitoring degraded" banner | `driver_monitor/reliability.py`, `R7_Jayesh_Contribution/reliability/` |
| **Calibrated for nobody:** one global threshold for every driver and camera | A 30-second per-driver calibration; all cues are measured relative to the driver's own baseline | `driver_monitor/calibration.py`, `temporal-model/features.py` |
| **Error costs ignored:** a miss and a false alarm are treated as equally bad | An expected-utility policy with a miss-to-false-alarm cost ratio, four alert levels and an annoyance penalty | `driver_monitor/policy.py`, `R7_Jayesh_Contribution/policy/` |
 
The whole system acts as one utility-based agent: the webcam is its sensor, and the visual cue, audio alert and "monitoring degraded" banner are its actuators. It picks the action with the highest expected utility under noisy, uncertain percepts.
 
---
 
## How it works
 
```
Webcam / video file
      |  (threaded capture)
      v
Perception: MediaPipe Face Landmarker (478 landmarks)
      |  EAR, MAR, yaw / pitch / roll, 64x64 CLAHE eye crops, 4 crop-quality signals
      +--> Eye CNN (MobileNetV2, ONNX, ~8.88 MB) --> calibrated P(eye closed)
      |
      v
Per-driver calibration (first 30 s) --> 13-feature row per frame
      |
      v
Reliability estimator (eye / mouth / head score in [0, 1])
      |  cues below R_min = 0.35 are zeroed out; all cues unreliable for 3 s = "monitoring degraded"
      v
Temporal GRU ensemble (4 folds, hidden 64, window (75, 13) = 5 s at 15 FPS, temperature scaled)
      |  P(fatigue); distraction comes from a head-pose rule (sustained yaw / pitch deviation)
      v
Expected-utility policy --> none | soft visual | visual + audio | urgent
      |
      v
HUD overlay, audio chimes, per-stage timing log (timing_log.csv)
```
 
**The 13 GRU input features per frame:** eye P(closed); MAR z-score; pitch, yaw, roll z-scores; four quality signals (brightness, contrast, glare fraction, log-sharpness); a face-detected flag; three cue-group availability flags. Frames with no face have the five cue columns set to zero, so no NaN reaches the network.
 
**Alert policy.** The policy works with P(alert), P(fatigue) and P(distraction) and picks the action with the highest expected utility. With the live default miss-to-false-alarm cost ratio of 3:1 (adjustable from 3 to 20 at run time), and no recent alerts, the soft alert needs P(fatigue) above about 0.31, the audio chime above about 0.58 and the urgent alert above about 0.91. A 4-second cool-down and smoothing of P(fatigue) with an exponential moving average (alpha 0.25) stop alert spam.
 
**Speed.** The GRU ensemble refreshes at 4 Hz, not on every frame, and the display width defaults to 960 px with linear scaling. Thread pools are capped so PyTorch, ONNX Runtime, MediaPipe and OpenCV do not fight for cores.
 
---
 
## Results
 
Everything below comes from the `integ1` branch. The eye CNN and GRU numbers are on people never seen in training.
 
### Eye-closure CNN (MRL Eye, 7 held-out people, 20,333 crops)
 
| Metric | Value |
| --- | --- |
| Accuracy | 0.978 (95% CI 0.946 to 0.993) |
| Recall of closed eyes | 0.982 |
| F1 (closed) | 0.979 |
| AUROC | 0.997 |
| Expected calibration error (ECE) | 0.008 |
| Per-person accuracy, min / median | 0.916 / 0.959 |
| Weakest condition (at least 5 people) | Low reflection, recall of closed eyes 0.908 |
 
Under graded synthetic stress, recall of closed eyes stays above 0.95 up to severity level 2 for blur, glare, low light, noise and occlusion. Glare and low light first drop by more than 10 points at level 4 (`results/r4/stress_sweep.png`).
 
### Temporal GRU (UTA-RLDD, 4-fold, person-independent, alert vs fatigue)
 
| Metric | Value |
| --- | --- |
| Window accuracy | 0.711 |
| Balanced accuracy | 0.710 |
| Macro-F1 | 0.707 |
| AUROC | 0.753 |
| ECE | 0.056 |
| Video-level accuracy | 0.737 |
| Per-person accuracy, min / median / max | 0.293 / 0.703 / 0.990 |
| Fitted temperature per fold | 1.86 / 2.04 / 1.05 / 15.49 |
 
The spread between people is wide, and we report it instead of hiding it behind the pooled number.
 
### End-to-end scenarios (initial policy settings, cost ratio 10)
 
| Scenario | Max P(fatigue) | Max P(distraction) | Alert level reached |
| --- | --- | --- | --- |
| Normal driving | 0.260 | 0.000 | 1 (soft visual) |
| 1 s mirror glance | 0.297 | 0.000 | 1 (soft visual) |
| Sustained look-away | 0.355 | 0.679 | 2 (visual + audio) |
| Extended eye closure | 0.848 | 0.000 | 3 (urgent) |
 
### Real-time performance (laptop CPU, target 15 FPS = 66.7 ms per frame)
 
| Build | Frame time | Throughput |
| --- | --- | --- |
| First integrated build | about 210 ms | 4.5 to 4.7 FPS |
| After capping threads | 57.7 to 66.0 ms | about 15 FPS |
| Final (throttled GRU, 960 px display) | 42 to 54 ms | 18.5 to 23.8 FPS |
 
Overall speed-up is above 3.2 times. A longer logged session (`driver_monitor/timing_log.csv`, 6,724 frames) has a mean total frame time of 33.3 ms (95th percentile 45.5 ms), with every frame inside the 66.7 ms budget. The standalone cue pipeline runs at 28.7 FPS.
 
### Tests
 
97 automated pytest tests plus an end-to-end smoke test pass on `integ1`.
 
---
 
## Repository layout
 
| Folder | Contents |
| --- | --- |
| `driver_monitor/` | Integrated real-time app (entry point `main.py`), HUD, audio, stress modes, smoke test, ONNX eye model, landmark model |
| `cues/`, `scripts/r4/` | Eye CNN, mouth cue and yawn logic, live cue pipeline, ONNX export, evaluation scripts |
| `head_pose/` | Head-pose core, per-driver head baseline, off-axis stress test |
| `temporal-model/` | GRU, features, per-driver calibration, training and evaluation, four fold checkpoints in `runs/with_eye/` |
| `R7_Jayesh_Contribution/` | Reliability estimator, expected-utility policy, annoyance tracker, cost sweep, ablation runner |
| `R2_Aadi_Contribution/` | Dataset access log, labels, subject splits, stress generators and the 456-image DMD stress set |
| `RLDD-preprocess/` | Preprocessing of UTA-RLDD into 15 FPS feature archives |
| `configs/` | YAML configuration for the eye CNN |
| `results/r4/` | Eye CNN evaluation report, FPS report, training curves, stress sweep plot |
| `tests/`, `conftest.py`, `pytest.ini` | Automated tests |
 
Each module folder has its own README with more detail.
 
---
 
## Setup
 
You need **Python 3.10 or newer** (3.9 may also work), `git`, and a webcam for the live demo (a video file works too). A normal laptop CPU is enough; no GPU is needed to run the demo.
 
```bash
git clone https://github.com/AI-Projects-S1-26-27/Real-Time-Driver-Fatigue-and-Distraction-Alert-System.git
cd Real-Time-Driver-Fatigue-and-Distraction-Alert-System
git checkout integ1
```
 
Create a virtual environment and install the dependencies.
 
**Windows (PowerShell)**
 
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r driver_monitor/requirements.txt
pip install -r requirements-r4.txt
```
 
**Linux / macOS**
 
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r driver_monitor/requirements.txt
pip install -r requirements-r4.txt
```
 
`requirements-r4.txt` brings in PyTorch, ONNX Runtime, scikit-learn, pandas and pytest, which the GRU and tests need.
 
> **Important:** run every command from the **repository root** and set `PYTHONPATH` to the repository root first. The app imports modules from several top-level folders (`cues/`, `head_pose/`, `R7_Jayesh_Contribution/`), and `main.py` does not add the root to the path itself.
>
> - PowerShell: `$env:PYTHONPATH = "."`
> - Linux / macOS: `export PYTHONPATH=.`
 
---
 
## Run the live demo
 
The trained models are already in the repository: the eye CNN (`driver_monitor/models/eye_cnn.onnx`), the MediaPipe landmark model (`driver_monitor/models/face_landmarker.task`) and the four GRU fold checkpoints (`temporal-model/runs/with_eye/model_Fold1.pt` to `model_Fold4.pt`).
 
```bash
# live webcam (sit still and look at the camera for the first 30 s while it calibrates)
python driver_monitor/main.py
 
# a video file, with a short calibration, and save the annotated result
python driver_monitor/main.py --source path/to/clip.mp4 --calib-seconds 5 --save-out demo.mp4
```
 
Useful options: `--source` (camera index or video path), `--calib-seconds` (default 30), `--no-audio`, `--no-mirror`, `--display-width` (default 960), `--log` (timing CSV path), `--save-out`.
 
**Keys while it runs**
 
| Key | Action |
| --- | --- |
| `1` / `2` / `3` / `4` / `5` | Stress modes: glare, low light, occlusion, face lost, off-axis rotation |
| `0` | Clear stress |
| `+` / `-` | Stress severity |
| `c` | Recalibrate |
| `[` / `]` | Change the miss-to-false-alarm cost ratio (3 to 20) |
| `m` | Mute audio |
| `l` | Toggle landmarks |
| `h` | Help |
| `q` | Quit |
 
Every run writes a per-stage timing log (`timing_log.csv`) and prints mean and 95th percentile processing time on exit.
 
A cue-only demo (eye and mouth cues, without the GRU and policy) is also available:
 
```bash
python scripts/r4/webcam_demo.py --eye_model driver_monitor/models/eye_cnn.onnx
```
 
> The live webcam path depends on your camera, lighting and framing. It was mainly validated on video files, so check on your own face that yaw and pitch signs look right (`YAW_SIGN` and `PITCH_SIGN` in `driver_monitor/config.py`).
 
---
 
## Run the tests
 
```bash
# all unit tests (97)
python -m pytest
 
# only the eye / cue tests
python -m pytest tests/r4 -v
 
# end-to-end smoke test (no camera needed)
python driver_monitor/tests/smoke_test.py
```
 
The smoke test checks that the GRU input is (75, 13), that normal driving and a mirror glance never go above alert level 1, that look-away and drowsy scenarios reach level 2 or higher, and that a dark synthetic degradation triggers the "monitoring degraded" banner.
 
---
 
## Reproduce the training and evaluation
 
You only need this section if you want to retrain the models or recompute the numbers. The datasets are public but are **not** included in the repository (see [Data](#data)).
 
### 1. Eye-closure CNN (MRL Eye)
 
```bash
# quick pipeline check (about a minute)
python scripts/r4/train_eye_cnn.py --mrl_root <MRL folder or mrlEyes_2018_01.zip> --out_dir runs/smoke --smoke
 
# full training (a GPU helps; add --cpu to force CPU)
python scripts/r4/train_eye_cnn.py --mrl_root <MRL folder or mrlEyes_2018_01.zip> --out_dir runs/eye_cnn
 
# evaluate on the held-out people, including per-condition tables and stress sweeps
python scripts/r4/eval_eye_cnn.py --eye_model runs/eye_cnn/best.pt \
    --splits runs/eye_cnn/splits.csv --mrl_root <MRL folder or zip> --out_dir results/r4/eye_eval
 
# export to ONNX for the real-time app
python scripts/r4/export_onnx.py --checkpoint runs/eye_cnn/best.pt --out runs/eye_cnn/eye_cnn.onnx
```
 
Settings live in `configs/r4.yaml`. The split is by person (25 train, 5 validation, 7 test), so no person appears in two sets.
 
### 2. Preprocess UTA-RLDD
 
```bash
python RLDD-preprocess/preprocess-rldd.py \
    --rldd_dir <RLDD root with Fold1_part1, Fold1_part2, ...> \
    --output_dir ./processed_rldd \
    --model_path ./face_landmarker.task
```
 
This writes 15 FPS `.npz` feature archives and `rldd_metadata_index.csv`. Add `--exclude_low_vigilant` to skip the KSS 5 videos. The expected folder layout and the label mapping (KSS 0 = alert, 5 = low vigilant, 10 = fatigue) are described in `RLDD-preprocess/README.md`.
 
### 3. Add the eye-CNN probability to each frame
 
```bash
python temporal-model/add_eye_probs.py --data_dir ./processed_rldd --eye_model driver_monitor/models/eye_cnn.onnx
```
 
### 4. Train and evaluate the temporal GRU (4-fold, person-independent)
 
```bash
cd temporal-model
 
# no dataset yet? make fake data in the same format to check the pipeline
python make_synthetic_rldd.py --out_dir /tmp/synthetic_rldd
python train.py --data_dir /tmp/synthetic_rldd --out_dir runs/synthetic --epochs 6
 
# real data
python train.py --data_dir ../processed_rldd --out_dir runs/with_eye
 
# ablations
python train.py --data_dir ../processed_rldd --out_dir runs/no_calibration --no_calibration
python train.py --data_dir ../processed_rldd --out_dir runs/no_reliability --no_reliability
 
python test_temporal.py
cd ..
```
 
For each test fold, the next fold is used for validation (early stopping and temperature scaling) and the rest for training, and the code asserts that train, validation and test people never overlap. We evaluated Folds 1 to 4 (see the next sections for why). The four fold checkpoints used by the live app are the files in `temporal-model/runs/with_eye/`.
 
### 5. Reliability, policy and ablation framework
 
```bash
python R7_Jayesh_Contribution/run_r7_pipeline.py
```
 
The ablation runner can compare no reliability, no calibration and a fixed 0.5 threshold. Its numbers in the repository come from synthetic sequences, so we do **not** report them as results.
 
### 6. Synthetic stress tests
 
The 456-image DMD stress set (low light, glare, off-axis, occlusion, face lost at graded severities, fixed random seeds) is built by the generators in `R2_Aadi_Contribution/code/`. See `R2_Aadi_Contribution/docs/` for the dataset access log, stress-test validation and hand-off notes.
 
---
 
## Data
 
No new human data was collected. All data are public datasets used under their own terms for internal course work and are **not** redistributed in this repository (only code, documentation and metadata).
 
| Dataset | What we used | Role |
| --- | --- | --- |
| **UTA-RLDD** | Folds 1 to 4: 48 people, 144 videos (alert, low-vigilant, drowsy), processed to 15 FPS archives | Trains and evaluates the temporal GRU and per-driver calibration |
| **MRL Eye** | 84,898 infrared eye crops from 37 people, with glasses, reflection and lighting tags; 25 / 5 / 7 people for train / validation / test | Trains and evaluates the eye CNN, with per-condition tables |
| **DMD (Kaggle subset)** | Drowsiness face videos: 16 sessions from 13 people, RGB, 1280x720 at 29.76 FPS; test people 1, 5, 6, 7 | Real in-car source for the 456 synthetic stress images and event-level checks |
 
---
 
## Changes from the proposal
 
| Item in the proposal | What happened | Why |
| --- | --- | --- |
| NTHU-DDD dataset | Not used | The dataset is private. We requested access several times and could not get it. Its role (scenarios with glasses, sunglasses and night) is covered by MRL Eye condition tags, DMD real in-car footage and graded synthetic stress tests |
| UTA-RLDD Fold 5 | Excluded | Team decision, so we evaluate with four folds (leave one fold out) instead of the provider's five |
| DMD | Approved 14-person Kaggle subset, drowsiness videos only | No distraction or gaze videos were available, so distraction is not evaluated on DMD |
| YawDD | Tooling only | The final evaluation was narrowed to two datasets. MAR extraction and yawn scripts are built and unit-tested, ready for the data |
| Learned reliability weights | Hand-set weights plus an illumination gate | The fitting code (`fit_cue`, logistic regression) is implemented and tested but was not fitted on labelled data in time |
| Learned distraction class | Rule on sustained head deviation | There were no distraction labels to learn from |
| RLDD stress tests | Not built | The RLDD originals are about 100 GB, too large for the time available |
 
---
 
## Known limitations
 
- The data are natural or stationary recordings, not long real road trips. Head pose is only a proxy for gaze.
- The GRU separates alert from fatigue only. Distraction is a rule on head pose, and low-vigilant videos are left out of training and evaluation.
- Accuracy varies a lot between people (some below 0.3, some above 0.9), and Fold 4 was the hardest fold.
- Live reliability scores use set weights, not weights fitted to data.
- Synthetic stress tests only approximate real cameras and are reported separately from natural results. Some DMD stress cells have only 4 people, so they are indicative.
- A formal false-alarms-per-hour measurement and event-level recall on long natural driving were not run.
- This is a research prototype and must not be relied on to keep a driver safe.
---
 
## Ethics and privacy
 
- No new human-subject data was collected. Public datasets are used under their terms for internal course work.
- Only 36 of the 60 UTA-RLDD participants allowed their faces to be published, so raw faces are not shown in our report or slides.
- The live demo runs locally and records nothing unless you ask for a saved video (`--save-out`) or a timing log.
- The alert policy makes the safety trade-off explicit: a miss can contribute to a crash, but too many false alarms get a system ignored or switched off. Both costs are adjustable.
- Results are reported per person and per condition, so performance for any one kind of user or camera is visible.
### Use of AI tools
 
We used an AI assistant (Claude) in a supporting role: for data-processing scripts and feature windows, drafting parts of the temporal model code and its tests, fixing environment errors, explaining the pipeline, and wording the report. We checked the output by running the code on real data, running the full test suite, and comparing every number in the report with the result files and code. All design decisions, experiments and results are ours.
 
---
 
## References
 
1. World Health Organization, *Global status report on road safety 2023*. Geneva, 2023.
2. T. Soukupova and J. Cech, "Real-time eye blink detection using facial landmarks," CVWW, 2016.
3. D. F. Dinges and R. Grace, "PERCLOS: a valid psychophysiological measure of alertness," FHWA Tech Brief FHWA-MCRT-98-006, 1998.
4. R. Ghoddoosian, M. Galib and V. Athitsos, "A realistic dataset and baseline temporal model for early drowsiness detection," CVPR Workshops, 2019 (UTA-RLDD).
5. J. D. Ortega et al., "DMD: a large-scale multi-modal driver monitoring dataset for attention and alertness analysis," ECCV Workshops, 2020.
6. R. Fusek, *MRL Eye Dataset*, Media Research Lab, VSB - Technical University of Ostrava.
7. M. Sandler et al., "MobileNetV2: inverted residuals and linear bottlenecks," CVPR, 2018.
8. K. Cho et al., "Learning phrase representations using RNN encoder-decoder for statistical machine translation," EMNLP, 2014 (GRU).
9. Y. Kartynnik et al., "Real-time facial surface geometry from monocular video on mobile GPUs," CVPR Workshops, 2019 (MediaPipe Face Landmarker).
10. C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, "On calibration of modern neural networks," ICML, 2017 (temperature scaling).
11. S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed., Pearson, 2020.
12. G. Ersoy, M. A. Tatar, E. Tonbul and S. Kirbiz, "Improving driver drowsiness detection via personalized EAR/MAR thresholds and CNN-based classification," arXiv:2604.22479, 2026.
13. A. E. Cox, E. R. Teoh and J. B. Cicchino, "Subaru drivers' experiences with the DriverFocus driver monitoring system," *Journal of Safety Research*, vol. 98, 2026.
14. C.-H. Weng, Y.-H. Lai and S.-H. Lai, "Driver drowsiness detection via a hierarchical temporal deep belief network," ACCV Workshops, 2016 (NTHU-DDD).
15. S. Abtahi et al., "YawDD: a yawning detection dataset," ACM MMSys, 2014.
16. Regulation (EU) 2019/2144 (General Safety Regulation) and Commission Delegated Regulation (EU) 2023/2590.
 
