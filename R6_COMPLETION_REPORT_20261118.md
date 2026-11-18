# R6 Completion Report
Date: 18 November 2026
Scope: R6 Temporal Model and Calibration

## 1. R6 implementation
The R6 package contains:
- Feature schema and metadata validation (`r6/common.py`)
- Driver baseline loading and feature extraction
- Temporal window generation
- GRU model and subject-wise training pipeline (`r6/train.py`)
- Checkpoint loading and inference (`r6/inference.py`)
- Baseline generation (`r6/baseline.py`)
- Dataset auditing (`r6/audit.py`)

Official output classes:
- alert
- fatigue
- distraction

The GRU architecture uses a 64-unit hidden layer and a three-class output head.

## 2. Software validation
- Full test suite: 15 passed.
- Release-copy test suite: 15 passed.
- Integration tests cover inference with a synthetic test checkpoint, missing subject baseline rejection, and insufficient recording length.
- The synthetic checkpoint validates software integration only. It is not an officially trained or accurate R6 model.

## 3. Calibration validation

### Mirror
- 89 baseline rows.
- 89 unique subjects.
- No duplicate or missing subject IDs.
- No non-finite numeric values.
- No negative pitch standard deviations.
- Calibration validation: PASS.
- Selected baseline durations include 10, 20, and 30 seconds.
- Mean selected duration: approximately 16.85 seconds.

These baselines are integrity-valid, but must not all be described as 30-second neutral baselines.

### YawDD
- 29 baseline rows and 29 unique subjects.
- No duplicate or missing subject IDs.
- No non-finite numeric values.
- No negative pitch standard deviations.
- 25 approximately 30-second baselines.
- 4 partial baselines:
  - Female_10: 12.60 seconds
  - Female_4: 14.87 seconds
  - Female_6: 19.20 seconds
  - Female_9: 28.60 seconds
- Calibration validation: PASS.

### Calibration handoff note
The R5 interface requires a subject-specific neutral-pose baseline, intended to use approximately the first 30 seconds of normal driving. Shorter baselines must be identified by their actual duration and must not silently be represented as full 30-second baselines.

## 4. Dataset and official training assessment

### Mirror
- 319 recordings and 90 unique subjects.
- Source labels: Normal, Talking, Yawning, TalkingYawning.
- These are activity labels, not validated alert/fatigue/distraction labels.

### YawDD
- 29 processed subjects.
- 32,303 feature annotation rows.
- Contains annotations such as Normal, Distracted, Yawning, Looking Away, and ambiguous/activity-only labels.
- These annotations do not currently establish a validated mapping to all three official R6 classes.

### Training decision
Official three-class training is BLOCKED pending a suitable labelled dataset or an approved annotation protocol.

Do not map:
- Yawning directly to fatigue.
- Looking away directly to distraction.
- Normal directly to alert.

Such mappings would require a defensible, documented labelling protocol and validation.

The official training script is implemented, but no official trained three-class checkpoint is claimed by this report.

## 5. Separate experiments
Any Mirror activity classifier checkpoint is an activity experiment, not the official R6 vigilance model. It must not be presented as the official R6 checkpoint.

The synthetic checkpoint used in integration tests is also not a trained model.

## 6. Remaining requirements for official model training
1. Obtain or approve ground-truth labels for alert, fatigue, and distraction.
2. Ensure sufficient distinct subjects and examples for all three classes.
3. Verify labels and subject identities.
4. Perform subject-independent train, validation, and test splitting.
5. Train the official GRU using the R6 training pipeline.
6. Evaluate per-class precision, recall, F1, confusion matrix, and subject-level performance.
7. Save and validate the trained checkpoint and preprocessing artifacts.

## 7. Completion status

| Area | Status |
|---|---|
| R6 feature and model implementation | Implemented |
| Inference pipeline | Implemented |
| Baseline generation and loading | Implemented |
| Software tests | PASS: 15/15 |
| Mirror calibration integrity | PASS |
| YawDD calibration integrity | PASS |
| Dataset feature compatibility | Previously validated |
| Official three-class ground truth | Not established |
| Official three-class model training | Blocked pending valid labels |
| Official trained checkpoint | Not available |

## Final assessment
The R6 software implementation, inference path, tests, and calibration validation are completed to the extent described above. Official three-class model training and performance validation remain pending because the currently available datasets do not provide established ground truth for the required R6 classes.

No unsupported model accuracy or official trained-checkpoint claim should be made.

## Exploratory YawDD Behaviour GRU Experiment

A separate subject-independent YawDD behaviour experiment was completed. It achieved 53.12% test accuracy and 0.4288 macro F1, compared with a majority-class baseline of 57.81% accuracy and 0.2442 macro F1. LookingAway recall was 0.20, and the small test set showed substantial subject variation. These results are exploratory and do not qualify as an official R6 model. Full metrics, per-subject results, limitations, and artifact inventory are recorded in `YAWDD_BEHAVIOR_EXPERIMENT_REPORT.md`.
