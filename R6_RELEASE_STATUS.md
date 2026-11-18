# R6 — Release Status and Technical Handoff

**Date:** 18 November 2026  
**Role:** R6 — Temporal Model and Calibration

## 1. Implementation status

The R6 software pipeline has been implemented.

Components include:
- Feature extraction and feature-schema validation.
- Neutral-pose baseline loading.
- Temporal window generation.
- GRU-based three-class model architecture.
- Subject-wise data splitting.
- Training and temperature-calibration utilities.
- Checkpoint contract validation.
- Inference through `R6Predictor`.

## 2. Official class contract

The official R6 output classes are:
- alert
- fatigue
- distraction

Dataset labels from unrelated activity-recognition tasks must not be
automatically mapped to these classes.

## 3. Automated testing

Full test suite:

Command:
`python -m pytest .\tests -v`

Recorded result:
- 15 tests collected.
- 15 tests passed.
- 0 failures.
- Runtime: 7.55 seconds.

Coverage includes:
- Temporal window generation.
- GRU architecture and output.
- Three-class label contract.
- Subject-wise split isolation.
- Checkpoint validation and rejection.
- End-to-end inference using a temporary synthetic checkpoint.
- Probability output validation.
- Missing calibration rejection.
- Short-recording rejection.

**Important:** The synthetic checkpoint used in integration testing verifies
software compatibility only. It is not an officially trained model.

## 4. Calibration handoff to R5

R6 provides neutral-pose calibration baselines for downstream use.

R5 requires a driver's neutral-pose baseline to calculate relative
yaw and pitch deviations.

Expected calibration fields:
- subject_id
- mean_pitch
- mean_yaw
- std_pitch

The baseline should be calculated from approximately the first 30 seconds
of normal driving frames, subject to sufficient valid face detections.

Calibration files currently prepared:
- Mirror: 89 subject baselines.
- YawDD: 29 subject baselines.

Four YawDD baselines are partial and must remain explicitly flagged:
- Female_10
- Female_4
- Female_6
- Female_9

R5 must not silently treat partial baselines as complete 30-second
calibration records.

## 5. Dataset and experiment limitations

Mirror activity experiments and YawDD feature extraction support
pipeline development and compatibility testing.

Their activity labels are not verified ground truth for the official
R6 vigilance classes.

The separate Mirror activity experiment checkpoint must not be
presented as the official R6 vigilance checkpoint.

A validated three-class training dataset has not yet been established.

## 6. Official model status

The official three-class checkpoint remains pending.

Before claiming a trained and validated R6 model, the project requires:
1. Suitable labelled data for alert, fatigue and distraction.
2. Subject-independent training, validation and test splits.
3. Training and evaluation on the official class contract.
4. Evaluation metrics and per-class performance.
5. Export of the official checkpoint and preprocessing scaler.
6. Successful inference using those official artifacts.

## 7. Inference artifact contract

The inference implementation expects the artifact directory to contain:
- `r6_gru.pt`
- `r6_preprocessing.joblib`

The checkpoint must contain the required model configuration,
feature schema, labels, temperature and model state dictionary.

The baseline CSV must contain the required calibration fields.

## 8. Release conclusion

**Software implementation:** Implemented  
**Automated software tests:** Passed (15/15)  
**Inference integration:** Passed with synthetic test artifacts  
**Calibration interface:** Implemented and validated  
**R5 calibration dependency:** Documented  
**Official three-class model training and validation:** Pending suitable ground-truth data

R6 is ready for software-level handoff and integration work.
It must not yet be described as an accuracy-validated operational
vigilance classifier.
