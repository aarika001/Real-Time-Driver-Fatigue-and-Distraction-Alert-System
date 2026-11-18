# R6 → R5 Calibration Integration Contract

**Date:** 18 November 2026  
**Provider:** R6 — Temporal Model and Calibration  
**Consumer:** R5 — Head-Pose / Driver-State Feature Processing

## 1. Purpose

R6 supplies subject-specific neutral-pose calibration values.
R5 uses these values to calculate relative head-pose deviations.

The baseline is a prerequisite for calculating meaningful relative
yaw and pitch features.

## 2. Calibration file locations

### Mirror — full SQPNP dataset

`data/processed/mirror_sqpnp_full/calibration/driver_baselines_r5.csv`

Related calibration artifacts:
- `baseline_candidates.csv`
- `baseline_rejections.csv`
- `calibration_audit.csv`
- `driver_baselines_10s.csv`
- `driver_baselines_20s.csv`
- `driver_baselines_30s.csv`

Use `driver_baselines_r5.csv` as the R5 handoff file.

### YawDD

`data/processed/yawdd_r6/calibration/driver_baselines_r5.csv`

This file contains 29 subject records. Four baselines are partial:
- Female_10
- Female_4
- Female_6
- Female_9

R5 must preserve their partial status and must not silently treat them
as complete 30-second baselines.

## 3. Required calibration fields

| Field | Meaning |
|---|---|
| subject_id | Unique driver/subject identifier |
| mean_pitch | Neutral baseline pitch |
| mean_yaw | Neutral baseline yaw |
| std_pitch | Pitch variation during calibration |

YawDD also contains calibration metadata such as:
- baseline_frames
- baseline_duration_seconds
- status

Retain additional metadata when loading the CSV.

## 4. Baseline calculation

The intended calibration period is approximately the first 30 seconds
of normal driving frames.

The baseline is subject-specific.

R5 should calculate relative deviations using the matching subject's
baseline, rather than using a global reference:

`delta_pitch = current_pitch - mean_pitch`

`delta_yaw = wrap_angle_difference(current_yaw, mean_yaw)`

Angle differences must account for angular wraparound.

Use the project's established feature implementation where possible
instead of independently implementing a conflicting convention.

## 5. Integration requirements for R5

1. Load the appropriate R5 calibration CSV.
2. Match each recording to its `subject_id`.
3. Retrieve that subject's mean pitch and yaw.
4. Calculate relative pitch and yaw features.
5. Preserve missing-baseline and partial-baseline status.
6. Do not substitute another subject's baseline.
7. Do not silently replace missing values with zero.

If a subject has no valid baseline, flag the recording for handling
rather than treating zero as that driver's neutral pose.

## 6. Validation

R6's calibration validation script:

`scripts/validate_r5_calibration.py`

Previously recorded validation:
- Mirror: 89 rows, 89 unique subject IDs, no duplicate IDs,
  missing required values, non-finite values or negative pitch standard deviations.
- YawDD: 29 rows, 29 unique subject IDs, no duplicate IDs,
  missing required values, non-finite values or negative pitch standard deviations.
- YawDD: 25 complete and 4 partial baselines.
- Overall calibration validation: PASS.

Validation of CSV structure does not establish that every baseline
has a complete 30-second duration.

## 7. Ownership and dependencies

**R6 responsibilities**
- Generate and maintain neutral-pose calibration baselines.
- Provide subject identifiers and baseline fields.
- Document incomplete or rejected calibration records.
- Maintain a consistent angular-difference convention.

**R5 responsibilities**
- Consume the correct calibration file.
- Match baselines to subjects.
- Calculate relative head-pose features.
- Handle partial and missing baselines explicitly.
- Report integration failures rather than silently substituting values.

## 8. Important model limitation

The R6 software pipeline has passed its 15-test automated suite.

The official three-class vigilance checkpoint remains pending suitable
ground-truth data for alert, fatigue and distraction.

Calibration availability and software integration do not imply that
the official R6 classifier has been trained or accuracy-validated.

## 9. Handoff conclusion

The calibration interface and file locations are documented for R5.

R5 can integrate the subject-specific neutral-pose baseline interface
independently of the pending official R6 model training.

Any partial baseline must remain identifiable throughout the
downstream pipeline.
