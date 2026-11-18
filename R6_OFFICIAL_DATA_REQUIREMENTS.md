# R6 Official Training Data Requirements
Date: 18 November 2026
Status: Draft specification — confirm class definitions with the project team before annotation.

## 1. Objective

Prepare subject-independent, temporally labelled data for the official R6
three-class GRU classifier.

Official labels:
- alert
- fatigue
- distraction

Do not treat activity labels such as talking, smiling, yawning, or looking
away as automatically equivalent to these three classes.

## 2. Required recording information

Every recording must have:
- A unique subject_id.
- A unique recording identifier.
- A corresponding feature NPZ file.
- A valid subject-specific neutral-pose calibration baseline.
- Frame timestamps or a reliable frame-to-time mapping.
- Explicit annotations with start and end timestamps.
- An annotation source and annotator identifier.

## 3. Calibration requirement

R5 requires a 30-second neutral-pose baseline for each subject.

The baseline must be computed from an appropriate neutral recording segment.
Do not silently treat a shorter baseline as a complete 30-second baseline.

Required baseline fields:
- subject_id
- mean_pitch
- mean_yaw
- std_pitch

The existing R6 feature extraction expects these baseline fields.

## 4. Annotation format

Recommended annotation CSV:

subject_id
recording_id
start_time_seconds
end_time_seconds
label
annotator_id
confidence
notes

The label field must contain exactly one of:
- alert
- fatigue
- distraction

Use a separate ambiguity/review mechanism for uncertain segments.
Do not force uncertain observations into one of the three classes.

## 5. Annotation rules

The project team must approve operational definitions for all three classes
before annotation begins.

Annotators should:
- Label observable evidence, not guess a driver's internal state.
- Mark transitions separately when the state changes.
- Exclude unclear, occluded, or unusable intervals.
- Avoid assigning labels based on a single isolated frame.
- Record uncertainty rather than manufacturing a label.
- Apply consistent rules across all subjects.

Yawning, talking, smiling, and looking away are observable behaviours.
They are not sufficient by themselves to establish fatigue or distraction.

## 6. Training data requirements

- Split by subject, never randomly by individual frames or windows.
- Keep each subject in only one of train, validation, or test.
- Ensure all three official classes are represented in each split.
- Check class and subject distributions before training.
- Prevent overlapping windows from crossing annotation boundaries.
- Exclude windows containing ambiguous or conflicting labels.
- Report per-class precision, recall, and F1.
- Report macro F1, confusion matrix, and per-subject performance.
- Compare against a majority-class baseline.

## 7. Existing data limitations

Mirror activity labels and YawDD behaviour annotations are exploratory
sources only. They must not be mapped directly to official R6 labels
without an approved annotation protocol and validated ground truth.

The existing YawDD behaviour experiment is not an official R6 model.
The existing Mirror activity experiment is not an official R6 model.
A synthetic checkpoint is only a software integration test.

## 8. Training release criteria

Official training can be considered complete only when:
1. The annotation definitions are approved.
2. Ground-truth labels are available for all three classes.
3. Subject-wise train, validation, and test splits are valid.
4. No subject leakage exists between splits.
5. All required feature and calibration checks pass.
6. Training completes and the best checkpoint is saved.
7. Held-out subject evaluation is documented.
8. The checkpoint passes the existing R6 inference tests.

## 9. Current status

R6 implementation and software validation: complete.
Calibration generation and validation: complete with duration caveats.
Exploratory experiments: complete.
Official three-class model training: blocked pending validated labels.

This document specifies the data needed to unblock official training.
It does not claim that the official model has been trained.
