# YawDD Behaviour GRU Experiment — 18 November 2026

## Purpose
A separate exploratory experiment to test whether explicitly annotated YawDD behaviour windows can support a subject-independent temporal GRU pipeline. This is not the official R6 alert/fatigue/distraction model.

## Dataset and split
- Input: `yawdd_behavior_windows_30f.npz`
- Window length: 30 frames
- Feature count: 11
- Classes: Neutral, LookingAway, YawningRelated
- Windows: 324
- Subject-independent split: 15 train subjects, 5 validation subjects, 5 test subjects
- Test set: 64 windows
- Labels were derived from selected explicit annotations. Annotation provenance and ground-truth validity are not independently established.

## Held-out test results

| Metric | Majority baseline | GRU |
|---|---:|---:|
| Accuracy | 0.5781 | 0.5312 |
| Macro F1 | 0.2442 | 0.4288 |

### Per-class results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Neutral | 0.64 | 0.73 | 0.68 | 37 |
| LookingAway | 0.33 | 0.20 | 0.25 | 20 |
| YawningRelated | 0.30 | 0.43 | 0.35 | 7 |

### Per-subject results

| Subject | Windows | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Female_12 | 14 | 0.714 | 0.278 |
| Female_6 | 4 | 0.000 | 0.000 |
| Male_11 | 10 | 0.400 | 0.190 |
| Male_16 | 20 | 0.500 | 0.311 |
| Male_4 | 16 | 0.625 | 0.627 |

## Interpretation and limitations
- Overall accuracy is below the majority-class baseline.
- Macro F1 is higher than the baseline, suggesting some improvement in minority-class recognition.
- LookingAway recall is only 0.20: 16 of 20 LookingAway test windows were missed.
- YawningRelated has only 7 test windows, so its metric is highly uncertain.
- Results vary substantially across held-out subjects.
- The validation split had only one subject contributing LookingAway windows.
- The small, imbalanced dataset and uncertain annotation provenance prevent reliable generalisation claims.
- YawningRelated must not be treated as equivalent to the official R6 fatigue class.

## Artifacts
Saved under `outputs/yawdd_behavior_experiment/`:
- `yawdd_behavior_gru_best.pt`
- `yawdd_behavior_experiment_checkpoint.pt`
- `yawdd_behavior_scaler.joblib`
- `metrics.json`
- `training_history.csv`
- `test_predictions.csv`
- `subject_split.csv`
- `window_manifest.csv`
- `yawdd_behavior_windows_30f.npz`

## R6 status
This is an exploratory behaviour experiment only. It does not replace or qualify as the official R6 three-class model. The official model still requires validated subject-wise alert, fatigue, and distraction labels and an appropriate training/evaluation dataset. The experiment artifacts are kept separate from the official R6 model artifacts.
