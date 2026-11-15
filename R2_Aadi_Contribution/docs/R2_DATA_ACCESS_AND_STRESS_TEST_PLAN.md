# R2 Data Access and Stress-Test Plan

Status: completed for the current scope on 2026-11-15. This file records the plan and how each part was met. Details are in the linked files.

## Scope

R2 works with UTA-RLDD and DMD only (decision 1). This narrows the five-dataset list in proposal v4; the supervisor has been informed. The project is internal course work and is not planned for publication.

## Dataset roles

| Dataset | Role | Scope |
|---|---|---|
| UTA-RLDD | Drowsiness classes and participant-independent evaluation | Fold1 to Fold4, 48 participants (decision 2) |
| DMD | Fatigue cue events (yawns, eye closure) and synthetic stress tests | Kaggle copy, 16 drowsiness face videos (decisions 3, 4, 7) |

## Plan and outcome

| Plan item | Outcome | Evidence |
|---|---|---|
| Track access and terms | Done for both datasets | `metadata/dataset_access_log.csv`, `docs/dataset_sources.md` |
| Index metadata | Done; RLDD array audit optional | `metadata/dataset_metadata_index.csv`, `docs/rldd_drive_check.md`, `docs/dmd_kaggle_source_review.md` |
| Document native labels and project mappings | Done | `metadata/label_mapping.csv`, `metadata/dmd_native_labels.csv` |
| Person-independent splits | Done | `metadata/dmd_subject_split.csv`; RLDD four-fold, leave one fold out |
| Validate stress generators on real frames | Done; glare generator fixed after validation | `docs/stress_test_validation.md` |
| Build stress tests on held-out people with a manifest | Done: 456 samples from DMD subjects 1, 5, 6, 7 | `metadata/stress_test_manifest_dmd.csv` |
| Evaluation handoff | Done | `docs/R2_HANDOFF.md` |

## Synthetic stress tests

The tests run on held-out people only and are reported separately from natural conditions.

| Stress | Operation |
|---|---|
| Low light | Gamma and noise |
| Glare | Saturated blob with halo on the eye band |
| Off-axis camera | Perspective warp of 10, 20, and 30 degrees |
| Partial occlusion | Mask, cap brim, and hand patch |
| Face lost | Face box filled black |

The levels are `mild`, `moderate`, and `severe`. Validated parameters are in `code/stress_generators.py` and `docs/stress_test_validation.md`.

## Data handling

- Raw and derived data are kept outside the repository (`C:\NeuroNova_data`, to be copied to the team Drive).
- The repository is public for now (decision 11). Only code, documentation, and metadata go into it.
- Do not distribute DMD-derived material outside the team and supervisor.
- Cite both datasets in the course report.

## Not in the current scope

- DMD distraction and gaze evaluation (decisions 4 and 5)
- RLDD stress tests (decision 9)
- RLDD Fold5 (decision 2)
- YawDD, MRL Eye, and NTHU-DDD (decisions 1 and 10)
