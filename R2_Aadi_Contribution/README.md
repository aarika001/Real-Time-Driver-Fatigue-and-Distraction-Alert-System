# Aadi Deshmukh: R2 Contribution (Data Access and Stress Tests)

NeuroNova, Real-Time Driver Fatigue and Distraction Alert System
BITS Hyderabad, CS F407 Artificial Intelligence
Last updated: 2026-11-15

This is internal course work. It is not intended for publication or external distribution. See `docs/PRIVATE_COURSE_USE_POLICY.md`.

## Start here

1. `docs/R2_HANDOFF.md`: what is ready for the model team, the splits, and the suggested evaluation protocol.
2. `docs/DECISIONS.md`: the 12 decisions that define the current scope.
3. `docs/REMAINING_STEPS.md`: what is done and what is left.

## Scope

R2 covers two datasets:

| Dataset | Used for | In scope |
|---|---|---|
| UTA-RLDD | Drowsiness classes (alert, low_vigilant, fatigue) | Fold1 to Fold4, 48 participants, 144 videos (processed by the team) |
| DMD (Kaggle copy `dmd-video-v1`) | Fatigue cue events (yawns, eye closure) and synthetic stress tests | 16 drowsiness sessions, 13 people (downloaded and audited) |

Proposal v4 listed five datasets. The reduction to two was a team decision, and the supervisor has been informed. DMD distraction is not evaluated in the current implementation (decision 5).

## R2 responsibilities and status

| Responsibility | Status | Main files |
|---|---|---|
| Dataset access and terms | Done | `metadata/dataset_access_log.csv`, `docs/dataset_sources.md` |
| Metadata index | Done (RLDD array audit optional) | `metadata/dataset_metadata_index.csv`, `docs/rldd_drive_check.md` |
| Native labels and project mapping | Done | `metadata/label_mapping.csv`, `metadata/dmd_native_labels.csv` |
| Person-independent splits | Done | `metadata/dmd_subject_split.csv`, `docs/R2_HANDOFF.md` section 3 |
| Synthetic stress tests on held-out people | Done (DMD, 456 samples) | `metadata/stress_test_manifest_dmd.csv`, `docs/stress_test_validation.md` |
| QA | Done | `docs/qa_checklist.md` |
| Handoff to model team | Done | `docs/R2_HANDOFF.md` |

## Folder guide

- `docs/`: handoff, decisions, source reviews, dataset card, stress-test plan and validation, QA checklist, remaining steps
- `metadata/`: access log, dataset index, label mapping, DMD session inventory, native labels, subject split, stress-test manifest, and validation summary
- `code/`: download, audit, extraction, sanitization, and stress-test tools (see `code/README.md`)
- `templates/`: empty stress-test manifest template
- `data/`: placeholders only; real data lives outside this folder

## Data rule

This package contains only code, documentation, and metadata. Videos, frames, stress images, DMD annotation files, and event CSVs are kept outside the repository (`C:\NeuroNova_data` on Aadi's PC, to be copied to the team Drive). The GitHub repository is public for now by team decision (decision 11), so restricted material must never be committed.
