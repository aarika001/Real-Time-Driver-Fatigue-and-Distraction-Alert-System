# R2 Status: Done and Remaining

Last updated: 2026-11-15. Scope and decisions: `docs/DECISIONS.md`.

## Done

**Access, terms, and metadata**
- [x] Scope reduced to UTA-RLDD and DMD; supervisor informed.
- [x] Access and terms recorded for both datasets (`metadata/dataset_access_log.csv`, `docs/dataset_sources.md`).
- [x] Metadata index updated with verified counts and formats (`metadata/dataset_metadata_index.csv`).
- [x] DMD Kaggle copy inventoried: 14 people, 45 sessions, all files present (`metadata/dmd_kaggle_session_inventory.csv`).
- [x] DMD drowsiness face videos downloaded (16 sessions, 5.4 GB) and audited: 29.76 fps, frame counts match annotations, 0 issues.
- [x] RLDD processed output checked on the shared Drive: Fold1 to Fold4 present. The pipeline owner confirmed 144 videos, default settings, no failures, and the latest pipeline commit (`docs/rldd_drive_check.md`).

**Labels and splits**
- [x] All 48 DMD native labels verified from the annotation files (`metadata/dmd_native_labels.csv`).
- [x] Project mapping finalised for the current scope (`metadata/label_mapping.csv`, decisions 7 and 8).
- [x] DMD subject-independent split verified and adopted (`metadata/dmd_subject_split.csv`).
- [x] RLDD evaluation set to four folds, leave one fold out (decision 2).
- [x] DMD event files written with seconds: 76 merged yawns and 285 eye-closure intervals (kept outside the repository).

**Stress tests**
- [x] All five generators validated on real DMD frames; glare generator fixed after validation (`docs/stress_test_validation.md`).
- [x] 456 stress samples built from held-out people only (DMD subjects 1, 5, 6, 7) and recorded in `metadata/stress_test_manifest_dmd.csv`.

**Handoff and QA**
- [x] Handoff to the model team written (`docs/R2_HANDOFF.md`).
- [x] Tools for download, audit, extraction, sanitization, and stress building written and tested (`code/`).
- [x] QA checklist completed (`docs/qa_checklist.md`); package checked for em dashes, personal paths, and data files.

## Remaining (Aadi)

- [ ] Copy `C:\NeuroNova_data\dmd` and `C:\NeuroNova_data\generated_stress` to the team Drive.
- [ ] Push this package to the team GitHub repository after the step above.

## Optional (not required for the current scope)

- [ ] Run `code/audit_rldd_processed.py` on a local sync of `processed_rldd` to confirm array shapes. The pipeline owner's answers already cover the counts.
- [ ] Run `code/sanitize_rldd_metadata.py` before any copy of `rldd_metadata_index.csv` is shared, because it contains personal paths.
- [ ] Remove the stale YawDD, MRL Eye, and NTHU-DDD references from the RLDD pipeline README in the GitHub repository. This needs the repository owner.

## Out of scope by decision

- DMD distraction and gaze evaluation (decisions 4 and 5)
- RLDD stress tests (decision 9)
- RLDD Fold5 (decision 2)
- YawDD, MRL Eye, and NTHU-DDD (decisions 1 and 10)
- Volunteer recruitment and new recordings
- Private GitHub repository and restricted Drive link (decisions 11 and 12)
