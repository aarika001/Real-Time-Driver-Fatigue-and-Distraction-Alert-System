# R2 QA Checklist: UTA-RLDD and DMD

Checked: 2026-11-15 by Aadi Deshmukh (R2). Items that are not ticked are out of scope or belong to the model team, as noted.

## Scope

- [x] Scope reduced from proposal v4's five datasets to UTA-RLDD and DMD; supervisor informed.
- [x] MRL Eye, NTHU-DDD, and YawDD excluded from the R2 implementation.
- [x] Project marked as internal course use only.
- [x] No public publication, public demo, or external distribution planned.
- [ ] Private repository and restricted Drive link: not applied by team decision (decisions 11 and 12). Restricted files are kept out of the repository instead.

## Dataset access and licences

- [x] UTA-RLDD access recorded.
- [x] DMD access recorded (approved Kaggle copy).
- [x] UTA-RLDD terms and citation recorded (no licence stated; cite the CVPRW 2019 paper; face-publication consent limits).
- [x] DMD academic-use and CC BY-NC-ND 4.0 terms recorded.
- [x] Raw and derived data excluded from this package.
- [x] Dataset citations prepared (`docs/PRIVATE_COURSE_USE_POLICY.md`).

## Metadata index

- [x] Both datasets appear in `metadata/dataset_metadata_index.csv`.
- [x] Release or version recorded (DMD: Kaggle `dmd-video-v1` version 1; RLDD: provider release, processed with the latest pipeline commit).
- [x] Acquisition dates recorded (DMD downloaded 2026-11-15; RLDD processed 2026-11-14).
- [x] File and video counts verified (DMD by audit; RLDD 144 videos, teammate-confirmed, folder structure checked).
- [x] Subject counts verified (DMD 14 in the copy, 13 with drowsiness sessions; RLDD 48 in scope).
- [x] Modalities and formats verified (DMD 1280 x 720, 29.76 fps; RLDD 15 fps features and 64 x 64 eye crops).
- [x] Conditions and native labels verified.
- [x] Logical shared paths recorded.

## Label mapping

- [x] UTA-RLDD native labels verified (0 alert, 5 low vigilant, 10 drowsy).
- [x] DMD native annotation names verified (48 labels).
- [x] Project mappings documented separately from native labels.
- [x] Mappings not used in the current scope are marked as such.

## Synthetic stress tests

- [x] Low-light generator validated on real frames.
- [x] Glare generator validated on real frames (after a fix).
- [x] Off-axis generator validated on real frames.
- [x] Partial-occlusion generator validated on real frames.
- [x] Face-lost case validated on real frames.
- [x] Tests use held-out people only (DMD subjects 1, 5, 6, 7).
- [x] Severity recorded for every sample.
- [x] Source IDs, parameters, and seeds recorded in the manifest.
- [ ] Synthetic results reported separately from natural results: model team, at evaluation time (instructions in `docs/R2_HANDOFF.md`).

## Evaluation handoff

- [x] RLDD participant-independent split documented (four folds, leave one fold out).
- [x] DMD split documented without person leakage.
- [x] Training and test roles decided (`docs/DECISIONS.md`).
- [x] Cells with fewer than five people flagged (stress set has 4 people).
- [ ] Metrics and baselines assigned to team members: model team (protocol suggested in `docs/R2_HANDOFF.md`).
- [x] Integration path recorded (`docs/R2_HANDOFF.md`, `code/README.md`).

## Package checks

- [x] README updated.
- [x] No raw dataset, video, frame, or stress image in the package.
- [x] No personal absolute path in the package.
- [x] No em dash, en dash, or curly quotes in the package.
- [x] No Drive link or credential in the package.
- [x] All CSV files parse with consistent columns.
- [x] All Python scripts compile and were run or smoke-tested.
