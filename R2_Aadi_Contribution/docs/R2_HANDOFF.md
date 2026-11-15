# R2 Handoff to the Model-Training and Evaluation Team

From: Aadi Deshmukh (R2, Data Access and Stress Tests)
Date: 2026-11-15
Scope: UTA-RLDD and DMD (team decision). YawDD support was prepared separately for a teammate. It is outside the R2 scope unless the team reopens it.

## 1. What is ready

| Item | Where | Status |
|---|---|---|
| Decision log | `docs/DECISIONS.md` | 12 decisions recorded |
| DMD drowsiness face videos, 16 sessions, 13 people | `C:\NeuroNova_data\dmd\dmd_video_v1` (to be copied to the team Drive) | Downloaded and audited, 0 issues |
| DMD session inventory (45 sessions, 14 people) | `metadata/dmd_kaggle_session_inventory.csv` | Verified from Kaggle files |
| DMD native labels (48) and project mapping | `metadata/dmd_native_labels.csv`, `metadata/label_mapping.csv` | Labels verified; mapping adopted for the current scope (decision 7) |
| DMD subject split | `metadata/dmd_subject_split.csv` | Subject-independent, verified; adopted 2026-11-15 |
| DMD drowsiness events in seconds | `C:\NeuroNova_data\dmd\r2_manifests\` (restricted) | 7,538 intervals; 76 merged yawns; 285 eye closures |
| DMD stress-test set, held-out people only | `C:\NeuroNova_data\generated_stress\dmd\` (restricted), manifest `metadata/stress_test_manifest_dmd.csv` | 456 samples, validated |
| Stress validation report | `docs/stress_test_validation.md` | Done |
| RLDD processed output check | `docs/rldd_drive_check.md` | Fold1 to Fold4 present; pipeline owner confirms 144 videos, default settings, no failures; R2 audit of arrays pending |
| Tools | `code/` | Download, audit, extraction, stress generation |

## 2. DMD facts the model team needs

- **Source:** Kaggle copy `momarberliansyah/dmd-video-v1`, approved by the team. It is a 14-person subset of DMD, not the full release.
- **Video:** face camera, 1280 x 720, H.264, **29.76 fps** (744/25), about 3 minutes per drowsiness session.
- **Annotation frame numbers** use face-camera frames starting at 0. Seconds = frame / 29.76. Body and hands videos are offset by the per-session `body_frame_shift` and `hands_frame_shift`.
- **Native drowsiness labels:** `yawning/Yawning without hand`, `yawning/Yawning with hand`, `eyes_state/{open, closing, close, opening, undefined}`, `blinks/blinking`.
- **There is no native drowsy-state or microsleep label.** The uploader's session-level `drowsy=1` is not a DMD label. Do not train a drowsy-state classifier on it without naming it as a derived label.
- **One yawn often appears as two touching segments** (without hand, then with hand). Merge them before counting yawns. The merged file already does this.
- **All drowsiness sessions were recorded with the car stopped.** Report this as a limitation.

## 3. Splits

**DMD** (`metadata/dmd_subject_split.csv`):

| Split | Subjects | Drowsiness sessions | Distraction sessions | Gaze sessions |
|---|---|---|---|---|
| train | 23, 28, 29, 33, 36, 37 | 7 | 8 | 8 |
| validation | 9, 10, 13, 14 | 5 | 6 | 4 |
| test (held-out) | 1, 5, 6, 7 | 4 | **0** | 3 |

- No person appears in more than one split.
- **Decided (2026-11-15):** this split is adopted as is. The test split has no distraction sessions, so distraction is not evaluated on DMD in the current implementation. If added later, use subjects 13 and 14 as the distraction test set. See `docs/DECISIONS.md`.

**UTA-RLDD:** the team uses Fold1 to Fold4 only (48 participants, 12 per fold); Fold5 is excluded by team decision (2026-11-15). Use participant-independent fold-based evaluation over these four folds (for example, leave one fold out). Never put a participant in both train and test.

## 4. Suggested evaluation protocol

1. **Natural conditions first.** Report results per dataset; do not pool RLDD and DMD.
2. **RLDD:** fold-based and participant-independent over Fold1 to Fold4 (four-fold, leave one fold out). The provider recommends five folds, so state in the report that Fold5 was excluded. Classes are alert, low_vigilant, and fatigue. Fatigue is the project name for native drowsy.
3. **DMD drowsiness** (native events only, no derived drowsy-state label; decision 7):
   - Event-level yawn detection on the 76 merged yawns: recall and false alarms per hour.
   - Frame-level eye closure on the 285 intervals, if the model outputs eye state.
4. **Stress tests:**
   - Run the model on the 456 stressed images and the 24 originals.
   - Report the paired change per stress type and severity, with person counts (4 people), separately from natural results.
5. **Cells with fewer than five people** must be flagged as indicative.

## 5. Things that are not done and why

| Item | Blocker | Owner |
|---|---|---|
| RLDD array audit (`code/audit_rldd_processed.py`) | Optional; the pipeline owner already confirmed counts. Needs the Drive folder synced to a PC | Aadi (optional) |
| RLDD stress tests | Not in current scope (decision 9). Originals (about 100 GB) exist on the teammate's storage, so the teammate could run them locally later | Optional |
| DMD distraction and gaze | Not in current scope (decisions 4 and 5) | Optional later |
| Copy of DMD and stress outputs to the team Drive | Currently only on Aadi's PC | Aadi |

## 6. Data handling reminders

- Never commit videos, frames, stress images, DMD annotation JSON, or event CSVs to GitHub. The repository is public for now (decision 11), and the Drive folder is link-shared (decision 12), so do not post the Drive link anywhere.
- DMD is CC BY-NC-ND 4.0, academic use only. Cite the DMD paper and the Kaggle copy as the source. Do not redistribute.
- Sanitize `rldd_metadata_index.csv` with `code/sanitize_rldd_metadata.py` before sharing. It contains personal drive paths.
- Do not show RLDD participant faces in the report: only 36 of 60 participants consented to face publication.
- Cite UTA-RLDD (Ghoddoosian et al., CVPRW 2019) and DMD (Ortega et al., ECCVW 2020).
