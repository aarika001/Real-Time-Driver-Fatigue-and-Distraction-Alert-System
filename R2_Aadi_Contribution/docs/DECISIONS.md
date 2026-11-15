# R2 Decision Log

Decisions were taken by Aadi Deshmukh (R2) on 2026-11-15 on behalf of the team, choosing the simplest option because of network and time constraints. The team or supervisor may revisit any of them.

| # | Topic | Decision | Reason | What it means |
|---|---|---|---|---|
| 1 | Dataset scope | UTA-RLDD and DMD only | Earlier team decision | Proposal v4 listed five datasets. Supervisor informed on 2026-11-15. |
| 2 | RLDD folds | Fold1 to Fold4 only (48 participants); Fold5 excluded | Team decision | Four-fold, leave-one-fold-out evaluation. The report states the deviation from the provider's five-fold protocol. |
| 3 | DMD source | Kaggle copy `momarberliansyah/dmd-video-v1` only | Team decision | 14-person subset. Cite DMD and the Kaggle copy. |
| 4 | DMD data used | Drowsiness face videos only (16 sessions, already downloaded) | No further downloads | No distraction, gaze, body, or hands videos. |
| 5 | DMD distraction evaluation | Not evaluated on DMD in the current implementation | The test split has no distraction sessions, and adding them needs new downloads | The report states that distraction was not evaluated on DMD. If added later, use subjects 13 and 14 as the distraction test set. |
| 6 | DMD split | Keep the Kaggle uploader split: train 23, 28, 29, 33, 36, 37; validation 9, 10, 13, 14; test 1, 5, 6, 7 | Already subject-independent and verified | `metadata/dmd_subject_split.csv` status changed to Adopted. |
| 7 | DMD labels | Use native events only: merged yawns and eye-closure intervals as fatigue cues. No derived drowsy-state label | DMD has no native drowsy-state label; deriving one needs extra work and justification | DMD evaluation is event-level (yawn recall and false alarms per hour; eye-closure detection). |
| 8 | RLDD labels | Native classes: alert, low_vigilant, and drowsy (called fatigue in the project) | Already used by the pipeline | No change. |
| 9 | Stress tests | DMD held-out set only (456 samples, subjects 1, 5, 6, 7). No RLDD stress tests | RLDD originals are about 100 GB on a teammate's storage, too large to share | Stress results are reported for DMD only. Optional later: the teammate could run the stress builder locally on one held-out fold, sharing only the small outputs. |
| 10 | YawDD | Stays outside R2 scope | Reduces work | The yawn CSV already given to the teammate is used at their discretion and reported by them. |
| 11 | GitHub visibility | Public for now | Team decision | Restricted files (videos, frames, stress images, DMD annotations, event CSVs) never go on GitHub. |
| 12 | Shared Drive access | The `processed_rldd` folder stays viewable by anyone with the link | Team decision | The link is not posted publicly or added to the repository. |
