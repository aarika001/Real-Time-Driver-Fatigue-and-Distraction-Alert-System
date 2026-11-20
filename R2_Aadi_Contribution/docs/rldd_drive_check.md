# RLDD Processed Output: Shared Drive Check

Checked: 2026-11-15 by Aadi Deshmukh (R2), through a browser that was not signed in to Google.
Scope: folder listings and the Drive preview of `rldd_metadata_index.csv` only. No NPZ file was opened. This is not the full audit.

## Access finding

- The `processed_rldd` folder, its subfolders, the NPZ files, and `rldd_metadata_index.csv` were all viewable **without signing in**. The folder is shared as "anyone with the link", not restricted to team members and supervisors.
- Re-checked on 2026-11-15 at 19:52 IST after a restriction attempt: the folder and its file list were still viewable without signing in.
- The team accepted link sharing for now (decision 12 in `docs/DECISIONS.md`). The link must not be posted publicly or added to the repository.

## What is on the Drive

| Item | Observed |
|---|---|
| Fold folders | Fold1, Fold2, Fold3, Fold4 |
| Fold5 | Not present; excluded by team decision (2026-11-15) |
| Subject folders | 12 per fold, subject_01 to subject_48 |
| Files per subject (spot check of subject_48) | `kss_0_alert.npz`, `kss_5_low_vigilant.npz`, `kss_10_fatigue.npz` |
| Metadata | `rldd_metadata_index.csv`, 22 KB, modified 14 Nov |
| All folders modified | 14 Nov |

Fold5 (participants 49 to 60) is out of scope by team decision on 2026-11-15. The working RLDD set is Fold1 to Fold4: 48 participants and 144 expected NPZ files (3 per participant).

## Metadata preview (first 49 rows only)

- Columns (the Drive preview renders the file as a table, so the exact delimiter was not confirmed): `fold_id, subject_id, raw_kss, label, video_path, npz_path, num_frames_15fps, face_detection_rate`.
- `video_path` holds a teammate's personal absolute path (`/media/<user>/<drive>/archive/...`). Run `code/sanitize_rldd_metadata.py` before any copy goes into the repository.
- `npz_path` is relative (`./processed_rldd/FoldN/subject_XX/kss_<k>_<label>.npz`).
- Label mapping matches the pipeline: 0 to alert, 5 to low_vigilant, 10 to fatigue.
- Original video files mix `.mov`, `.MOV`, and `.mp4`.
- In the preview rows, `num_frames_15fps` is about 7,200 to 9,100 (8 to 10 minutes at 15 fps) and `face_detection_rate` is close to 1.0.

## Pipeline owner's answers (received 2026-11-15)

| Question | Answer (teammate-reported) |
|---|---|
| Pipeline version | Latest commit at processing time. The last commit recorded on 2026-11-14 is `754cb7570b7a` ("add gitignore"); the pipeline code itself dates from `6636f0cfc417` ("RLDD preprocessing pipeline with readme"). |
| Settings | Defaults: 15 fps target, 64 x 64 eye crops |
| Failed or skipped videos | None. All videos in Fold1 to Fold4 were processed. |
| Metadata completeness | All 144 videos for Fold1 to Fold4 are in `rldd_metadata_index.csv` |
| Original videos | Still available on the teammate's own storage (about 100 GB). Too large to share. The teammate can run processing locally if needed. |
| Drive folder restriction | Not needed; link sharing accepted by team decision (decision 12) |

These answers are teammate-reported. The R2 audit script can confirm counts and array shapes once the folder is synced locally.

## Optional full audit

1. Sync `processed_rldd` locally (Google Drive for desktop), then run:
   `python code/audit_rldd_processed.py --processed-dir "<synced path>/processed_rldd" --output "<synced path>/rldd_processed_audit.json"`
   Add `--metadata` if the CSV is not inside the processed folder.
2. Run `code/sanitize_rldd_metadata.py`.
