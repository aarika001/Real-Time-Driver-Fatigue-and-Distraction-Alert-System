# RLDD Repository and Drive Status

Checked: 2026-11-14. This is a historical review of the pipeline repository. For the current status, see `docs/rldd_drive_check.md` (Drive check and pipeline owner's answers, 2026-11-15) and `docs/DECISIONS.md` (Fold5 excluded, repository public for now).

## GitHub repository findings

Repository: https://github.com/aarika001/Real-Time-Driver-Fatigue-and-Distraction-Alert-System

The current repository contains:

- `RLDD-preprocess/README.md`
- `RLDD-preprocess/preprocess-rldd.py`
- `face_landmarker.task`
- `.gitignore`

The latest visible commits are:

- `RLDD preprocessing pipeline with readme`, 2026-11-13
- `add gitignore`, 2026-11-14

The current repository `.gitignore` covers `processed_rldd/` and `.venv/`.
It does not cover general raw-data archives, videos, or generated stress
outputs. The R2 package has its own `.gitignore` that covers these. Add
similar rules to the repository root before placing local data near it.

The RLDD preprocessing README states that the pipeline:

- Reads Fold1 through Fold5 RLDD folders (the project uses Fold1 to Fold4; decision 2).
- Discovers `.mp4`, `.mov`, `.m4v`, and `.avi` files.
- Maps `0` to `alert`, `5` to `low_vigilant`, and `10` to `fatigue`.
- Samples to a target of 15 FPS by default.
- Extracts mouth aspect ratio, head pose, eye quality, face detection, and 64 x 64 eye crops.
- Writes compressed `.npz` archives and `rldd_metadata_index.csv`.
- Records face detection rate per processed video.

The same README still contains references to the original five-dataset plan,
including YawDD, MRL Eye, and NTHU-DDD in the downstream notes. Those
references are stale for the current UTA-RLDD and DMD scope and should be
updated by the repository owner.

The README also describes `video_path` and `npz_path` as full paths in the
processed metadata index. Do not commit unredacted personal device paths.
Use a relative path or a shared logical path in any metadata copied into the
private course package.

These are repository-documented pipeline capabilities. They are not proof
that every processed output was successfully generated.

## Drive status (updated 2026-11-15)

- The processed output for Fold1 to Fold4 is on the shared Drive: 48 participants, three NPZ files each, plus `rldd_metadata_index.csv`.
- The pipeline owner confirmed the latest commit, default settings (15 fps, 64 x 64 eye crops), no failed videos, and all 144 videos in the metadata.
- The original videos (about 100 GB) remain on the pipeline owner's storage.
- Details: `docs/rldd_drive_check.md`.
- `code/audit_rldd_processed.py` can confirm NPZ keys and shapes on a local sync (optional). `code/sanitize_rldd_metadata.py` removes personal paths from the metadata before it is shared.

## Stress-test implication

The RLDD pipeline produces feature archives and eye crops, not full frames. Whole-face stress tests would need the original videos. RLDD stress tests are out of the current scope (decision 9); the DMD stress set covers the R2 stress-test requirement.

## Repository visibility

The repository was public on 2026-11-14. The team decided to keep it public for now (decision 11). Only code, documentation, and metadata may be committed.
