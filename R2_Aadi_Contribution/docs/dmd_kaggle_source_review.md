# DMD Kaggle Source Review

Reviewed: 2026-11-15 by Aadi Deshmukh (R2)
Method: Kaggle file listing, the uploader's README and manifests, and all 60 OpenLABEL annotation files were read through the Kaggle API before any download. The drowsiness face videos were downloaded and audited later the same day (see Download record).

## Source

- Kaggle dataset: `momarberliansyah/dmd-video-v1` (https://www.kaggle.com/datasets/momarberliansyah/dmd-video-v1), version 1, last updated 2026-08-12.
- Size: 227 files, 56.9 GB (164 videos, 60 annotation JSON files, 3 manifest or README files).
- Kaggle licence field: `Unknown`.
- Uploader statement (paraphrased): the folder was assembled from local DMD tar.gz archives for Kaggle training. The session-level labels were inferred by the uploader from the archive or annotation type.

## What the Kaggle copy contains

- **This is a partial, repackaged subset of DMD, not the full public release.**
- 14 subjects: 1, 5, 6, 7, 9, 10, 13, 14, 23, 28, 29, 33, 36, 37 (groups gA, gB, gC, gE, gF, gZ).
- 45 sessions: 16 drowsiness (`s5`), 14 distraction (`s1`, `s2`, `s3`), 15 gaze and hands (`s6`).
- Per session: face, body, and (except drowsiness) hands RGB `.mp4`, plus a `mosaic.avi` combining the views.
- Annotations are DMD OpenLABEL JSON (`openlabel` root, schema 1.0.0, ontology `http://dmd.vicomtech.org/ontology`), one file per session, plus a separate `hands.json` for gaze sessions.
- The uploader's split is train 23 sessions, validation 15, test 7. **It is subject-independent: no subject appears in more than one split.** Test subjects: 1, 5, 6, 7. Validation: 9, 10, 13, 14. Train: 23, 28, 29, 33, 36, 37.
- All 164 videos and 60 annotation files listed in the uploader manifest exist in the dataset.
- Full per-session details: `metadata/dmd_kaggle_session_inventory.csv`.

## Native labels found (verified from the annotation files)

Full list with counts: `metadata/dmd_native_labels.csv` (48 native labels).

- Drowsiness sessions: `yawning/Yawning with hand`, `yawning/Yawning without hand`, `eyes_state/{open, closing, close, opening, undefined}`, `blinks/blinking`. 85 yawn intervals across all 16 sessions.
- **No native microsleep or drowsy-state label is present in this copy.** The uploader's `drowsy=1` column is a session-level guess applied to every drowsiness session, not a DMD label.
- Distraction sessions: `driver_actions/*` (13 actions including safe_drive, texting and phone calls left and right, radio, drinking, reach_side, reach_backseat, hair_and_makeup, talking_to_passenger), `gaze_on_road/*`, `hands_using_wheel/*`, `talking/talking`, `hand_on_gear`.
- Gaze sessions: `gaze_zone/*` (9 zones plus `not_valid`) and `blinks/blinking`. The hands file has `hands_on_wheel/*`, `moving_hands/*`, and `transition/*`.

## Capture facts from the annotation files

- Cameras: face Intel RealSense D415, body D435, hands D415. All streams are 1280 x 720.
- **Annotation frame numbers follow the face camera.** The body and hands streams have a per-session `frame_shift` (for example 54 and 74 frames in `gA_1_s5`). Recorded per session in the inventory.
- **All 16 drowsiness sessions were recorded with the car stopped** (`setup = Car Stopped`). Distraction `s1` sessions were recorded with the car moving.
- Frame rate is not stated in the annotation files. Verified from the downloaded face videos on 2026-11-15: 29.76 fps (744/25), 1280 x 720, H.264, about 3 minutes per drowsiness session. Video frame counts equal the annotation frame counts exactly.

## Decisions recorded (2026-11-15)

- Aadi Deshmukh (R2) confirmed that the team will use this Kaggle copy for the course project. No other DMD source will be downloaded.
- The GitHub repository may stay public for now, by team decision. Restricted files (videos, annotation JSON, event CSVs) still stay off GitHub.
- Supervisor awareness of the source is still recommended and should be noted in the report.

## Licence and provenance concerns

- The official DMD licence is CC BY-NC-ND 4.0, academic use only, users 18 or older. The official site also says reproduction or transfer of site contents is prohibited without Vicomtech's written permission, and that material from some participants was removed after a licence revision.
- This Kaggle copy is a third-party redistribution with licence `Unknown`. We cannot confirm that it matches the current official release, or that it excludes participants who were withdrawn.
- Handling: record the mirror as the source in the report, cite the DMD paper, keep all files on the restricted team Drive, and do not redistribute.

## Implications for R2 and the main project

- DMD results from this copy cover 14 subjects only and must not be described as the full DMD.
- DMD drowsiness here supports yawn and eye-closure events, not a drowsy-state classifier. Any drowsy label would be our own derived label and must be named as such.
- Use the uploader's subject-independent split, or define our own subject-level split from the inventory. Never split by session.
- Held-out people for stress tests should come from the test subjects (1, 5, 6, 7) unless the team chooses a different split.
- Use the face video for face-based models. If body or hands video is used, apply the recorded frame shift.

## Fatigue events and split (prepared 2026-11-15)

- `dmd_fatigue_events.csv` (restricted, keep on the team Drive): 370 intervals from the 16 drowsiness sessions. 85 yawn segments (47 without hand, 38 with hand) and 285 eye-closure intervals, in face-camera frames.
- A single yawn is often annotated as a "without hand" segment followed immediately by a "with hand" segment. Merge adjacent segments before counting yawn events (`code/extract_dmd_openlabel.py --merge-yawn-phases`).
- `metadata/dmd_subject_split.csv`: subject-level split with session and event counts.
- **The uploader test split (subjects 1, 5, 6, 7) has no distraction sessions.** Distraction can only be tested on held-out people by using the validation subjects (13, 14) as the distraction test set, or by re-splitting. Team decision needed.
- Subject 28 has distraction and gaze sessions but no drowsiness session.

## Download options

Script: `code/download_dmd_kaggle_subset.py` (requires the user's own Kaggle API token, which the script never reads or stores).

| Selection | Files | Approximate size |
|---|---|---|
| Annotations and manifests only | 63 | 79 MB |
| Drowsiness face videos and annotations | 35 | 5.75 GB |
| All face videos and annotations | 108 | 18.4 GB |
| Everything | 227 | 56.8 GB |

## Download record (2026-11-15)

- Downloaded: 16 drowsiness face videos and their 16 annotation files, plus the README and manifests (35 files, 5.4 GB), with `code/download_dmd_kaggle_subset.py`.
- Local location: `C:\NeuroNova_data\dmd\dmd_video_v1` on Aadi's PC. Logical team path: `shared Google Drive:/NeuroNova/datasets/external/dmd`.
- Audit (`code/audit_dmd_download.py`): 32 of 32 downloaded data files present, sizes match, 0 issues.
- Event files written next to the data in `r2_manifests/`: `dmd_drowsiness_events_all.csv` (7,538 intervals with seconds at 29.76 fps) and `dmd_yawn_events_merged.csv` (76 yawns after merging hand and no-hand phases; 37 train, 21 validation, 18 test; 2.3 to 9.3 s, median 5.3 s).
- Not downloaded: distraction and gaze videos, and body, hands, and mosaic views.

## Remaining

- Copy of the downloaded subset to the restricted team Drive.
