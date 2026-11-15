# Dataset Card: UTA-RLDD and DMD (R2 Scope)

## Project

- Project: Real-Time Driver Fatigue and Distraction Alert System
- Team: NeuroNova
- R2 owner: Aadi Deshmukh
- R2 role: Data access and stress tests
- Last updated: 2026-11-15

## Scope note

Proposal v4 listed five public datasets. The team limited the implementation to UTA-RLDD and DMD, and the supervisor has been informed. All scope choices are in `docs/DECISIONS.md`.

## Datasets

| | UTA-RLDD | DMD (Kaggle copy) |
|---|---|---|
| Source | https://sites.google.com/view/utarldd/home | https://www.kaggle.com/datasets/momarberliansyah/dmd-video-v1 (official site https://dmd.vicomtech.org/) |
| Full release | 60 participants, 180 RGB videos of about 10 minutes, five folds of 12, 111.3 GB | Original DMD: 37 drivers; the Kaggle copy holds 14 people, 45 sessions, 56.9 GB |
| In project scope | Fold1 to Fold4: 48 participants, 144 videos | 16 drowsiness sessions, 13 people, face camera only (5.4 GB) |
| Capture | Self-recorded indoors with a phone or webcam; not real driving | Real car, RealSense cameras; all drowsiness sessions recorded with the car stopped |
| Video format | Mixed `.mov` and `.mp4`; processed to 15 fps features and 64 x 64 eye crops | 1280 x 720 H.264 at 29.76 fps (verified) |
| Native labels | Video-level: alert (0), low vigilant (5), drowsy (10), self-reported | Interval-level OpenLABEL: yawning (with or without hand), eye state, blinks; no drowsy-state label |
| Project use | alert, low_vigilant, fatigue classes | Yawn and eye-closure events as fatigue cues; stress tests |
| Split | Four folds, leave one fold out, participant-independent | Subject-independent: train 6, validation 4, test 4 people |
| Terms | No licence stated; cite the 2019 CVPRW paper; faces of only 36 of 60 participants may be shown | Academic use, CC BY-NC-ND 4.0; cite the ECCVW 2020 paper |
| Status | Processed output on the shared Drive; teammate confirmed 144 videos, default settings, no failures | Downloaded and audited, 0 issues |

## Project labels

- `alert`: RLDD native alert
- `low_vigilant`: RLDD native low vigilant
- `fatigue`: RLDD native drowsy; DMD yawn and eye-closure events are used as fatigue cues
- `distraction`: defined in the label mapping for later use. It is not evaluated on DMD in the current implementation (decision 5).

Native labels are preserved. Mappings are in `metadata/label_mapping.csv`.

## Synthetic stress tests

The tests use DMD held-out people only (subjects 1, 5, 6, 7) and are reported separately from natural conditions: low light, glare, off-axis, partial occlusion (mask, cap brim, hand patch), and face lost, each at mild, moderate, and severe where applicable. There are 456 samples in total. All five transforms were validated on real frames (`docs/stress_test_validation.md`).

## Limitations

- UTA-RLDD was recorded indoors with consumer cameras, not during driving, and its labels are self-reported.
- UTA-RLDD Fold5 is excluded, so evaluation differs from the provider's five-fold protocol.
- The DMD Kaggle copy is a 14-person subset with an unknown Kaggle licence field. DMD drowsiness sessions were recorded with the car stopped.
- DMD has no native drowsy-state label, so DMD evaluation is event-level only.
- The stress test set has 4 people, which is fewer than five per cell, so its results are indicative only.
- Synthetic stress approximates camera degradation and is not equivalent to real conditions.
- The two datasets differ in labels, capture settings, and populations, so results are reported per dataset.
- The system is a course research prototype, not a certified safety system.
