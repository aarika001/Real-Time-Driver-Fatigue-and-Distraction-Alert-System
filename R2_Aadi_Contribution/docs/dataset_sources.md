# UTA-RLDD and DMD Source Notes

Updated: 2026-11-15

Current working scope: UTA-RLDD and DMD only. The proposal v4 dataset list was reduced by the team decision. MRL Eye, NTHU-DDD, and YawDD are out of scope for the current implementation unless the team reopens the decision. The project is private internal course work and is not planned for publication or public distribution.

## UTA-RLDD

Sources:

- [UTA-RLDD official page](https://sites.google.com/view/utarldd/home)
- [UTA-RLDD paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/AMFG/Ghoddoosian_A_Realistic_Dataset_and_Baseline_Temporal_Model_for_Early_Drowsiness_CVPRW_2019_paper.pdf)

Provider and paper facts:

- Around 30 hours of RGB video.
- 60 healthy participants.
- 180 videos, with one video per participant for each of three states.
- The three states are alert, low vigilant, and drowsy.
- The official page reports five folds of 12 participants.
- The official page reports a total size of 111.3 GB.
- Videos were self-recorded using a phone or webcam.
- The paper states that frame rates are below 30 FPS.
- The paper states that recordings were made indoors in participant-selected environments and were not collected during actual driving.
- The paper describes approximately 10 minutes per video.
- The paper states that both eyes were intended to remain visible and the camera was approximately one arm's length away.
- Labels were supplied by participants based on their predominant state.
- The paper reports numeric labels of 0 for alert, 5 for low vigilant, and 10 for drowsy.
- The paper relates the states to the Karolinska Sleepiness Scale: alert corresponds to levels 1 to 3, low vigilant to levels 6 to 7, and drowsy to levels 8 to 9.
- The paper describes quality review of submitted videos and rejection of recordings that were clearly acted or did not follow the instructions.
- The provider recommends five-fold evaluation with one fold as test and four folds as training, repeated across all folds.
- Project decision (2026-11-15): only Fold1 to Fold4 (48 participants) are used. Fold5 is excluded, so evaluation is four-fold and the report must state the deviation.
- The provider requests citation of the associated paper and lists Reza Ghoddoosian as a contact.
- Terms checked 2026-11-15: the provider page states no licence or redistribution terms. Data are shared through a Google Drive link. Only 36 of the 60 participants allowed their faces to be published, and identities must not be revealed.

Project handling:

- Preserve the native class names and numeric labels.
- Do not describe UTA-RLDD as actual driving data.
- Verify the downloaded folder structure, fold membership, file count, duration, resolution, and frame-rate variation.
- Use participant-level separation for all splits.
- The current GitHub repository contains an RLDD preprocessing pipeline that maps the filename stems `0`, `5`, and `10` to `alert`, `low_vigilant`, and `fatigue`, respectively.
- The repository README describes 15 FPS `.npz` outputs, a master metadata CSV, face detection rate, head pose, mouth aspect ratio, eye quality, and 64 x 64 eye crops.
- The processed output on the shared Drive must be audited before these pipeline outputs are marked complete.

## DMD

Sources:

- [DMD official site](https://dmd.vicomtech.org/)
- [DMD provider repository](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset)
- [DMD paper](https://arxiv.org/abs/2008.12085)

Original dataset design described by the provider and paper:

- 37 drivers.
- Approximately 41 hours in the original multimodal collection.
- Real and simulated driving scenarios in the original collection.
- Face, body, and hands camera views.
- RGB, infrared, and depth channels in the original collection.
- VCD annotation format.
- Distraction activities such as phone use, texting, drinking, talking to a passenger, reaching, and radio operation.
- Gaze regions such as the front, windows, mirrors, radio zone, and lap zone.
- Fatigue-related labels including yawning, microsleep, and sleepy driving.

Current public-release restrictions:

- The provider states that simulator recordings, infrared material, and depth material were removed from the public release.
- Only RGB material from the real-car scenario is currently available.
- The provider states that the current data is divided into application downloads including gaze and hands, drowsiness, and distraction.
- The current application download sizes are not stated in the page text and must be checked on the download service.
- The original site's approximately 25 TB figure describes the historical raw collection, not the current required download.
- The provider requires academic use and lists CC BY-NC-ND 4.0.
- The licence requires attribution, prohibits commercial use, and prohibits distribution of modified versions.
- The provider lists `info-dmd@vicomtech.org` for questions.

Project handling:

- Download only the current RGB application material required by the team.
- Do not attempt to download the historical 25 TB collection.
- Verify the current available subjects, sessions, file count, annotations, and application package size.
- Keep generated stress outputs on the restricted team Drive. Do not distribute them externally.

## DMD Kaggle mirror (reviewed 2026-11-15)

- Source: [Kaggle dmd-video-v1](https://www.kaggle.com/datasets/momarberliansyah/dmd-video-v1), a third-party repackaged subset. Kaggle licence field: Unknown.
- Contents verified from the file list and annotations: 14 subjects, 45 sessions, 164 RGB videos, 60 OpenLABEL JSON files, 56.9 GB.
- Native drowsiness labels are yawning, eye state, and blinks. No microsleep or drowsy-state label is present.
- All drowsiness sessions were recorded with the car stopped.
- Use requires supervisor confirmation because the official terms prohibit redistribution without Vicomtech permission. Details: `docs/dmd_kaggle_source_review.md`.

## Current data interpretation

- UTA-RLDD is the naturalistic participant-recorded drowsiness source in the current scope, but it was not recorded during actual driving.
- DMD is the current source for distraction, gaze, and fatigue-related driving-monitoring labels, but the public revision is RGB-only real-car material.
- Results should be reported separately by dataset and condition because the two datasets differ in capture setting, labels, and data collection protocol.
- Final dataset counts, paths, access terms, and label mappings require verification against the current downloaded releases.
