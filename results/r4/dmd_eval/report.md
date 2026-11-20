# R4 cues on DMD (RGB, unseen people)

Eye CNN trained only on MRL infrared crops; mouth thresholds from configs/r4.yaml. No DMD data was used for training or tuning.

## Eye closure, frame level (labelled open vs closed frames)

32908 labelled frames at 15 FPS; 32902 had a face and both-eye prediction (100.0%). Closed frames: 4440.

| group | n | n_people | accuracy | acc_ci_low | acc_ci_high | f1_closed | recall_closed | auroc | ece | person_acc_min | person_acc_median | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 32902 | 13 | 0.986 | 0.980 | 0.992 | 0.949 | 0.964 | 0.997 | 0.008 | 0.962 | 0.993 |  |
| split=test | 7879 | 4 | 0.981 | 0.968 | 0.996 | 0.929 | 0.967 | 0.997 | 0.006 | 0.962 | 0.983 | fewer than 5 people |
| split=train | 14221 | 5 | 0.984 | 0.975 | 0.992 | 0.941 | 0.949 | 0.995 | 0.012 | 0.973 | 0.982 |  |
| split=validation | 10802 | 4 | 0.993 | 0.989 | 0.998 | 0.975 | 0.982 | 0.999 | 0.009 | 0.987 | 0.996 | fewer than 5 people |
| glasses=no | 27255 | 10 | 0.987 | 0.980 | 0.993 | 0.950 | 0.966 | 0.998 | 0.005 | 0.962 | 0.993 |  |
| glasses=yes | 5647 | 3 | 0.982 | 0.973 | 0.993 | 0.949 | 0.957 | 0.997 | 0.026 | 0.973 | 0.982 | fewer than 5 people |

## Yawns, event level (merged yawn intervals)

| set | people | yawns | detected | recall | false alarms | per hour | median delay s | recall by type |
|---|---|---|---|---|---|---|---|---|
| all DMD people | 13 | 76 | 40 | 0.526 | 2 | 2.43 | 2.59 | {'without hand': '35/38', 'with hand': '3/29', 'merged': '2/9'} |
| R2 test split | 4 | 18 | 9 | 0.500 | 2 | 9.93 | 2.05 | {'without hand': '9/10', 'with hand': '0/7', 'merged': '0/1'} |

Per-session mouth calibration: median outer MAR over the first 30 s of each session. With-hand yawns hide the mouth, so MAR is expected to miss some of them; R6's temporal model and R5's head pose can help there.

## Stress set (R2, held-out people 1, 5, 6, 7)

Paired with the unstressed original of the same frame. `yawn_mar_kept` = median stressed/original inner-lip MAR on yawn frames.

| stress_type | severity | n | n_people | face_rate_original | face_rate_stressed | eye_acc_original | eye_acc_stressed | eye_scored_frac | yawn_mar_kept | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| face_lost | single_level | 24 | 4 | 1.000 | 0.083 | 1.000 | 0.000 | 0.062 | 0.000 | indicative (< 5 people) |
| glare | mild | 24 | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.994 | indicative (< 5 people) |
| glare | moderate | 24 | 4 | 1.000 | 0.750 | 1.000 | 1.000 | 0.875 | 0.271 | indicative (< 5 people) |
| glare | severe | 24 | 4 | 1.000 | 0.625 | 1.000 | 0.923 | 0.812 | 0.000 | indicative (< 5 people) |
| low_light | mild | 24 | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.988 | indicative (< 5 people) |
| low_light | moderate | 24 | 4 | 1.000 | 0.958 | 1.000 | 1.000 | 1.000 | 1.022 | indicative (< 5 people) |
| low_light | severe | 24 | 4 | 1.000 | 0.833 | 1.000 | 0.929 | 0.875 | 1.015 | indicative (< 5 people) |
| occlusion | mild | 72 | 4 | 1.000 | 0.972 | 1.000 | 1.000 | 0.979 | 1.031 | indicative (< 5 people) |
| occlusion | moderate | 72 | 4 | 1.000 | 0.958 | 1.000 | 0.915 | 0.979 | 1.064 | indicative (< 5 people) |
| occlusion | severe | 72 | 4 | 1.000 | 0.847 | 1.000 | 0.773 | 0.917 | 0.887 | indicative (< 5 people) |
| off_axis | mild | 24 | 4 | 1.000 | 1.000 | 1.000 | 0.938 | 1.000 | 1.010 | indicative (< 5 people) |
| off_axis | moderate | 24 | 4 | 1.000 | 0.917 | 1.000 | 1.000 | 0.938 | 1.031 | indicative (< 5 people) |
| off_axis | severe | 24 | 4 | 1.000 | 0.792 | 1.000 | 1.000 | 0.812 | 0.684 | indicative (< 5 people) |
