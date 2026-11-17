# R4 cues on DMD (RGB, unseen people)

Eye CNN trained only on MRL infrared crops; mouth thresholds from configs/r4.yaml. No DMD data was used for training or tuning.

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
