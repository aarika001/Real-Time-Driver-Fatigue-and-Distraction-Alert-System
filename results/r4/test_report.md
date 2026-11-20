# Eye-state CNN: test on unseen people

Best epoch 11 | temperature 0.948 | 7 test people, 20333 crops

| group | n | n_people | accuracy | acc_ci_low | acc_ci_high | f1_closed | recall_closed | auroc | ece | person_acc_min | person_acc_median | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 20333 | 7 | 0.978 | 0.946 | 0.993 | 0.979 | 0.982 | 0.997 | 0.008 | 0.916 | 0.959 |  |
| dataset=mrl | 20333 | 7 | 0.978 | 0.946 | 0.993 | 0.979 | 0.982 | 0.997 | 0.008 | 0.916 | 0.959 |  |
| glasses=no | 13506 | 6 | 0.991 | 0.982 | 0.999 | 0.991 | 0.992 | 1.000 | 0.004 | 0.975 | 0.984 |  |
| glasses=yes | 6827 | 5 | 0.952 | 0.920 | 0.962 | 0.961 | 0.966 | 0.987 | 0.019 | 0.916 | 0.939 |  |
| reflection=high | 1386 | 4 | 0.957 | 0.863 | 0.980 | 0.957 | 0.942 | 0.987 | 0.027 | 0.842 | 0.923 | fewer than 5 people |
| reflection=low | 1102 | 6 | 0.920 | 0.901 | 0.985 | 0.951 | 0.908 | 0.996 | 0.038 | 0.899 | 0.982 |  |
| reflection=none | 17845 | 7 | 0.983 | 0.972 | 0.997 | 0.984 | 0.993 | 0.999 | 0.009 | 0.959 | 0.978 |  |
| lighting=bad | 16978 | 7 | 0.975 | 0.943 | 0.992 | 0.976 | 0.980 | 0.997 | 0.009 | 0.916 | 0.960 |  |
| lighting=good | 3355 | 6 | 0.990 | 0.986 | 0.999 | 0.993 | 0.992 | 0.999 | 0.004 | 0.875 | 0.994 |  |
| sensor=realsense | 20333 | 7 | 0.978 | 0.946 | 0.993 | 0.979 | 0.982 | 0.997 | 0.008 | 0.916 | 0.959 |  |

Worst condition (>= 5 people): reflection=low (recall_closed 0.908)