# R2 Code

Run every script from the package root (`R2_Aadi_Contribution`). Data paths must point outside this folder; the download and stress scripts refuse to write inside a Git repository.

## Requirements

```text
Python 3.10 or later
numpy
Pillow
opencv-python 4.x      (build_dmd_stress_set.py; needs CascadeClassifier)
kaggle and six         (download_dmd_kaggle_subset.py)
ffprobe (FFmpeg) or opencv-python   (audit_dmd_download.py)
```

## DMD

| Script | Purpose |
|---|---|
| `download_dmd_kaggle_subset.py` | Download selected parts of the approved Kaggle copy. Needs your own Kaggle API token in `KAGGLE_API_TOKEN`; the script never reads or prints it. |
| `audit_dmd_download.py` | Check a download against `metadata/dmd_kaggle_session_inventory.csv`: presence, size, fps, frame counts. |
| `extract_dmd_openlabel.py` | Convert DMD annotation JSON into an event CSV, with optional seconds (`--fps 29.76`) and merged yawn phases (`--merge-yawn-phases`). |
| `build_dmd_stress_set.py` | Build and measure the stress-test set from held-out DMD people. |

Commands used on 2026-11-15:

```powershell
python code/download_dmd_kaggle_subset.py --dest "C:\NeuroNova_data\dmd" --tasks drowsiness --views face
python code/audit_dmd_download.py --root "C:\NeuroNova_data\dmd\dmd_video_v1" --output "C:\NeuroNova_data\dmd\dmd_download_audit.json"
python code/extract_dmd_openlabel.py --root "C:\NeuroNova_data\dmd\dmd_video_v1" --out "C:\NeuroNova_data\dmd\r2_manifests\dmd_drowsiness_events_all.csv" --fps 29.76
python code/extract_dmd_openlabel.py --root "C:\NeuroNova_data\dmd\dmd_video_v1" --out "C:\NeuroNova_data\dmd\r2_manifests\dmd_yawn_events_merged.csv" --groups yawning --merge-yawn-phases --fps 29.76
python code/build_dmd_stress_set.py --root "C:\NeuroNova_data\dmd\dmd_video_v1" --events "C:\NeuroNova_data\dmd\r2_manifests\dmd_drowsiness_events_all.csv" --out "C:\NeuroNova_data\generated_stress" --subjects 1
python code/build_dmd_stress_set.py --root x --events x --out "C:\NeuroNova_data\generated_stress" --merge
```

The stress builder was run once per held-out subject (1, 5, 6, 7) and then with `--merge`.

## UTA-RLDD

| Script | Purpose |
|---|---|
| `audit_rldd_processed.py` | Check the processed RLDD folder: metadata columns, label, fold, and subject counts, missing NPZ files, and NPZ keys and shapes for a sample. Paths in the report are reduced to relative paths or basenames. |
| `sanitize_rldd_metadata.py` | Write a copy of `rldd_metadata_index.csv` with relative paths or basenames, removing personal device paths. |

```powershell
python code/audit_rldd_processed.py --processed-dir "<synced Drive path>\processed_rldd" --output "<synced Drive path>\rldd_processed_audit.json"
python code/sanitize_rldd_metadata.py --metadata "<synced Drive path>\processed_rldd\rldd_metadata_index.csv" --root "<synced Drive path>\processed_rldd" --output "<synced Drive path>\rldd_metadata_index_sanitized.csv"
```

Use `--sample-limit 0` for a metadata-only audit, or a larger value to open more NPZ files.

## Stress generator

`stress_generators.py` applies one stress to one image and is used by `build_dmd_stress_set.py`.

```powershell
python code/stress_generators.py --input input.png --output output.png --stress glare --severity moderate --seed 42 --face-box 480 300 260 260
```

- Stress types: `low_light`, `glare`, `off_axis`, `occlusion`, `face_lost`
- Severities: `mild`, `moderate`, `severe`
- `--face-box X Y W H` positions glare, occlusion, and face lost
- `--occlusion-kind`: `mask`, `cap_brim`, or `hand_patch`

The Python interface is `apply_stress(image, stress_type, severity, seed, face_box, direction, occlusion_kind)`.
