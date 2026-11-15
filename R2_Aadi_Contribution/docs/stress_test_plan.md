# Synthetic Stress-Test Plan: UTA-RLDD and DMD

## Purpose

Create controlled stress tests without collecting new human-subject data. Tests must run on held-out people and be reported separately from natural dataset conditions.

Status (2026-11-15): implemented on DMD held-out subjects 1, 5, 6, and 7 (456 samples). RLDD stress tests are out of the current scope (decision 9). Results and validated parameters: `docs/stress_test_validation.md`. Manifest: `metadata/stress_test_manifest_dmd.csv`.

## Inputs

- Held-out people only.
- Source dataset and person or video IDs recorded in a manifest.
- Original data preserved.
- Generated outputs are restricted internal course-evaluation artifacts and must remain on the team Drive.

## Required transformations

| Stress test | Operation | Required metadata |
|---|---|---|
| Low light | Gamma change and noise | Source ID, gamma, noise setting, severity |
| Glare | Saturated blob with halo on the eye band of the face box (updated 2026-11-15 after validation) | Source ID, blob placement and size, severity |
| Off-axis camera | Perspective warp up to `+/-30` degrees | Source ID, angle, severity |
| Partial occlusion | Mask, cap brim, or hand-sized patches | Source ID, patch type and placement, severity |
| Face lost | Remove or invalidate the face observation | Source ID, failure type |

## Rules

- Apply transformations only to held-out people.
- Do not place the same person in both train and test through a transformed copy.
- Keep synthetic results separate from natural-condition results.
- Report severity and affected-person counts.
- Do not claim that synthetic stress is equivalent to real camera degradation.
- The current DMD release is licence-restricted. Keep transformed DMD outputs on the restricted team Drive and do not distribute them externally.

## Manifest

Required fields:

```text
source_dataset
source_person_id
source_video_or_image_id
split
stress_type
severity
parameter_summary
random_seed
output_path
```

Levels: `mild`, `moderate`, and `severe` (face lost has a single level).

Seeds: base seed `42`; the DMD set uses `42 + 100 x subject + frame index`, recorded per row.

The DMD manifest adds five columns after the required ones: `source_frame`, `source_state`, `face_box_xywh`, `face_box_source`, and `variant`.

Generator interface:

`apply_stress(image, stress_type, severity, seed, face_box, direction, occlusion_kind)`

## R2 handoff

R2 verifies that every generated sample has traceable source and stress metadata and provides the integration owner with the generator and manifest contract.
