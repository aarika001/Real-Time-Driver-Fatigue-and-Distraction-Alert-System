# R2 Native Label and QA Guidelines

## Scope

The current working scope contains UTA-RLDD and DMD only. No manual annotation is done by R2; only dataset-native labels are used.

## Native labels

- Preserve UTA-RLDD native labels: alert, low vigilant, and drowsy.
- Preserve DMD native annotation names exactly as they appear in the OpenLABEL files (the successor of the VCD format).
- Do not silently convert native labels into project labels.
- Record project mappings in `metadata/label_mapping.csv`.

## Mapping rules

- Mark mappings as Confirmed only after the current dataset documentation or files are inspected.
- Mark uncertain mappings as Provisional or Pending team decision.
- Record whether a label describes fatigue, distraction, gaze, an action, a condition, or metadata.
- Preserve multiple native signals when collapsing them would lose information.

## Current-scope mapping (2026-11-15)

- RLDD: alert to alert, low vigilant to low_vigilant, drowsy to fatigue (decision 8).
- DMD: merged yawns and `eyes_state/close` intervals are fatigue cue events. No derived drowsy-state label is created (decision 7).
- DMD distraction, gaze, and hands labels are mapped for later use but not used now (decisions 4 and 5).

## QA

- Verify that native labels exist in the current downloaded release.
- Verify that the label unit is video-level, segment-level, frame-level, or spatial.
- Verify that no person is present in both training and test splits.
- Record all unresolved labels and conditions in the metadata index.
