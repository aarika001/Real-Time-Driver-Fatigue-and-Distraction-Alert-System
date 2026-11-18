
import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from .common import load_baselines, extract_feature_matrix


DEFAULT_METADATA = (
    "data/processed/mirror_sqpnp_full/mirror_metadata.csv"
)

DEFAULT_BASELINES = (
    "data/processed/mirror_sqpnp_full/calibration/"
    "driver_baselines_r5.csv"
)

DEFAULT_OUTPUT = "outputs/r6_dataset_audit"

WINDOW = 75
STRIDE = 15
MIN_FACE_RATE = 0.60


def audit_recordings(metadata_path, baseline_path, output_dir):
    metadata = pd.read_csv(metadata_path)
    baselines = load_baselines(baseline_path)

    required = {"subject_id", "npz_path"}
    missing = required - set(metadata.columns)

    if missing:
        raise ValueError(
            f"Metadata is missing required columns: {sorted(missing)}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    reference_features = None
    schema_mismatches = []

    for index, rec in metadata.iterrows():
        sid = str(rec["subject_id"])
        npz_path = Path(str(rec["npz_path"]))

        row = {
            "metadata_row": index,
            "subject_id": sid,
            "source_label": rec.get("source_label", ""),
            "npz_path": str(npz_path),
            "metadata_frames": rec.get("num_frames", ""),
            "metadata_face_rate": rec.get(
                "face_detection_rate", ""
            ),
            "metadata_pose_rate": rec.get(
                "pose_valid_rate", ""
            ),
            "baseline_available": sid in baselines.index,
            "npz_exists": npz_path.is_file(),
            "actual_frames": 0,
            "feature_count": 0,
            "feature_names": "",
            "actual_face_rate": np.nan,
            "possible_windows": 0,
            "accepted_windows": 0,
            "rejected_windows": 0,
            "status": "not_checked",
            "reason": "",
        }

        reasons = []

        if sid not in baselines.index:
            reasons.append("no_neutral_baseline")

        if not npz_path.is_file():
            reasons.append("npz_file_missing")

        if reasons:
            row["status"] = "rejected"
            row["reason"] = ";".join(reasons)
            results.append(row)
            continue

        try:
            features, names, detected = extract_feature_matrix(
                npz_path,
                baselines.loc[sid]
            )

            frame_count = len(features)
            face_rate = (
                float(np.mean(detected))
                if len(detected) else 0.0
            )

            row["actual_frames"] = frame_count
            row["feature_count"] = len(names)
            row["feature_names"] = "|".join(names)
            row["actual_face_rate"] = face_rate

            if reference_features is None:
                reference_features = names
            elif names != reference_features:
                reasons.append("feature_schema_mismatch")
                schema_mismatches.append(sid)

            if frame_count < WINDOW:
                reasons.append("recording_shorter_than_window")
                row["status"] = "rejected"
                row["reason"] = ";".join(reasons)
                results.append(row)
                continue

            possible = (
                (frame_count - WINDOW) // STRIDE
            ) + 1

            accepted = 0

            for start in range(
                0,
                frame_count - WINDOW + 1,
                STRIDE
            ):
                window_detection = float(
                    np.mean(detected[start:start + WINDOW])
                )

                if window_detection >= MIN_FACE_RATE:
                    accepted += 1

            row["possible_windows"] = possible
            row["accepted_windows"] = accepted
            row["rejected_windows"] = possible - accepted

            if accepted == 0:
                reasons.append("no_windows_pass_face_threshold")

            if reasons:
                row["status"] = "rejected"
                row["reason"] = ";".join(reasons)
            else:
                row["status"] = "ready"
                row["reason"] = "passed_checks"

        except Exception as exc:
            row["status"] = "error"
            row["reason"] = (
                f"{type(exc).__name__}: {exc}"
            )

        results.append(row)

    audit = pd.DataFrame(results)

    csv_path = output_dir / "r6_recording_audit.csv"
    audit.to_csv(csv_path, index=False)

    status_counts = Counter(audit["status"])
    reason_counts = Counter()

    for reason_string in audit["reason"].fillna(""):
        for reason in reason_string.split(";"):
            if reason and reason != "passed_checks":
                reason_counts[reason] += 1

    summary = {
        "metadata_file": str(metadata_path),
        "baseline_file": str(baseline_path),
        "total_recordings": int(len(audit)),
        "unique_subjects": int(
            audit["subject_id"].nunique()
        ),
        "baseline_subjects": int(len(baselines)),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
        "total_possible_windows": int(
            audit["possible_windows"].sum()
        ),
        "total_accepted_windows": int(
            audit["accepted_windows"].sum()
        ),
        "total_rejected_windows": int(
            audit["rejected_windows"].sum()
        ),
        "feature_schema": reference_features,
        "schema_mismatch_recordings": len(schema_mismatches),
        "face_detection_threshold": MIN_FACE_RATE,
        "window_frames": WINDOW,
        "stride_frames": STRIDE,
        "note": (
            "Readiness checks only. This audit does not establish "
            "validity of vigilance labels or readiness for "
            "supervised three-class GRU training."
        ),
    }

    summary_path = output_dir / "r6_audit_summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n========== R6 DATASET AUDIT ==========")
    print("Total recordings:", summary["total_recordings"])
    print("Unique subjects:", summary["unique_subjects"])
    print("Baseline subjects:", summary["baseline_subjects"])

    print("\n--- RECORDING STATUS ---")
    for name, count in status_counts.items():
        print(f"{name}: {count}")

    print("\n--- REASONS ---")
    for name, count in reason_counts.items():
        print(f"{name}: {count}")

    print("\n--- WINDOW COUNTS ---")
    print("Possible:", summary["total_possible_windows"])
    print("Accepted:", summary["total_accepted_windows"])
    print("Rejected:", summary["total_rejected_windows"])

    print("\n--- OUTPUT FILES ---")
    print(csv_path.resolve())
    print(summary_path.resolve())


def main():
    parser = argparse.ArgumentParser(
        description="Audit R6 temporal training data readiness."
    )

    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA
    )

    parser.add_argument(
        "--baselines",
        default=DEFAULT_BASELINES
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT
    )

    args = parser.parse_args()

    audit_recordings(
        args.metadata,
        args.baselines,
        args.output
    )


if __name__ == "__main__":
    main()
