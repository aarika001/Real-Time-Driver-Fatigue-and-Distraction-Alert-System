"""Audit processed UTA-RLDD artifacts without exposing local absolute paths.

The script checks the metadata index and a configurable number of processed
NPZ files. It does not copy, modify, or upload dataset files.
"""

from __future__ import annotations

import argparse
import re
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_COLUMNS = (
    "fold_id",
    "subject_id",
    "raw_kss",
    "label",
    "video_path",
    "npz_path",
    "num_frames_15fps",
    "face_detection_rate",
)

REQUIRED_ARRAYS = (
    "mar",
    "head_pose",
    "quality",
    "face_detected",
    "left_eye_crops",
    "right_eye_crops",
)


def _logical_parts(value: str, root: Path) -> Path | None:
    """Return the part of a path from the processed folder name or the first FoldN folder onward."""

    parts = [p for p in Path(value.replace("\\", "/")).parts if p not in (".", "/", "")]
    if root.name in parts:
        rest = parts[len(parts) - parts[::-1].index(root.name):]
        if rest:
            return Path(*rest)
    for i, part in enumerate(parts):
        if re.match(r"(?i)^fold\d", part):
            return Path(*parts[i:])
    return None


def _safe_path(value: str, root: Path) -> str:
    """Return a logical path, never a personal absolute path."""

    if not value:
        return ""
    candidate = Path(value)
    try:
        relative = candidate.resolve().relative_to(root.resolve())
        return relative.as_posix()
    except (OSError, ValueError):
        logical = _logical_parts(value, root)
        return logical.as_posix() if logical else (candidate.name or "<unnamed>")


def _resolve_npz(raw_path: str, metadata_path: Path, root: Path) -> Path | None:
    """Resolve a metadata path using local audit locations only."""

    if not raw_path:
        return None
    candidate = Path(raw_path)
    locations = []
    if candidate.is_absolute():
        locations.append(candidate)
    else:
        locations.extend((metadata_path.parent / candidate, root / candidate, root.parent / candidate))
    logical = _logical_parts(raw_path, root)
    if logical is not None:
        locations.append(root / logical)
    locations.append(root / candidate.name)
    for location in locations:
        if location.is_file() and location.suffix.lower() == ".npz":
            return location
    return None


def _number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _check_npz(path: Path, expected_frames: int | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.name,
        "exists": path.is_file(),
        "keys": [],
        "missing_keys": list(REQUIRED_ARRAYS),
        "shapes": {},
        "frame_count": None,
        "shape_errors": [],
        "load_error": None,
        "valid": False,
    }
    if not path.is_file():
        return result
    try:
        with np.load(path, allow_pickle=False) as arrays:
            result["keys"] = sorted(arrays.files)
            result["missing_keys"] = [key for key in REQUIRED_ARRAYS if key not in arrays]
            for key in REQUIRED_ARRAYS:
                if key in arrays:
                    shape = tuple(int(size) for size in arrays[key].shape)
                    result["shapes"][key] = list(shape)
            if "mar" in arrays:
                result["frame_count"] = int(arrays["mar"].shape[0])
            if "head_pose" in arrays and arrays["head_pose"].ndim != 2:
                result["shape_errors"].append("head_pose must be two-dimensional")
            elif "head_pose" in arrays and arrays["head_pose"].shape[1] != 3:
                result["shape_errors"].append("head_pose must have three columns")
            if "quality" in arrays and arrays["quality"].ndim != 2:
                result["shape_errors"].append("quality must be two-dimensional")
            elif "quality" in arrays and arrays["quality"].shape[1] != 4:
                result["shape_errors"].append("quality must have four columns")
            for key in ("mar", "face_detected", "left_eye_crops", "right_eye_crops"):
                if key in arrays and key not in ("left_eye_crops", "right_eye_crops") and arrays[key].ndim != 1:
                    result["shape_errors"].append(f"{key} must be one-dimensional")
            for key in ("left_eye_crops", "right_eye_crops"):
                if key in arrays and tuple(arrays[key].shape[1:]) != (64, 64):
                    result["shape_errors"].append(f"{key} must have shape (N, 64, 64)")
            if expected_frames is not None and result["frame_count"] not in (None, expected_frames):
                result["shape_errors"].append("metadata frame count does not match mar")
            result["valid"] = not result["missing_keys"] and not result["shape_errors"]
    except (OSError, ValueError, TypeError) as exc:
        result["load_error"] = str(exc)
    return result


def audit(processed_dir: Path, metadata_path: Path, sample_limit: int) -> dict[str, Any]:
    root = processed_dir.resolve()
    metadata_path = metadata_path.resolve()
    result: dict[str, Any] = {
        "schema_version": "rldd_processed_audit_v1",
        "processed_dir": root.name,
        "metadata_file": _safe_path(str(metadata_path), root),
        "metadata_exists": metadata_path.is_file(),
        "required_metadata_columns": list(REQUIRED_COLUMNS),
        "missing_metadata_columns": [],
        "row_count": 0,
        "labels": {},
        "folds": {},
        "subjects": {},
        "missing_npz_rows": 0,
        "sampled_npz_count": 0,
        "valid_sampled_npz_count": 0,
        "npz_checks": [],
        "metadata_value_issues": [],
        "notes": [
            "Paths are reduced to logical relative paths or basenames.",
            "This report does not establish dataset licence permission.",
            "A complete audit requires the full processed artifact and its metadata index.",
        ],
    }
    if not metadata_path.is_file():
        result["metadata_value_issues"].append("metadata index file was not found")
        return result

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        result["missing_metadata_columns"] = [column for column in REQUIRED_COLUMNS if column not in columns]
        rows = list(reader)

    result["row_count"] = len(rows)
    result["labels"] = dict(sorted(Counter(row.get("label", "") or "<blank>" for row in rows).items()))
    result["folds"] = dict(sorted(Counter(row.get("fold_id", "") or "<blank>" for row in rows).items()))
    result["subjects"] = dict(sorted(Counter(row.get("subject_id", "") or "<blank>" for row in rows).items()))

    sampled = 0
    for index, row in enumerate(rows, start=2):
        if result["missing_metadata_columns"]:
            break
        row_id = f"metadata row {index}"
        expected_frames = _integer(row.get("num_frames_15fps", ""))
        if expected_frames is None or expected_frames < 0:
            result["metadata_value_issues"].append(f"{row_id}: invalid num_frames_15fps")
        detection_rate = _number(row.get("face_detection_rate", ""))
        if detection_rate is None or not 0.0 <= detection_rate <= 1.0:
            result["metadata_value_issues"].append(f"{row_id}: face_detection_rate is not in [0, 1]")
        raw_npz = row.get("npz_path", "")
        npz_path = _resolve_npz(raw_npz, metadata_path, root)
        if npz_path is None:
            result["missing_npz_rows"] += 1
            if sampled < sample_limit:
                result["npz_checks"].append({"path": Path(raw_npz).name, "exists": False, "valid": False})
                sampled += 1
            continue
        if sampled < sample_limit:
            check = _check_npz(npz_path, expected_frames)
            check["path"] = _safe_path(str(npz_path), root)
            check["metadata_row"] = index
            result["npz_checks"].append(check)
            sampled += 1

    result["sampled_npz_count"] = len(result["npz_checks"])
    result["valid_sampled_npz_count"] = sum(1 for check in result["npz_checks"] if check.get("valid"))
    if result["missing_metadata_columns"]:
        result["metadata_value_issues"].append("required metadata columns are missing")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit processed UTA-RLDD outputs")
    parser.add_argument("--processed-dir", required=True, type=Path, help="Root of the processed RLDD artifact")
    parser.add_argument("--metadata", type=Path, default=None, help="Metadata CSV, default: rldd_metadata_index.csv under processed-dir")
    parser.add_argument("--sample-limit", type=int, default=20, help="Maximum NPZ files to inspect, default: 20")
    parser.add_argument("--output", required=True, type=Path, help="JSON report path")
    args = parser.parse_args()
    if args.sample_limit < 0:
        parser.error("--sample-limit must be non-negative")
    metadata = args.metadata or args.processed_dir / "rldd_metadata_index.csv"
    report = audit(args.processed_dir, metadata, args.sample_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + os.linesep, encoding="utf-8")
    print(json.dumps({
        "metadata_exists": report["metadata_exists"],
        "row_count": report["row_count"],
        "missing_npz_rows": report["missing_npz_rows"],
        "sampled_npz_count": report["sampled_npz_count"],
        "valid_sampled_npz_count": report["valid_sampled_npz_count"],
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
