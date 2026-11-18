from pathlib import Path
import json
import numpy as np
import pandas as pd

LABELS = ["alert", "fatigue", "distraction"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}

BASE_FEATURES = [
    "mar", "pitch", "yaw", "roll",
    "delta_pitch", "delta_yaw",
    "brightness", "contrast", "glare_fraction", "sharpness",
    "face_detected",
]

OPTIONAL_ARRAYS = ["ear", "eye_closure", "reliability"]

def load_metadata(path):
    df = pd.read_csv(path)
    required = {"subject_id", "label", "npz_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing columns: {sorted(missing)}")
    df["subject_id"] = df["subject_id"].astype(str)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    unknown = sorted(set(df["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels {unknown}; supported labels are {LABELS}")
    return df


def load_baselines(path):
    b = pd.read_csv(path)

    required = {
        "subject_id",
        "mean_pitch",
        "mean_yaw",
        "std_pitch",
    }

    missing = required - set(b.columns)
    if missing:
        raise ValueError(
            f"Baseline CSV missing required R5 columns: {sorted(missing)}"
        )

    b = b[[
    "subject_id",
    "mean_pitch",
    "mean_yaw",
    "std_pitch",
]].copy()

    if b["subject_id"].isna().any():
        raise ValueError("Baseline CSV contains missing subject_id values.")

    b["subject_id"] = b["subject_id"].astype(str).str.strip()

    if (b["subject_id"] == "").any():
        raise ValueError("Baseline CSV contains empty subject_id values.")

    duplicates = b.loc[
        b["subject_id"].duplicated(keep=False),
        "subject_id"
    ].unique()

    if len(duplicates):
        raise ValueError(
            f"Duplicate subject IDs in baseline CSV: {sorted(duplicates)}"
        )

    numeric_columns = ["mean_pitch", "mean_yaw", "std_pitch"]

    for column in numeric_columns:
        b[column] = pd.to_numeric(b[column], errors="coerce")

    values = b[numeric_columns].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            "Baseline CSV contains missing, NaN, or infinite numerical values."
        )

    if (b["std_pitch"] < 0).any():
        raise ValueError("Baseline CSV contains negative std_pitch values.")

    return b.set_index("subject_id")


def wrap_angle_difference(current, baseline):
    """Return the shortest signed angular difference in degrees."""
    return (current - baseline + 180.0) % 360.0 - 180.0


def extract_feature_matrix(npz_path, baseline_row):
    with np.load(npz_path, allow_pickle=False) as z:
        required = {"mar", "head_pose", "quality", "face_detected"}
        missing = required - set(z.files)

        if missing:
            raise ValueError(
                f"{npz_path} missing arrays: {sorted(missing)}"
            )

        mar = np.asarray(z["mar"], dtype=np.float32).reshape(-1)
        pose = np.asarray(z["head_pose"], dtype=np.float32)
        quality = np.asarray(z["quality"], dtype=np.float32)
        detected = np.asarray(
            z["face_detected"], dtype=np.float32
        ).reshape(-1)

        n = len(mar)

        if pose.shape != (n, 3):
            raise ValueError(
                f"{npz_path}: head_pose must have shape (N,3), "
                f"got {pose.shape}"
            )

        if quality.shape != (n, 4):
            raise ValueError(
                f"{npz_path}: quality must have shape (N,4), "
                f"got {quality.shape}"
            )

        if len(detected) != n:
            raise ValueError(
                f"{npz_path}: face_detected length differs from frames"
            )

        mean_pitch = float(baseline_row["mean_pitch"])
        mean_yaw = float(baseline_row["mean_yaw"])

        delta_pitch = wrap_angle_difference(
            pose[:, 0], mean_pitch
        )

        delta_yaw = wrap_angle_difference(
            pose[:, 1], mean_yaw
        )

        base = np.column_stack([
            mar,
            pose[:, 0],
            pose[:, 1],
            pose[:, 2],
            delta_pitch,
            delta_yaw,
            quality[:, 0],
            quality[:, 1],
            quality[:, 2],
            quality[:, 3],
            detected,
        ]).astype(np.float32)

        names = list(BASE_FEATURES)

        for key in OPTIONAL_ARRAYS:
            if key in z.files:
                arr = np.asarray(z[key], dtype=np.float32).reshape(-1)

                if len(arr) != n:
                    raise ValueError(
                        f"{npz_path}: optional {key} length differs "
                        "from frames"
                    )

                base = np.column_stack([base, arr])
                names.append(key)

        # Replace invalid numerical values only after calculating
        # the angular differences.
        base[~np.isfinite(base)] = 0.0

        return base, names, detected.astype(bool)

def make_windows(features, window=75, stride=15):
    if len(features) < window:
        return np.empty((0, window, features.shape[1]), dtype=np.float32)
    starts = range(0, len(features) - window + 1, stride)
    return np.stack([features[s:s+window] for s in starts]).astype(np.float32)

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))

