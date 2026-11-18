from pathlib import Path
import pandas as pd
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "Mirror": ROOT / "data/processed/mirror_sqpnp_full/calibration/driver_baselines_r5.csv",
    "YawDD": ROOT / "data/processed/yawdd_r6/calibration/driver_baselines_r5.csv",
}

REQUIRED = ["subject_id", "mean_pitch", "mean_yaw", "std_pitch"]
failed = False

for dataset, path in FILES.items():
    print(f"\n{'=' * 60}")
    print(f"{dataset} CALIBRATION VALIDATION")
    print(f"{'=' * 60}")

    if not path.exists():
        print(f"FAIL: Missing file: {path}")
        failed = True
        continue

    df = pd.read_csv(path)
    missing_columns = [c for c in REQUIRED if c not in df.columns]

    if missing_columns:
        print("FAIL: Missing columns:", missing_columns)
        failed = True
        continue

    duplicate_ids = df["subject_id"].duplicated().sum()
    missing_ids = df["subject_id"].isna().sum()

    numeric = df[["mean_pitch", "mean_yaw", "std_pitch"]].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid_values = (~np.isfinite(numeric.to_numpy())).sum()
    negative_std = (numeric["std_pitch"] < 0).sum()

    print("Baseline rows:", len(df))
    print("Unique subjects:", df["subject_id"].nunique())
    print("Duplicate subject IDs:", duplicate_ids)
    print("Missing subject IDs:", missing_ids)
    print("Non-finite numeric values:", invalid_values)
    print("Negative pitch standard deviations:", negative_std)

    if dataset == "YawDD" and "baseline_duration_seconds" in df.columns:
        durations = pd.to_numeric(
            df["baseline_duration_seconds"], errors="coerce"
        )
        complete = durations >= 29.5
        print("Complete (~30 sec):", int(complete.sum()))
        print("Partial (<29.5 sec):", int((~complete).sum()))
        print("\nPartial subjects:")
        print(
            df.loc[~complete, ["subject_id", "baseline_duration_seconds"]]
            .to_string(index=False)
        )

    elif dataset == "YawDD" and "status" in df.columns:
        print("Calibration status counts:")
        print(df["status"].value_counts(dropna=False).to_string())

    problems = (
        duplicate_ids > 0
        or missing_ids > 0
        or invalid_values > 0
        or negative_std > 0
    )

    if problems:
        print("RESULT: FAIL")
        failed = True
    else:
        print("RESULT: PASS")

print(f"\n{'=' * 60}")
print("OVERALL:", "FAIL" if failed else "PASS")
print(f"{'=' * 60}")

sys.exit(1 if failed else 0)
