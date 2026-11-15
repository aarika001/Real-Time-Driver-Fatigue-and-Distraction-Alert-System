"""Evaluate the mouth cue's yawn rule on YawDD (R4, Milestone 2).

Inputs: the output of ``extract_yawdd_mar.py`` and R2's yawn annotations.

What it reports
---------------
* Event recall: fraction of annotated yawns detected (overlap within
  +/- tolerance).
* False alarms per hour of normal behaviour (Normal + Talking clips), split
  into talking vs normal. The proposal's target is <= 3 per hour.
* Median detection delay (how long after the yawn starts the rule fires).
* All of the above per eyewear condition and per camera.
* Three calibration modes, to show why per-driver calibration matters:
    subject  - neutral MAR from the same person's Normal clip (our method)
    first_s  - neutral MAR from the first --calib_s seconds of the clip itself
    fixed    - one population-wide neutral MAR for everyone (no calibration)

Threshold choice without cheating
---------------------------------
With --sweep, (open_ratio, min_yawn_s) are tuned on half of the people and the
chosen setting is reported on the other half, so the headline number is not
fitted to the people it is measured on.

Example:
    python scripts/r4/eval_yawdd.py --features_dir data/yawdd_mar \
        --annotations data/yawdd_yawns.csv --out_dir results/r4/yawdd --sweep
"""
from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cues.config import load_config  # noqa: E402
from cues.mouth_cue import MouthConfig, yawn_events, yawn_features  # noqa: E402
from cues.yawdd import EventScore, YawddVideoInfo, ground_truth_for, load_annotations, score_events  # noqa: E402

OPEN_RATIOS = (1.3, 1.45, 1.6, 1.8, 2.0)
MIN_YAWN_S = (1.0, 1.5, 2.0, 2.5)
MAX_FA_PER_HOUR = 3.0   # proposal success criterion


def load_videos(features_dir: Path, annotations: dict) -> list[dict]:
    index = pd.read_csv(features_dir / "index.csv", dtype=str)
    videos, skipped = [], 0
    for r in index.itertuples(index=False):
        info = YawddVideoInfo(r.video_id, r.subject_id, r.action, r.eyewear, r.camera)
        gt = ground_truth_for(info, annotations)
        with np.load(features_dir / r.npz) as z:
            mar = z["mar"].astype(np.float64)
            mar[z["face_detected"] == 0] = np.nan
            fps = float(z["fps"])
        if gt is None:
            skipped += 1
        videos.append({"info": info, "mar": mar, "fps": fps, "gt": gt})
    if skipped:
        print(f"Note: {skipped} Yawning/mixed videos have no annotations and are not scored "
              "(still used for the subject baseline).")
    return videos


def neutral_baselines(videos: list[dict], mode: str, calib_s: float) -> dict[str, float]:
    """Neutral MAR per video_id for the chosen calibration mode."""
    def first_s(v):
        n = max(1, int(calib_s * v["fps"]))
        return float(np.nanmedian(v["mar"][:n]))

    out = {}
    if mode == "fixed":
        normals = [v["mar"] for v in videos if v["info"].action == "normal"]
        pool = np.concatenate(normals) if normals else np.concatenate([v["mar"] for v in videos])
        pop = float(np.nanmedian(pool))
        return {v["info"].video_id: pop for v in videos}
    subj_normal = {v["info"].subject_id: float(np.nanmedian(v["mar"]))
                   for v in videos if v["info"].action == "normal"}
    for v in videos:
        if mode == "subject" and v["info"].subject_id in subj_normal:
            out[v["info"].video_id] = subj_normal[v["info"].subject_id]
        else:
            out[v["info"].video_id] = first_s(v)
    return out


def run(videos: list[dict], baselines: dict[str, float], cfg: MouthConfig,
        tolerance_s: float) -> pd.DataFrame:
    rows = []
    for v in videos:
        if v["gt"] is None:
            continue
        info = v["info"]
        c = replace(cfg, fps=v["fps"])
        feats = yawn_features(v["mar"], baselines[info.video_id], c)
        ev = yawn_events(feats["yawn_flag"], feats["open_run_s"], c.fps)
        s = score_events(ev, v["gt"], tolerance_s)
        rows.append({"video_id": info.video_id, "subject_id": info.subject_id, "action": info.action,
                     "eyewear": info.eyewear, "camera": info.camera,
                     "hours": len(v["mar"]) / c.fps / 3600.0, "n_gt": s.n_gt, "n_hit": s.n_hit,
                     "n_false": s.n_false, "delays_s": list(s.delays_s)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"recall": np.nan, "fa_per_h_normal_behaviour": np.nan, "fa_per_h_talking": np.nan,
                "fa_per_h_normal": np.nan, "median_delay_s": np.nan, "n_gt": 0, "n_videos": 0,
                "n_people": 0}
    nb = df[df["action"].isin(["normal", "talking"])]

    def fa_rate(d):
        h = d["hours"].sum()
        return float(d["n_false"].sum() / h) if h > 0 else np.nan

    delays = [x for xs in df["delays_s"] for x in xs]
    n_gt = int(df["n_gt"].sum())
    return {
        "recall": float(df["n_hit"].sum() / n_gt) if n_gt else np.nan,
        "fa_per_h_normal_behaviour": fa_rate(nb),
        "fa_per_h_talking": fa_rate(df[df["action"] == "talking"]),
        "fa_per_h_normal": fa_rate(df[df["action"] == "normal"]),
        "median_delay_s": float(np.median(delays)) if delays else np.nan,
        "n_gt": n_gt,
        "n_videos": int(len(df)),
        "n_people": int(df["subject_id"].nunique()),
    }


def split_subjects(videos: list[dict]) -> tuple[set, set]:
    """Deterministic 50/50 split of people into tune / test halves."""
    subjects = sorted({v["info"].subject_id for v in videos})
    return set(subjects[0::2]), set(subjects[1::2])


def sweep(videos: list[dict], baselines: dict, cfg: MouthConfig, tol: float) -> pd.DataFrame:
    rows = []
    for open_r, min_s in itertools.product(OPEN_RATIOS, MIN_YAWN_S):
        c = replace(cfg, open_ratio=open_r, min_yawn_s=min_s,
                    yawn_peak_ratio=max(cfg.yawn_peak_ratio, open_r),
                    talk_ratio=min(cfg.talk_ratio, open_r - 0.05))
        rows.append({"open_ratio": open_r, "min_yawn_s": min_s, **summarize(run(videos, baselines, c, tol))})
    return pd.DataFrame(rows)


def pick_operating_point(sw: pd.DataFrame) -> pd.Series:
    """Highest recall with <= 3 false alarms/hour of normal behaviour;
    if none qualifies, the fewest false alarms."""
    ok = sw[sw["fa_per_h_normal_behaviour"] <= MAX_FA_PER_HOUR]
    if len(ok):
        return ok.sort_values(["recall", "fa_per_h_normal_behaviour"], ascending=[False, True]).iloc[0]
    return sw.sort_values(["fa_per_h_normal_behaviour", "recall"], ascending=[True, False]).iloc[0]


def fmt_table(df: pd.DataFrame) -> str:
    cols = ["recall", "fa_per_h_normal_behaviour", "fa_per_h_talking", "median_delay_s", "n_gt", "n_people"]
    lines = ["| group | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for name, r in df.iterrows():
        vals = [str(int(r[c])) if c.startswith("n_") and np.isfinite(r[c])
                else f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--annotations", required=True, help="CSV with video_id,start_s,end_s")
    ap.add_argument("--out_dir", default="results/r4/yawdd")
    ap.add_argument("--config", default=None, help="YAML config (default configs/r4.yaml)")
    ap.add_argument("--calib_s", type=float, default=10.0, help="seconds used by the first_s mode")
    ap.add_argument("--tolerance_s", type=float, default=1.0)
    ap.add_argument("--sweep", action="store_true", help="tune thresholds on half the people")
    args = ap.parse_args(argv)

    cfg = MouthConfig.from_dict(load_config(args.config).get("mouth"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    videos = load_videos(Path(args.features_dir), load_annotations(args.annotations))
    report = ["# YawDD evaluation of the R4 mouth cue", ""]

    base_subject = neutral_baselines(videos, "subject", args.calib_s)
    if args.sweep:
        tune_s, test_s = split_subjects(videos)
        tune = [v for v in videos if v["info"].subject_id in tune_s]
        test = [v for v in videos if v["info"].subject_id in test_s]
        sw = sweep(tune, base_subject, cfg, args.tolerance_s)
        sw.to_csv(out / "sweep_tune_half.csv", index=False)
        best = pick_operating_point(sw)
        cfg = replace(cfg, open_ratio=float(best["open_ratio"]), min_yawn_s=float(best["min_yawn_s"]),
                      yawn_peak_ratio=max(cfg.yawn_peak_ratio, float(best["open_ratio"])),
                      talk_ratio=min(cfg.talk_ratio, float(best["open_ratio"]) - 0.05))
        held = summarize(run(test, base_subject, cfg, args.tolerance_s))
        report += ["## Threshold tuning (tuned on half the people, reported on the other half)", "",
                   f"Chosen: open_ratio={cfg.open_ratio}, min_yawn_s={cfg.min_yawn_s} "
                   f"(tune-half recall {best['recall']:.3f}, "
                   f"FA/h {best['fa_per_h_normal_behaviour']:.2f})", "",
                   fmt_table(pd.DataFrame([held], index=["held-out people"])), ""]

    modes = {}
    for mode in ("subject", "first_s", "fixed"):
        b = base_subject if mode == "subject" else neutral_baselines(videos, mode, args.calib_s)
        df = run(videos, b, cfg, args.tolerance_s)
        df.assign(mode=mode).to_csv(out / f"per_video_{mode}.csv", index=False)
        modes[mode] = (summarize(df), df)

    report += ["## Calibration modes (all people)", "",
               fmt_table(pd.DataFrame({m: s for m, (s, _) in modes.items()}).T), ""]
    df_main = modes["subject"][1]
    for col in ("eyewear", "camera"):
        groups = {f"{col}={k}": summarize(g) for k, g in df_main.groupby(col)}
        report += [f"## By {col} (subject calibration)", "",
                   fmt_table(pd.DataFrame(groups).T), ""]
    small = [k for k, g in df_main.groupby("eyewear") if g["subject_id"].nunique() < 5]
    if small:
        report += [f"Flag: fewer than 5 people in eyewear group(s) {small}; do not over-interpret.", ""]
    report += ["Config used:", "", "```", str(cfg.to_dict()), "```"]

    (out / "summary.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved results to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
