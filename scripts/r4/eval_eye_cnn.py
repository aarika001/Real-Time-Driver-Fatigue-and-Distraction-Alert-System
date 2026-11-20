"""Evaluate the trained eye CNN as Section 3.3 of the proposal asks (R4, Milestone 5).

Sections (each runs only if its inputs are given):

1. MRL test people  (--splits + --mrl_root)
   Overall / per condition (glasses, reflection, lighting, sensor) / per person,
   bootstrap 95% CIs over people, groups with < 5 people flagged.
2. Stress tests on the same held-out people (graded severity 0-4 of low light,
   glare, blur, occlusion, noise; cues/stress.py) -> where the CNN starts to fail.
3. Cross-dataset (--extra_index CSV of labelled crops from another camera,
   e.g. RGB DMD/NTHU crops) -> the IR-to-RGB generalisation gap.
4. UTA-RLDD (--rldd_dir with R3's processed .npz archives; no eye labels) ->
   P(closed) and PERCLOS per video, and Baseline B2 (CNN + PERCLOS rule):
   does PERCLOS separate fatigue from alert videos, overall and within each person?

Example:
    python scripts/r4/eval_eye_cnn.py --eye_model runs/eye_cnn/eye_cnn.onnx \
        --splits runs/eye_cnn/splits.csv --mrl_root C:/Users/hp/Documents/mrlEyes_2018_01.zip \
        --rldd_dir data/processed_rldd --out_dir results/r4/eye_eval
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cues.eye_crop import _CLAHE  # noqa: E402
from cues.eye_dataset import TAG_COLUMNS, build_mrl_index, load_csv_index, load_raw_crops  # noqa: E402
from cues.eye_metrics import binary_metrics, summarize_predictions, to_markdown, worst_condition  # noqa: E402
from cues.perclos import perclos  # noqa: E402
from cues.stress import FACTORS, LEVELS, degrade_batch  # noqa: E402

RLDD_NAME_RE = re.compile(r"kss_(\d+)_([a-z_]+)\.npz$")


# ---------------------------------------------------------------------------
def predict_raw(predictor, raw: np.ndarray, batch: int = 512) -> np.ndarray:
    """P(closed) for raw (pre-CLAHE) 64x64 crops."""
    out = np.empty(len(raw), dtype=np.float32)
    for s in range(0, len(raw), batch):
        chunk = np.stack([_CLAHE.apply(np.ascontiguousarray(im)) for im in raw[s:s + batch]])
        out[s:s + batch] = predictor.predict_crops(chunk)
    return out


def predict_normalized(predictor, crops: np.ndarray, batch: int = 512) -> np.ndarray:
    """P(closed) for crops that are already CLAHE-normalised (e.g. RLDD archives)."""
    out = np.empty(len(crops), dtype=np.float32)
    for s in range(0, len(crops), batch):
        out[s:s + batch] = predictor.predict_crops(crops[s:s + batch])
    return out


def resolve_test_images(splits: pd.DataFrame, mrl_root: str | Path) -> pd.DataFrame:
    """Re-locate the test people's images on this machine (paths in splits.csv come from Kaggle)."""
    test = splits[splits["split"] == "test"]
    test_mrl = test[test["dataset"] == "mrl"][["subject_id", "image_id"]]
    index = build_mrl_index(mrl_root)
    merged = index.merge(test_mrl, on=["subject_id", "image_id"], how="inner")
    if len(merged) != len(test_mrl):
        raise ValueError(f"Found {len(merged)} of {len(test_mrl)} MRL test images under {mrl_root}")
    return merged.reset_index(drop=True)


def stress_sweep(predictor, raw: np.ndarray, labels: np.ndarray, factors=FACTORS,
                 levels=LEVELS, seed: int = 0) -> pd.DataFrame:
    rows = []
    clean = None
    for factor in factors:
        for level in levels:
            if level == 0 and clean is not None:
                p = clean
            else:
                p = predict_raw(predictor, degrade_batch(raw, factor, level, seed))
                if level == 0:
                    clean = p
            m = binary_metrics(p, labels)
            rows.append({"factor": factor, "level": level, "accuracy": m["accuracy"],
                         "recall_closed": m["recall_closed"], "f1_closed": m["f1_closed"],
                         "auroc": m["auroc"], "ece": m["ece"]})
    return pd.DataFrame(rows)


def first_failure(sweep: pd.DataFrame, metric: str = "recall_closed", drop: float = 0.10) -> dict:
    """Lowest severity at which ``metric`` falls more than ``drop`` below clean, per factor."""
    out = {}
    for factor, g in sweep.groupby("factor"):
        g = g.sort_values("level")
        clean = g.iloc[0][metric]
        bad = g[g[metric] < clean - drop]
        out[factor] = int(bad.iloc[0]["level"]) if len(bad) else None
    return out


def rldd_videos(rldd_dir: str | Path) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(rldd_dir).rglob("*.npz")):
        m = RLDD_NAME_RE.search(p.name)
        if not m:
            continue
        fold = next((part.split("_")[0] for part in p.parts if part.startswith("Fold")), "unknown")
        subject = p.parent.name.replace("subject_", "")
        rows.append({"path": str(p), "fold": fold, "subject_id": f"rldd/{subject}",
                     "raw_kss": m.group(1), "label": m.group(2)})
    df = pd.DataFrame(rows)
    if len(df):   # the same video can arrive twice when the Drive download is split into overlapping parts
        df = df.drop_duplicates(subset=["fold", "subject_id", "raw_kss", "label"]).reset_index(drop=True)
    return df


def rldd_analysis(predictor, videos: pd.DataFrame, stride: int = 3, fps: float = 15.0,
                  window_s: float = 60.0, max_frames: int = 0) -> pd.DataFrame:
    rows = []
    for i, v in enumerate(videos.itertuples(index=False), 1):
        t0 = time.time()
        with np.load(v.path) as z:
            det = z["face_detected"][::stride].astype(bool)
            left = z["left_eye_crops"][::stride]
            right = z["right_eye_crops"][::stride]
        if max_frames:
            det, left, right = det[:max_frames], left[:max_frames], right[:max_frames]
        p = np.full(len(det), np.nan, dtype=np.float32)
        if det.any():
            both = np.concatenate([left[det], right[det]])
            pb = predict_normalized(predictor, both)
            k = int(det.sum())
            p[det] = (pb[:k] + pb[k:]) / 2.0
        pc = perclos(p, fps / stride, window_s)
        rows.append({"subject_id": v.subject_id, "fold": v.fold, "label": v.label, "frames_used": len(p),
                     "face_rate": float(det.mean()) if len(det) else 0.0,
                     "mean_p_closed": float(np.nanmean(p)) if det.any() else np.nan,
                     "closed_frac": float(np.nanmean(p >= 0.5)) if det.any() else np.nan,
                     "mean_perclos": float(np.nanmean(pc)) if np.isfinite(pc).any() else np.nan,
                     "max_perclos": float(np.nanmax(pc)) if np.isfinite(pc).any() else np.nan})
        print(f"  RLDD [{i}/{len(videos)}] {v.subject_id} {v.label}: PERCLOS {rows[-1]['mean_perclos']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def b2_video_level(per_video: pd.DataFrame, feature: str = "mean_perclos") -> dict:
    """Baseline B2 at video level: alert vs fatigue from PERCLOS.

    * AUROC over all videos (does more PERCLOS mean more fatigue?)
    * within-person: share of people whose fatigue video has higher PERCLOS
      than their alert video (removes person-to-person differences)
    * fixed threshold chosen on half the people, accuracy on the other half
    """
    d = per_video[per_video["label"].isin(["alert", "fatigue"])].dropna(subset=[feature])
    if d.empty or d["label"].nunique() < 2:
        return {"n_videos": int(len(d))}
    y = (d["label"] == "fatigue").astype(int).to_numpy()
    x = d[feature].to_numpy()
    res = {"n_videos": int(len(d)), "n_people": int(d["subject_id"].nunique()),
           "auroc": binary_metrics(x, y)["auroc"]}
    piv = d.pivot_table(index="subject_id", columns="label", values=feature, aggfunc="mean").dropna()
    if len(piv):
        res["within_person_fatigue_higher"] = float((piv["fatigue"] > piv["alert"]).mean())
        res["n_people_paired"] = int(len(piv))
    subjects = np.array(sorted(d["subject_id"].unique()))
    tune, test = set(subjects[0::2]), set(subjects[1::2])
    dt, de = d[d["subject_id"].isin(tune)], d[d["subject_id"].isin(test)]
    if len(dt) and len(de) and dt["label"].nunique() == 2:
        cands = np.unique(dt[feature])
        yt = (dt["label"] == "fatigue").to_numpy()

        def bal_acc(th, frame, ylab):
            pred = frame[feature].to_numpy() > th
            return 0.5 * ((pred[ylab]).mean() + (~pred[~ylab]).mean())

        best = max(cands, key=lambda th: bal_acc(th, dt, yt))
        ye = (de["label"] == "fatigue").to_numpy()
        if ye.any() and (~ye).any():
            res["b2_threshold"] = float(best)
            res["b2_balanced_accuracy_heldout_people"] = float(bal_acc(best, de, ye))
    return res


def maybe_plot_sweep(sweep: pd.DataFrame, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for factor, g in sweep.groupby("factor"):
        ax.plot(g["level"], g["recall_closed"], marker="o", label=factor)
    ax.set_xlabel("severity level (0 = clean)")
    ax.set_ylabel("recall of closed eyes")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(list(LEVELS))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None, predictor=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eye_model", help=".pt or .onnx")
    ap.add_argument("--splits", help="splits.csv from the training run")
    ap.add_argument("--mrl_root", help="MRL folder or zip on this machine")
    ap.add_argument("--extra_index", nargs="*", default=[], help="labelled crop CSVs for cross-dataset tests")
    ap.add_argument("--rldd_dir", help="folder with R3's processed RLDD .npz archives")
    ap.add_argument("--rldd_stride", type=int, default=3, help="use every Nth frame (3 -> 5 FPS)")
    ap.add_argument("--rldd_max_videos", type=int, default=0)
    ap.add_argument("--stress_images", type=int, default=3000, help="test images used for stress sweeps")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out_dir", default="results/r4/eye_eval")
    args = ap.parse_args(argv)

    if predictor is None:
        if not args.eye_model:
            ap.error("--eye_model is required")
        from cues.predictors import load_eye_predictor
        predictor = load_eye_predictor(args.eye_model, num_threads=args.threads)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = ["# Eye CNN evaluation (R4)", "", f"Model: `{args.eye_model}`", ""]

    if args.splits and args.mrl_root:
        print("[1/4] MRL test people")
        test = resolve_test_images(pd.read_csv(args.splits, dtype={"subject_id": str}), args.mrl_root)
        raw = load_raw_crops(test, verbose=False)
        test["p_closed"] = predict_raw(predictor, raw)
        test.drop(columns=["zip_path"]).to_csv(out / "mrl_test_predictions.csv", index=False)
        summ = summarize_predictions(test, ["dataset", *TAG_COLUMNS])
        summ.to_csv(out / "mrl_test_summary.csv", index=False)
        worst = worst_condition(summ)
        report += ["## 1. MRL Eye, unseen test people", "", to_markdown(summ), ""]
        if worst is not None:
            report += [f"Worst condition with >= 5 people: **{worst['group']}** "
                       f"(recall of closed eyes {worst['recall_closed']:.3f}).", ""]

        print("[2/4] Stress sweeps")
        rng = np.random.default_rng(0)
        pick = np.sort(rng.choice(len(test), size=min(args.stress_images, len(test)), replace=False))
        sweep = stress_sweep(predictor, raw[pick], test["label_closed"].to_numpy()[pick])
        sweep.to_csv(out / "stress_sweep.csv", index=False)
        fails = first_failure(sweep)
        plotted = maybe_plot_sweep(sweep, out / "stress_sweep.png")
        piv = sweep.pivot(index="factor", columns="level", values="recall_closed")
        report += ["## 2. Stress tests (recall of closed eyes by severity, held-out people)", "",
                   "| factor | " + " | ".join(f"L{c}" for c in piv.columns) + " | first fails (>10 pt drop) |",
                   "|---|" + "---|" * (len(piv.columns) + 1)]
        for factor, r in piv.iterrows():
            report.append(f"| {factor} | " + " | ".join(f"{v:.3f}" for v in r) + f" | {fails.get(factor) or '-'} |")
        report += ["", "![stress](stress_sweep.png)" if plotted else "", ""]

    if args.extra_index:
        print("[3/4] Cross-dataset")
        frames = [load_csv_index(c) for c in args.extra_index]
        ext = pd.concat(frames, ignore_index=True)
        ext["p_closed"] = predict_raw(predictor, load_raw_crops(ext, verbose=False))
        summ = summarize_predictions(ext, ["dataset", *TAG_COLUMNS])
        summ.to_csv(out / "cross_dataset_summary.csv", index=False)
        report += ["## 3. Cross-dataset (trained on MRL infrared, tested on other cameras)", "",
                   to_markdown(summ), ""]

    if args.rldd_dir:
        print("[4/4] UTA-RLDD PERCLOS and Baseline B2")
        vids = rldd_videos(args.rldd_dir)
        if args.rldd_max_videos:
            vids = vids.head(args.rldd_max_videos)
        if vids.empty:
            report += ["## 4. UTA-RLDD", "", f"No kss_*.npz archives found under {args.rldd_dir}", ""]
        else:
            per_video = rldd_analysis(predictor, vids, stride=args.rldd_stride)
            per_video.to_csv(out / "rldd_per_video.csv", index=False)
            by_label = per_video.groupby("label")[["mean_p_closed", "closed_frac", "mean_perclos"]].mean()
            b2 = b2_video_level(per_video)
            report += ["## 4. UTA-RLDD (no eye labels): PERCLOS and Baseline B2", "",
                       "Mean per label:", "", "```", by_label.round(4).to_string(), "```", "", "Baseline B2 (CNN + PERCLOS), video level:", "",
                       "```", "\n".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                        for k, v in b2.items()), "```", ""]

    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
