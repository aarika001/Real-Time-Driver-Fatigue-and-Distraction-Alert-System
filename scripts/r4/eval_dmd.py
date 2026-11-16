"""Evaluate the R4 cues on DMD (R2's adopted protocol, R2_HANDOFF.md section 4).

DMD (Ortega et al., 2020; Kaggle copy ``dmd-video-v1``) gives RGB face videos with
native per-frame labels, so this is the real-world RGB test of the eye CNN that
was trained on infrared MRL crops (hypothesis H1), and the real yawn test of the
mouth cue. No DMD data was used for training or tuning, so every DMD person is
unseen; results are reported for all people and for R2's held-out test split.

Parts (each runs only if its inputs are given):

A. Drowsiness videos (--videos_dir + --events)
   Every face video is sampled at 15 FPS and run through the full live pipeline
   (MediaPipe -> eye crops -> CNN, MAR). Per-frame cues are cached, so re-runs
   are instant.
   * Eye closure, frame level: frames inside ``eyes_state/close`` = closed,
     inside ``eyes_state/open`` = open; closing/opening/undefined are excluded.
   * Yawns, event level: R2's merged yawn intervals vs the mouth rule
     (per-session calibration on the first --calib_s seconds); recall, false
     alarms per hour, detection delay; with-hand yawns reported separately
     because the hand hides the mouth.
B. Stress set (--stress_root + --stress_manifest)
   R2's 456 stressed frames + 24 originals from 4 held-out people. Paired change
   per stress type and severity: face detection rate, eye-state accuracy,
   inner-lip MAR kept. Fewer than 5 people, so flagged as indicative.

Data stays outside the repo (DMD is CC BY-NC-ND; never commit frames or events).

Example:
    python scripts/r4/eval_dmd.py --eye_model runs/eye_cnn/eye_cnn.onnx \
        --videos_dir D:/NeuroNova_data/dmd/dmd_video_v1 --events D:/NeuroNova_data/dmd/r2_manifests/dmd_events.csv \
        --stress_root D:/NeuroNova_data --stress_manifest R2_Aadi_Contribution/metadata/stress_test_manifest_dmd.csv \
        --out_dir results/r4/dmd_eval
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cues.config import load_config  # noqa: E402
from cues.eye_metrics import summarize_predictions, to_markdown  # noqa: E402
from cues.mouth_cue import MouthConfig, robust_baseline, yawn_events, yawn_features  # noqa: E402
from cues.yawdd import EventScore, score_events  # noqa: E402

DMD_FPS = 744 / 25          # 29.76, R2-verified face-camera rate
EYE_CLOSED, EYE_OPEN = "eyes_state/close", "eyes_state/open"
MIN_PEOPLE = 5


# ---------------------------------------------------------------------------
# Part A: per-session cues
# ---------------------------------------------------------------------------
def find_face_videos(videos_dir: str | Path) -> dict[str, Path]:
    """{session_id: path} for every ``<session_id>_face.mp4`` under videos_dir."""
    out = {}
    for p in Path(videos_dir).rglob("*_face.mp4"):
        out[p.name[: -len("_face.mp4")]] = p
    return out


def extract_session_cues(video: Path, pipeline, target_fps: float = 15.0,
                         max_seconds: float = 0.0) -> pd.DataFrame:
    """Run the live pipeline on one video at ``target_fps``; one row per processed frame."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and np.isfinite(fps) and fps > 0 else DMD_FPS
    rows, idx, next_t = [], 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        if max_seconds and t > max_seconds:
            break
        if t + 1e-9 >= next_t:
            next_t += 1.0 / target_fps
            d = pipeline.process_frame(frame, t).output.to_flat_dict()
            rows.append({"frame": idx, "t_s": t, "video_fps": fps, "face_detected": d["face_detected"],
                         "p_closed": d["p_closed"], "p_closed_left": d["p_closed_left"],
                         "p_closed_right": d["p_closed_right"], "ear": d["ear"], "mar": d["mar"],
                         "mar_inner": d["mar_inner"], "eye_glare_frac": d["eye_glare_frac"]})
        idx += 1
    cap.release()
    return pd.DataFrame(rows)


def frame_eye_labels(frames: np.ndarray, events: pd.DataFrame) -> np.ndarray:
    """1 = closed, 0 = open, NaN = not judged (closing/opening/undefined/none) for each frame index."""
    lab = np.full(len(frames), np.nan)
    for label, value in ((EYE_OPEN, 0.0), (EYE_CLOSED, 1.0)):     # closed wins if both overlap
        for r in events[events["native_label"] == label].itertuples(index=False):
            lab[(frames >= r.start_frame) & (frames <= r.end_frame)] = value
    return lab


def session_yawns(cues: pd.DataFrame, events: pd.DataFrame, cfg: MouthConfig, fps_video: float,
                  calib_s: float, tolerance_s: float) -> dict:
    """Score the mouth rule on one session against its yawn intervals."""
    mar = cues["mar"].to_numpy(dtype=float)
    t = cues["t_s"].to_numpy()
    fps_cues = 1.0 / np.median(np.diff(t)) if len(t) > 1 else cfg.fps
    calib = mar[t < calib_s]
    try:
        neutral, calib_source = robust_baseline(calib, cfg.min_calib_frames), f"first {calib_s:.0f} s"
    except ValueError:
        neutral, calib_source = robust_baseline(mar, cfg.min_calib_frames), "whole session (start not usable)"
    c = MouthConfig(**{**cfg.to_dict(), "fps": fps_cues})
    feats = yawn_features(mar, neutral, c)
    detected = [(s + t[0], e + t[0], d + t[0]) for s, e, d in yawn_events(feats["yawn_flag"], feats["open_run_s"], c.fps)]
    yawns = events[events["native_label"].str.startswith("yawning/")]
    gt = [((r.start_frame) / fps_video, (r.end_frame + 1) / fps_video) for r in yawns.itertuples(index=False)]
    score = score_events(detected, gt, tolerance_s)
    by_type = {}
    for kind in ("without hand", "with hand", "merged"):
        sub = yawns[yawns["native_label"].str.contains(kind, case=False, regex=False)]
        if kind == "with hand":
            sub = sub[~sub["native_label"].str.contains("without", case=False)]
        g = [((r.start_frame) / fps_video, (r.end_frame + 1) / fps_video) for r in sub.itertuples(index=False)]
        if g:
            s = score_events(detected, g, tolerance_s)
            by_type[kind] = (s.n_gt, s.n_hit)
    return {"score": score, "hours": (t[-1] - t[0] + 1 / fps_cues) / 3600 if len(t) else 0.0,
            "neutral_mar": neutral, "calibration": calib_source, "by_type": by_type, "n_detected": len(detected)}


def yawn_summary(rows: list[dict]) -> dict:
    total = EventScore()
    for r in rows:
        total = total + r["score"]
    hours = sum(r["hours"] for r in rows)
    type_gt, type_hit = {}, {}
    for r in rows:
        for k, (g, h) in r["by_type"].items():
            type_gt[k] = type_gt.get(k, 0) + g
            type_hit[k] = type_hit.get(k, 0) + h
    return {"n_yawns": total.n_gt, "n_hit": total.n_hit,
            "recall": total.n_hit / total.n_gt if total.n_gt else float("nan"),
            "false_alarms": total.n_false, "hours": hours,
            "false_alarms_per_hour": total.n_false / hours if hours else float("nan"),
            "median_delay_s": float(np.median(total.delays_s)) if total.delays_s else float("nan"),
            "n_people": len({r["subject_id"] for r in rows}),
            "recall_by_type": {k: f"{type_hit[k]}/{type_gt[k]}" for k in type_gt}}


# ---------------------------------------------------------------------------
# Part B: stress set
# ---------------------------------------------------------------------------
def stress_eval(manifest: pd.DataFrame, stress_root: Path, pipeline) -> pd.DataFrame:
    """Per stressed image and its original: face found, P(closed), inner MAR.
    ``pipeline`` must use an IMAGE-mode landmarker (every image is independent)."""
    cache: dict[str, dict] = {}

    def run(path: Path) -> dict:
        key = str(path)
        if key not in cache:
            img = cv2.imread(key)
            if img is None:
                raise FileNotFoundError(f"cannot read {path}")
            d = pipeline.process_frame(img, 0.0).output.to_flat_dict()
            cache[key] = {"face": d["face_detected"], "p_closed": d["p_closed"], "mar_inner": d["mar_inner"]}
        return cache[key]

    rows = []
    for r in manifest.itertuples(index=False):
        p = stress_root / r.output_path
        o = run(p.parent / "original.png")
        s = run(p)
        rows.append({"person": str(r.source_person_id), "state": r.source_state, "stress_type": r.stress_type,
                     "severity": r.severity, "orig_face": o["face"], "face": s["face"],
                     "orig_p_closed": o["p_closed"], "p_closed": s["p_closed"],
                     "orig_mar_inner": o["mar_inner"], "mar_inner": s["mar_inner"]})
    return pd.DataFrame(rows)


def _median_ratio(a: pd.Series, b: pd.Series) -> float:
    """Median of a/b over pairs where both are finite (NaN = face not found -> nothing kept)."""
    r = (a / b).to_numpy(dtype=float)
    r = np.where(np.isfinite(r), r, 0.0) if len(r) else r
    return float(np.median(r)) if len(r) else float("nan")


def stress_summary(df: pd.DataFrame) -> pd.DataFrame:
    eye_true = df["state"].map({"eyes_closed": 1.0, "eyes_open_no_yawn": 0.0})
    df = df.assign(eye_true=eye_true)
    out = []
    for (st, sev), g in df.groupby(["stress_type", "severity"], sort=True):
        e = g.dropna(subset=["eye_true"])
        e_ok = e[np.isfinite(e["p_closed"])]
        o_ok = e[np.isfinite(e["orig_p_closed"])]
        y = g[(g["state"] == "yawn") & np.isfinite(g["orig_mar_inner"]) & (g["orig_mar_inner"] > 0)]
        out.append({
            "stress_type": st, "severity": sev, "n": len(g), "n_people": g["person"].nunique(),
            "face_rate_original": g["orig_face"].mean(), "face_rate_stressed": g["face"].mean(),
            "eye_acc_original": float(((o_ok["orig_p_closed"] >= 0.5) == (o_ok["eye_true"] == 1)).mean()) if len(o_ok) else np.nan,
            "eye_acc_stressed": float(((e_ok["p_closed"] >= 0.5) == (e_ok["eye_true"] == 1)).mean()) if len(e_ok) else np.nan,
            "eye_scored_frac": len(e_ok) / len(e) if len(e) else np.nan,
            "yawn_mar_kept": _median_ratio(y["mar_inner"], y["orig_mar_inner"]),
            "flag": "indicative (< 5 people)" if g["person"].nunique() < MIN_PEOPLE else "",
        })
    return pd.DataFrame(out)


def df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None, pipeline_factory=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eye_model", help=".onnx or .pt")
    ap.add_argument("--videos_dir", help="folder with DMD *_face.mp4 drowsiness videos")
    ap.add_argument("--events", help="R2 events CSV (session_id, subject_id, kaggle_split, native_label, start_frame, end_frame)")
    ap.add_argument("--stress_root", help="folder that contains generated_stress/")
    ap.add_argument("--stress_manifest", help="R2 stress_test_manifest_dmd.csv")
    ap.add_argument("--out_dir", default="results/r4/dmd_eval")
    ap.add_argument("--config", default=None)
    ap.add_argument("--target_fps", type=float, default=15.0)
    ap.add_argument("--calib_s", type=float, default=30.0, help="per-session mouth calibration window")
    ap.add_argument("--tolerance_s", type=float, default=1.0)
    ap.add_argument("--max_seconds", type=float, default=0.0, help="debug: only the first N seconds per video")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args(argv)

    if pipeline_factory is None:
        if not args.eye_model:
            ap.error("--eye_model is required")
        from cues.face_landmarker import FaceLandmarker
        from cues.inference_r4 import R4Pipeline
        from cues.predictors import load_eye_predictor
        predictor = load_eye_predictor(args.eye_model, num_threads=args.threads)

        def pipeline_factory(mode: str = "video"):
            return R4Pipeline(eye_predictor=predictor, landmarker=FaceLandmarker(mode=mode))

    out = Path(args.out_dir)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    cfg = MouthConfig.from_dict(load_config(args.config).get("mouth"))
    report = ["# R4 cues on DMD (RGB, unseen people)", "",
              "Eye CNN trained only on MRL infrared crops; mouth thresholds from configs/r4.yaml. "
              "No DMD data was used for training or tuning.", ""]

    if args.videos_dir and args.events:
        events = pd.read_csv(args.events, dtype={"subject_id": str})
        videos = find_face_videos(args.videos_dir)
        sessions = sorted(set(events["session_id"]) & set(videos))
        missing = sorted(set(events["session_id"]) - set(videos))
        print(f"[A] {len(sessions)} sessions with video" + (f"; {len(missing)} without video" if missing else ""))
        eye_rows, yawn_rows = [], []
        for i, sid in enumerate(sessions, 1):
            ev = events[events["session_id"] == sid]
            cache = out / "cache" / f"{sid}.csv"
            if cache.exists():
                cues = pd.read_csv(cache)
            else:
                t0 = time.time()
                cues = extract_session_cues(videos[sid], pipeline_factory("video"), args.target_fps, args.max_seconds)
                cues.to_csv(cache, index=False)
                print(f"  [{i}/{len(sessions)}] {sid}: {len(cues)} frames, face {cues['face_detected'].mean():.0%} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if cues.empty:
                print(f"  {sid}: no frames read, skipped")
                continue
            fps_video = float(cues["video_fps"].iloc[0])
            subj, split = str(ev["subject_id"].iloc[0]), str(ev["kaggle_split"].iloc[0])
            lab = frame_eye_labels(cues["frame"].to_numpy(), ev)
            keep = np.isfinite(lab)
            eye_rows.append(pd.DataFrame({"subject_id": f"dmd/{subj}", "split": split, "session_id": sid,
                                          "label_closed": lab[keep].astype(int),
                                          "p_closed": cues["p_closed"].to_numpy()[keep]}))
            y = session_yawns(cues, ev, cfg, fps_video, args.calib_s, args.tolerance_s)
            yawn_rows.append({"subject_id": subj, "split": split, "session_id": sid, **y})

        eye = pd.concat(eye_rows, ignore_index=True) if eye_rows else pd.DataFrame()
        if len(eye):
            judged = len(eye)
            eye_ok = eye[np.isfinite(eye["p_closed"])]
            eye_ok.to_csv(out / "eye_frames.csv", index=False)
            summ = summarize_predictions(eye_ok, ["split"])
            summ.to_csv(out / "eye_summary.csv", index=False)
            report += ["## Eye closure, frame level (labelled open vs closed frames)", "",
                       f"{judged} labelled frames at 15 FPS; {len(eye_ok)} had a face and both-eye prediction "
                       f"({len(eye_ok) / judged:.1%}). Closed frames: {int(eye_ok['label_closed'].sum())}.", "",
                       to_markdown(summ), ""]
        ys_all = yawn_summary(yawn_rows) if yawn_rows else {}
        ys_test = yawn_summary([r for r in yawn_rows if r["split"] == "test"]) if any(
            r["split"] == "test" for r in yawn_rows) else {}
        per_session = pd.DataFrame([{k: v for k, v in r.items() if k not in ("score", "by_type")} |
                                    {"n_yawns": r["score"].n_gt, "n_hit": r["score"].n_hit,
                                     "false_alarms": r["score"].n_false} for r in yawn_rows])
        per_session.to_csv(out / "yawn_per_session.csv", index=False)
        report += ["## Yawns, event level (merged yawn intervals)", "",
                   "| set | people | yawns | detected | recall | false alarms | per hour | median delay s | recall by type |",
                   "|---|---|---|---|---|---|---|---|---|"]
        for name, s in (("all DMD people", ys_all), ("R2 test split", ys_test)):
            if s:
                report.append(f"| {name} | {s['n_people']} | {s['n_yawns']} | {s['n_hit']} | {s['recall']:.3f} | "
                              f"{s['false_alarms']} | {s['false_alarms_per_hour']:.2f} | {s['median_delay_s']:.2f} | "
                              f"{s['recall_by_type']} |")
        report += ["", "Per-session mouth calibration: median outer MAR over the first "
                   f"{args.calib_s:.0f} s of each session. With-hand yawns hide the mouth, so MAR is expected "
                   "to miss some of them; R6's temporal model and R5's head pose can help there.", ""]
        (out / "yawn_summary.json").write_text(json.dumps({"all": ys_all, "test": ys_test}, indent=2, default=str),
                                               encoding="utf-8")

    if args.stress_root and args.stress_manifest:
        man = pd.read_csv(args.stress_manifest)
        print(f"[B] stress set: {len(man)} samples")
        st = stress_eval(man, Path(args.stress_root), pipeline_factory("image"))
        st.to_csv(out / "stress_per_image.csv", index=False)
        ss = stress_summary(st)
        ss.to_csv(out / "stress_summary.csv", index=False)
        report += ["## Stress set (R2, held-out people 1, 5, 6, 7)", "",
                   "Paired with the unstressed original of the same frame. "
                   "`yawn_mar_kept` = median stressed/original inner-lip MAR on yawn frames.", "",
                   df_to_md(ss), ""]

    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
