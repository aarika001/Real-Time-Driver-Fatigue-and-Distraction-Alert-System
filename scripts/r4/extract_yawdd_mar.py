"""Extract per-frame MAR from YawDD videos (R4, Milestone 2).

For every video under --videos_dir:
  1. sample frames at --target_fps (time-based, so 25/30/60 FPS sources all
     give a true 15 FPS series),
  2. run MediaPipe Face Landmarker (video mode),
  3. store outer MAR (RLDD-compatible), inner 3-D MAR and the face-detected
     flag in <out_dir>/<video_id>.npz,
and write <out_dir>/index.csv with one row per video.

Already-processed videos are skipped, so the script can be stopped and resumed.

Example (Windows PowerShell, from the repo root):
    python scripts/r4/extract_yawdd_mar.py --videos_dir D:/datasets/YawDD --out_dir data/yawdd_mar
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cues.face_landmarker import FaceLandmarker  # noqa: E402
from cues.mouth_cue import mar_inner, mar_outer  # noqa: E402
from cues.yawdd import parse_yawdd_path  # noqa: E402

VIDEO_EXTS = (".avi", ".mp4", ".mov", ".m4v", ".mkv")


def extract_video(path: Path, model_path: Path | None, target_fps: float) -> dict[str, np.ndarray] | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or not np.isfinite(src_fps) or src_fps <= 0:
        src_fps = 30.0

    mars, mars_in, detected, times = [], [], [], []
    next_t = 0.0
    frame_idx = 0
    with FaceLandmarker(model_path, mode="video") as lm:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / src_fps
            frame_idx += 1
            if t + 1e-9 < next_t:
                continue
            next_t += 1.0 / target_fps
            pts = lm.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(round(t * 1000)))
            times.append(t)
            if pts is None:
                mars.append(np.nan)
                mars_in.append(np.nan)
                detected.append(0)
            else:
                mars.append(mar_outer(pts))
                mars_in.append(mar_inner(pts))
                detected.append(1)
    cap.release()
    if not times:
        return None
    return {
        "mar": np.asarray(mars, dtype=np.float32),
        "mar_inner": np.asarray(mars_in, dtype=np.float32),
        "face_detected": np.asarray(detected, dtype=np.uint8),
        "t_s": np.asarray(times, dtype=np.float32),
        "fps": np.float32(target_fps),
        "source_fps": np.float32(src_fps),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos_dir", required=True, help="YawDD root (searched recursively)")
    ap.add_argument("--out_dir", required=True, help="where .npz files and index.csv go")
    ap.add_argument("--model_path", default=None, help="face_landmarker.task (default: repo root)")
    ap.add_argument("--target_fps", type=float, default=15.0)
    ap.add_argument("--limit", type=int, default=0, help="process at most N videos (0 = all)")
    args = ap.parse_args(argv)

    videos_dir, out_dir = Path(args.videos_dir), Path(args.out_dir)
    if not videos_dir.is_dir():
        print(f"ERROR: videos_dir not found: {videos_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path) if args.model_path else None

    videos = sorted(p for p in videos_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print(f"ERROR: no videos found under {videos_dir}", file=sys.stderr)
        return 2
    ids = [parse_yawdd_path(v.relative_to(videos_dir)).video_id for v in videos]
    dupes = sorted({x for x in ids if ids.count(x) > 1})
    if dupes:
        print(f"ERROR: duplicate video names would overwrite each other: {dupes[:5]}. "
              "Run once per camera folder with separate --out_dir.", file=sys.stderr)
        return 2
    print(f"Found {len(videos)} videos")

    rows = []
    for i, vp in enumerate(videos, 1):
        info = parse_yawdd_path(vp.relative_to(videos_dir))
        out_npz = out_dir / f"{info.video_id}.npz"
        t0 = time.time()
        if out_npz.exists():
            with np.load(out_npz) as z:
                n, rate = len(z["mar"]), float(z["face_detected"].mean())
            status = "cached"
        else:
            data = extract_video(vp, model_path, args.target_fps)
            if data is None:
                print(f"[{i}/{len(videos)}] SKIP unreadable {vp.name}")
                continue
            np.savez_compressed(out_npz, **data)
            n, rate = len(data["mar"]), float(data["face_detected"].mean())
            status = f"{time.time() - t0:.1f}s"
        print(f"[{i}/{len(videos)}] {info.video_id}: {n} frames, face {rate:.0%} ({status})")
        rows.append({**info.__dict__, "rel_path": str(vp.relative_to(videos_dir)),
                     "npz": out_npz.name, "n_frames": n, "face_detection_rate": round(rate, 4)})

    pd.DataFrame(rows).to_csv(out_dir / "index.csv", index=False)
    print(f"Wrote {out_dir / 'index.csv'} ({len(rows)} videos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
