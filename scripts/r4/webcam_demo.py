"""Live R4 test: webcam (or video file) -> eye P(closed) + MAR overlays + FPS (Milestone 6).

Keys in the window:  c = calibrate (look ahead, mouth relaxed, for --calib_s seconds)
                     q or Esc = quit
At the end it prints latency per stage (landmarks, crops, eye CNN, mouth) and the
achieved FPS, and saves them to --report, as evidence for the >= 15 FPS target.

Examples (from the repo root, venv active):
    python scripts/r4/webcam_demo.py --eye_model runs/eye_cnn/eye_cnn.onnx
    python scripts/r4/webcam_demo.py --eye_model runs/eye_cnn/best.pt --camera 1
    python scripts/r4/webcam_demo.py --eye_model runs/eye_cnn/eye_cnn.onnx --benchmark_frames 300 --no_window
    python scripts/r4/webcam_demo.py --video some_clip.mp4 --no_window          # mouth only, no eye model
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cues.config import load_config  # noqa: E402
from cues.inference_r4 import FrameResult, R4Pipeline  # noqa: E402
from cues.mouth_cue import MouthConfig  # noqa: E402

GREEN, RED, YELLOW, WHITE = (60, 200, 60), (40, 40, 230), (0, 210, 255), (255, 255, 255)


def draw_overlay(frame: np.ndarray, res: FrameResult, fps: float, status: str) -> np.ndarray:
    out = frame.copy()
    d = res.output.to_flat_dict()
    for (x1, y1, x2, y2), key in zip(res.eye_boxes, ("p_closed_left", "p_closed_right")):
        p = d[key]
        color = WHITE if not np.isfinite(p) else (RED if p >= 0.5 else GREEN)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if np.isfinite(p):
            cv2.putText(out, f"{p:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def fmt(v, spec=".2f"):
        return "--" if not np.isfinite(v) else format(v, spec)

    lines = [
        f"FPS {fps:4.1f} | {res.timings_ms.get('total', 0):5.1f} ms/frame",
        f"face {'yes' if d['face_detected'] else 'NO'} | P(closed) {fmt(d['p_closed'])} | EAR {fmt(d['ear'], '.3f')}",
        f"MAR {fmt(d['mar'])} | inner {fmt(d['mar_inner'])} | norm {fmt(d['mar_norm'])}",
        f"open {fmt(d['mouth_open_run_s'], '.1f')}s | osc {fmt(d['mouth_osc_hz'], '.1f')}Hz | glare {fmt(d['eye_glare_frac'], '.3f')}",
        status,
    ]
    for i, text in enumerate(lines):
        y = 20 + 20 * i
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    if d["yawn_flag"] == 1.0:
        cv2.putText(out, "YAWN", (out.shape[1] - 110, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, YELLOW, 3)
    return out


def summarize(timings: list[dict], wall_s: float, n_frames: int) -> dict:
    report = {"frames": n_frames, "wall_seconds": round(wall_s, 2),
              "fps_overall": round(n_frames / wall_s, 2) if wall_s > 0 else None,
              "machine": platform.processor() or platform.machine(), "python": platform.python_version()}
    for key in ("landmarks", "crops", "eye_cnn", "mouth", "total"):
        vals = np.array([t[key] for t in timings if key in t])
        if len(vals):
            report[f"{key}_ms_mean"] = round(float(vals.mean()), 2)
            report[f"{key}_ms_p95"] = round(float(np.percentile(vals, 95)), 2)
    if "total_ms_mean" in report:
        report["fps_processing_only"] = round(1000.0 / report["total_ms_mean"], 1)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eye_model", default=None, help=".onnx or .pt eye model (omit for mouth-only)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", default=None, help="use a video file instead of the webcam")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--calib_s", type=float, default=10.0, help="calibration length when you press c")
    ap.add_argument("--benchmark_frames", type=int, default=0, help="stop after N frames (0 = run until q)")
    ap.add_argument("--no_window", action="store_true", help="headless (benchmarks, servers)")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for the eye model")
    ap.add_argument("--config", default=None)
    ap.add_argument("--report", default="results/r4/fps_report.json")
    args = ap.parse_args(argv)

    predictor = None
    if args.eye_model:
        from cues.predictors import load_eye_predictor
        predictor = load_eye_predictor(args.eye_model, num_threads=args.threads)
    mouth_cfg = MouthConfig.from_dict(load_config(args.config).get("mouth"))
    pipe = R4Pipeline(eye_predictor=predictor, mouth_cfg=mouth_cfg)

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("ERROR: could not open " + (args.video or f"camera {args.camera}"), file=sys.stderr)
        return 2
    video_fps = cap.get(cv2.CAP_PROP_FPS) if args.video else 0

    timings, n, fps = [], 0, 0.0
    calib_end = None
    status = "press c to calibrate (mouth relaxed, look ahead)"
    t_start = last = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = (n / video_fps) if video_fps else (time.perf_counter() - t_start)
        res = pipe.process_frame(frame, ts)
        timings.append(res.timings_ms)
        n += 1
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - last)) if n > 1 else 0.0
        last = now

        if calib_end is not None and ts >= calib_end:
            calib_end = None
            try:
                base = pipe.finish_calibration()
                status = f"calibrated: neutral MAR {base:.3f}"
            except ValueError as e:
                status = f"calibration failed ({e}); press c to retry"
        elif calib_end is not None:
            status = f"calibrating... {calib_end - ts:4.1f}s left"

        if not args.no_window:
            cv2.imshow("R4 cues (q to quit, c to calibrate)", draw_overlay(frame, res, fps, status))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c") and calib_end is None:
                pipe.start_calibration()
                calib_end = ts + args.calib_s
        if args.benchmark_frames and n >= args.benchmark_frames:
            break

    wall = time.perf_counter() - t_start
    cap.release()
    if not args.no_window:
        cv2.destroyAllWindows()
    if n == 0:
        print("ERROR: no frames read", file=sys.stderr)
        return 2
    report = summarize(timings, wall, n)
    report["eye_model"] = args.eye_model or "none"
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    target = report.get("fps_processing_only") or 0
    print(f"\nProcessing speed {target} FPS -> {'meets' if target >= 15 else 'BELOW'} the 15 FPS target "
          f"(report saved to {args.report})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
