"""Real-Time Driver Fatigue and Distraction Alert System - demo app (R8).

    python main.py                     # webcam 0
    python main.py --source clip.mp4   # replay a video file
"""
import argparse
import cv2
import numpy as np
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import config
import degrade
import reliability
from audio import AudioAlert
from calibration import Calibrator
from capture import FrameSource
from interfaces import Decision, TemporalOut
from overlay import Hud, draw_hud
from perception import Perception
from policy import AlertPolicy
from temporal import TemporalModel
from timing import Timers


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--max-width", type=int, default=960)
    ap.add_argument("--display-width", type=int, default=960,
                    help="HUD is drawn at this width so text stays crisp (0 = native frame size)")
    ap.add_argument("--calib-seconds", type=float, default=config.CALIB_SECONDS)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--log", default="timing_log.csv", help="per-stage timing CSV ('' to disable)")
    ap.add_argument("--save-out", default="", help="write the annotated video here (e.g. demo.mp4)")
    return ap.parse_args()


def main():
    args = parse()
    src = FrameSource(args.source, args.width, args.height, mirror=not args.no_mirror, max_width=args.max_width)
    perception = Perception()
    calib = Calibrator(args.calib_seconds)
    temporal, policy = TemporalModel(), AlertPolicy()
    audio = AudioAlert(not args.no_audio)
    timers = Timers(args.log or None)
    deg = degrade.DegradeState()
    hud = Hud()
    anchors, writer, fails = None, None, 0
    idle_tout, idle_dec = TemporalOut(np.array([1.0, 0.0, 0.0])), Decision()

    win = "Driver Fatigue & Distraction Alert (NeuroNova)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if args.display_width:
        cv2.resizeWindow(win, args.display_width, int(args.display_width * args.height / args.width))

    while True:
        timers.begin()
        with timers.stage("capture"):
            ok, frame, t = src.read()
        if not ok:
            if src.is_file:
                break
            fails += 1
            if fails >= 5:
                print("Camera not delivering frames.")
                break
            continue
        fails = 0

        with timers.stage("degrade"):
            frame = degrade.apply(frame, deg, anchors)
        with timers.stage("perception"):
            p = perception.process(frame, t)
        anchors = degrade.anchors_from(p)
        with timers.stage("reliability"):
            rel = reliability.estimate(p)

        if not calib.done:
            calib.add(p, t)
            tout, decision = idle_tout, idle_dec
        else:
            with timers.stage("temporal"):
                tout = temporal.update(calib.normalise(p), rel, t)
            with timers.stage("policy"):
                decision = policy.decide(tout, rel, t)
            audio.handle(decision, t)

        with timers.stage("draw"):
            hud.fps, hud.percept, hud.rel, hud.tout, hud.decision = timers.fps, p, rel, tout, decision
            hud.calib_progress = calib.progress(t)
            hud.calib_waiting_face = calib.waiting_for_face(t)
            hud.effects, hud.severity = deg.active, deg.severity
            hud.miss_ratio, hud.muted = policy.miss_ratio, audio.muted
            if args.display_width and frame.shape[1] != args.display_width:
                k = args.display_width / frame.shape[1]
                frame = cv2.resize(frame, (args.display_width, int(frame.shape[0] * k)),
                   interpolation=cv2.INTER_LINEAR)
                hud.lm_scale = k
            draw_hud(frame, hud)

        if args.save_out:
            if writer is None:
                hh, ww = frame.shape[:2]
                writer = cv2.VideoWriter(args.save_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                         src.fps if src.is_file else 20.0, (ww, hh))
            writer.write(frame)

        with timers.stage("display"):
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
        timers.end_frame(t)

        if key in (ord("q"), 27):
            break
        elif key in degrade.KEYS:
            deg.toggle(degrade.KEYS[key])
        elif key == ord("0"):
            deg.active.clear()
        elif key in (ord("+"), ord("=")):
            deg.bump(0.1)
        elif key == ord("-"):
            deg.bump(-0.1)
        elif key == ord("c"):
            calib.reset()
            temporal, policy = TemporalModel(), AlertPolicy(policy.miss_ratio)
        elif key == ord("["):
            policy.set_ratio(policy.miss_ratio - 1)
        elif key == ord("]"):
            policy.set_ratio(policy.miss_ratio + 1)
        elif key == ord("m"):
            audio.muted = not audio.muted
        elif key == ord("h"):
            hud.show_help = not hud.show_help
        elif key == ord("l"):
            hud.show_landmarks = not hud.show_landmarks

    src.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    timers.close()


if __name__ == "__main__":
    main()
