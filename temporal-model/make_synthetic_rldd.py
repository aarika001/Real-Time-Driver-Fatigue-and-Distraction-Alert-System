"""Write a small FAKE dataset in the same layout as RLDD-preprocess/preprocess-rldd.py
output, so the temporal-model code can be tested without the real (large) dataset.

    python make_synthetic_rldd.py --out_dir /tmp/synthetic_rldd

The signals are invented (fatigue = yawn bursts, slow head nods, lower image
sharpness, noisier eyes-region quality).  Accuracy on this data says NOTHING about
real-world performance - it only proves the pipeline runs end to end.  The eye-crop
arrays are omitted because the temporal model never reads them.
"""
import argparse
import os

import numpy as np
import pandas as pd

FPS = 15


def render_eyes(rng, closed, det, bright):
    """(N,64,64) uint8 crops: bright skin, dark elliptical eye whose height is ~0 when closed."""
    n = len(closed)
    yy, xx = np.mgrid[0:64, 0:64]
    semi_y = np.where(closed, 1.2, rng.uniform(9, 13, n))
    inside = ((xx[None] - 32) / 22.0) ** 2 + ((yy[None] - 32) / semi_y[:, None, None]) ** 2 <= 1.0
    img = np.where(inside, 0.12, np.clip(bright, 0.45, 0.8)[:, None, None]) + 0.02 * rng.randn(n, 64, 64)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img[~det.astype(bool)] = 0
    return img


def eye_closure(rng, label, n):
    """Boolean per-frame 'eye closed': short blinks always, long closures when drowsy."""
    closed = np.zeros(n, bool)
    for _ in range(int(n / (FPS * 4))):                           # a blink about every 4 s
        s = rng.randint(0, n - 6)
        closed[s:s + rng.randint(2, 5)] = True
    k = {"alert": 0.0, "low_vigilant": 0.5, "fatigue": 1.0}[label]
    for _ in range(int(k * n / (FPS * 15))):                      # long closures when drowsy
        s = rng.randint(0, max(1, n - 40))
        closed[s:s + rng.randint(8, 30)] = True
    return closed


def make_video(rng, label, n, subj, with_crops=False):
    t = np.arange(n) / FPS
    mar = subj["mar_mu"] + 0.01 * rng.randn(n)
    pitch = subj["pitch_mu"] + 1.5 * rng.randn(n)
    yaw = subj["yaw_mu"] + 2.0 * rng.randn(n)
    roll = subj["roll_mu"] + 1.0 * rng.randn(n)
    bright = np.clip(subj["bright"] + 0.02 * rng.randn(n), 0, 1)
    contrast = np.clip(0.18 + 0.02 * rng.randn(n), 0, 1)
    glare = np.clip(0.01 + 0.01 * rng.rand(n), 0, 1)
    sharp = np.abs(subj["sharp"] + 20 * rng.randn(n))

    if label in ("fatigue", "low_vigilant"):
        k = 1.0 if label == "fatigue" else 0.4
        for _ in range(int(k * n / (FPS * 25))):            # a yawn roughly every 25-60 s
            s = rng.randint(0, max(1, n - 60))
            mar[s:s + rng.randint(20, 45)] += rng.uniform(0.25, 0.5)
        nod = k * 6.0 * np.clip(np.sin(2 * np.pi * t / rng.uniform(25, 40)), 0, None)
        pitch = pitch - nod                                   # slow head drops
        yaw += k * 2.0 * rng.randn(n)
        contrast -= 0.03 * k
        sharp *= (1 - 0.25 * k)
    det = (rng.rand(n) > 0.03).astype(np.uint8)               # ~3 % lost frames

    pose = np.stack([pitch, yaw, roll], 1)
    # solvePnP pitch often sits near +/-180 deg: emulate the wrap
    pose[:, 0] = ((pose[:, 0] + 180) % 360) - 180
    quality = np.stack([bright, contrast, glare, sharp], 1)

    mar = np.where(det == 1, mar, 0.0)
    pose[det == 0] = 0.0
    quality[det == 0] = [0.0, 0.0, 1.0, 0.0]
    out = {"mar": mar.astype(np.float32), "head_pose": pose.astype(np.float32),
           "quality": quality.astype(np.float32), "face_detected": det}
    if with_crops:
        closed = eye_closure(rng, label, n)
        out["left_eye_crops"] = render_eyes(rng, closed, det, bright)
        out["right_eye_crops"] = render_eyes(rng, closed, det, bright)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/tmp/synthetic_rldd")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--subjects_per_fold", type=int, default=4)
    ap.add_argument("--frames", type=int, default=2400, help="frames per video at 15 FPS (2400 = 160 s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with_crops", action="store_true", help="also write fake 64x64 left/right eye crops")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    rows, sid = [], 0
    for f in range(1, args.folds + 1):
        for _ in range(args.subjects_per_fold):
            sid += 1
            subj = {
                "mar_mu": rng.uniform(0.03, 0.12),
                "pitch_mu": rng.choice([-1, 1]) * rng.uniform(165, 178),   # tests angle wrapping
                "yaw_mu": rng.uniform(-8, 8), "roll_mu": rng.uniform(-5, 5),
                "bright": rng.uniform(0.3, 0.6), "sharp": rng.uniform(150, 400),
            }
            for kss, label in (("0", "alert"), ("5", "low_vigilant"), ("10", "fatigue")):
                data = make_video(rng, label, args.frames, subj, args.with_crops)
                d = os.path.join(args.out_dir, f"Fold{f}", f"subject_{sid:02d}")
                os.makedirs(d, exist_ok=True)
                path = os.path.join(d, f"kss_{kss}_{label}.npz")
                np.savez_compressed(path, **data)
                rows.append({"fold_id": f"Fold{f}", "subject_id": f"{sid:02d}", "raw_kss": kss, "label": label,
                             "video_path": f"fake/{sid:02d}/{kss}.mp4", "npz_path": path,
                             "num_frames_15fps": args.frames, "face_detection_rate": float(data["face_detected"].mean())})
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "rldd_metadata_index.csv"), index=False)
    print(f"wrote {len(rows)} fake videos for {sid} subjects to {args.out_dir}")


if __name__ == "__main__":
    main()
