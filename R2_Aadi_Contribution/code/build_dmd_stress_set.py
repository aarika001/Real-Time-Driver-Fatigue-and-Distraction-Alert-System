"""Build the R2 synthetic stress-test set from held-out DMD people and validate it.

Inputs
- metadata/dmd_subject_split.csv           held-out people (stress_test_held_out = yes)
- metadata/dmd_kaggle_session_inventory.csv session list and Kaggle split
- --events: dmd_drowsiness_events_all.csv   from code/extract_dmd_openlabel.py
- --root:   downloaded dmd_video_v1 folder (face videos)

For each held-out person's drowsiness session the script picks source frames in
three native states (yawn, eyes closed, eyes open with no yawn), finds the face box,
and applies every stress type and severity from code/stress_generators.py.

Outputs (all under --out, which must be on the restricted team storage, never Git)
- dmd/<subject>/<frame_id>/original.png and one PNG per stress sample
- stress_test_manifest_dmd.csv   one row per generated sample (template columns first)
- stress_validation_dmd.csv      per-image measurements
- stress_validation_summary.json per stress and severity summary

Requires opencv-python (with CascadeClassifier), numpy, Pillow.
Usage (from the R2_Aadi_Contribution folder):
    python code/build_dmd_stress_set.py --root "C:/NeuroNova_data/dmd/dmd_video_v1" \
        --events "C:/NeuroNova_data/dmd/r2_manifests/dmd_drowsiness_events_all.csv" \
        --out "C:/NeuroNova_data/generated_stress"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stress_generators import SEVERITIES, apply_stress  # noqa: E402

META = HERE.parent / "metadata"
BASE_SEED = 42
FRAMES_PER_STATE = 2
OCCLUSION_KINDS = ("mask", "cap_brim", "hand_patch")
TEMPLATE = ["source_dataset", "source_person_id", "source_video_or_image_id", "split", "stress_type", "severity",
            "parameter_summary", "random_seed", "output_path"]
EXTRA = ["source_frame", "source_state", "face_box_xywh", "face_box_source", "variant"]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def merge_intervals(iv: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def pick_frames(events: list[dict]) -> list[tuple[int, str]]:
    by: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in events:
        by[r["native_label"]].append((int(r["start_frame"]), int(r["end_frame"])))
    yawns = merge_intervals([iv for k, v in by.items() if k.startswith("yawning/") for iv in v])
    closes = sorted(by.get("eyes_state/close", []), key=lambda iv: iv[1] - iv[0], reverse=True)
    opens = sorted(by.get("eyes_state/open", []), key=lambda iv: iv[1] - iv[0], reverse=True)

    def far_from_yawn(iv, margin=60):
        return all(iv[1] < ys - margin or iv[0] > ye + margin for ys, ye in yawns)

    picks = [((s + e) // 2, "yawn") for s, e in yawns[:FRAMES_PER_STATE]]
    picks += [((s + e) // 2, "eyes_closed") for s, e in closes[:FRAMES_PER_STATE]]
    picks += [((s + e) // 2, "eyes_open_no_yawn") for s, e in [iv for iv in opens if far_from_yawn(iv)][:FRAMES_PER_STATE]]
    return sorted(picks)


def read_frames(video: Path, wanted: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    got: dict[int, np.ndarray] = {}
    target = set(wanted)
    idx = 0
    while target - set(got):
        ok, frame = cap.read()
        if not ok:
            break
        if idx in target:
            got[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()
    return got


class Detector:
    def __init__(self) -> None:
        base = cv2.data.haarcascades
        self.face = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self.eye = cv2.CascadeClassifier(base + "haarcascade_eye.xml")
        if self.face.empty() or self.eye.empty():
            raise RuntimeError("OpenCV Haar cascades not found; install opencv-python 4.x")

    def faces(self, rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        found = self.face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))
        return sorted([tuple(map(int, f)) for f in found], key=lambda b: b[2] * b[3], reverse=True)

    def eyes(self, rgb: np.ndarray, box: tuple[int, int, int, int]) -> int:
        x, y, w, h = box
        roi = cv2.cvtColor(rgb[y:y + h // 2 + h // 8, x:x + w], cv2.COLOR_RGB2GRAY)
        return len(self.eye.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=6, minSize=(w // 10, w // 10)))


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter) if inter else 0.0


def measure(det: Detector, rgb: np.ndarray, ref_box) -> dict:
    x, y, w, h = ref_box
    roi = rgb[y:y + h, x:x + w].astype(np.float32)
    luma = 0.299 * roi[..., 0] + 0.587 * roi[..., 1] + 0.114 * roi[..., 2]
    faces = det.faces(rgb)
    match = [f for f in faces if iou(f, ref_box) >= 0.3]
    return {
        "face_luma_mean": round(float(luma.mean()), 1),
        "face_saturated_frac": round(float((luma >= 250).mean()), 4),
        "face_detected": int(bool(match)),
        "eyes_detected": det.eyes(rgb, match[0]) if match else 0,
    }


def param_summary(stress: str, severity: str, kind: str | None) -> str:
    p = dict(SEVERITIES[stress][severity])
    if stress == "occlusion":
        p["kind"] = kind
    if stress == "glare":
        p["placement"] = "eye band of face box (30 to 70 percent width, 30 to 45 percent height)"
    if stress == "off_axis":
        p["direction"] = 1
    if stress == "face_lost":
        p = {"operation": "face box filled black"}
    return "; ".join(f"{k}={v}" for k, v in p.items())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--subjects", nargs="*", default=None, help="Process only these held-out subject IDs (one part file each)")
    ap.add_argument("--merge", action="store_true", help="Combine part files into the final manifest, validation and summary")
    args = ap.parse_args()

    if any((p / ".git").exists() for p in [args.out, *args.out.parents]):
        sys.exit("Refusing to write stress outputs inside a Git repository.")

    parts = args.out / "_parts"
    if args.merge:
        merge_parts(args.out, parts)
        return

    held = {r["subject_id"] for r in read_csv(META / "dmd_subject_split.csv") if r["stress_test_held_out"] == "yes"}
    inv = [r for r in read_csv(META / "dmd_kaggle_session_inventory.csv")
           if r["subject_id"] in held and r["source_task"] == "drowsiness"
           and (not args.subjects or r["subject_id"] in args.subjects)]
    events = read_csv(args.events)
    det = Detector()

    manifest, validation = [], []
    for sess in sorted(inv, key=lambda r: int(r["subject_id"])):
        sid, subj, split = sess["session_id"], sess["subject_id"], sess["kaggle_split"]
        video = args.root / "videos" / split / "drowsiness" / sid / f"{sid}_face.mp4"
        if not video.exists():
            print(f"missing video, skipped: {video.name}")
            continue
        picks = pick_frames([e for e in events if e["session_id"] == sid])
        frames = read_frames(video, [f for f, _ in picks])
        for n, (fno, state) in enumerate(picks):
            rgb = frames.get(fno)
            if rgb is None:
                print(f"frame {fno} not read from {video.name}")
                continue
            faces = det.faces(rgb)
            if faces:
                box, box_src = faces[0], "haar_frontalface"
            else:
                h, w = rgb.shape[:2]
                box, box_src = (w // 4, h // 8, w // 2, h * 3 // 4), "default_center"
            frame_id = f"{sid.split('_20')[0]}_f{fno:05d}_{state}"
            folder = args.out / "dmd" / f"subject_{int(subj):02d}" / frame_id
            folder.mkdir(parents=True, exist_ok=True)
            orig = Image.fromarray(rgb)
            orig.save(folder / "original.png")
            seed = BASE_SEED + int(subj) * 100 + n
            base_row = {"source_dataset": "DMD (Kaggle dmd-video-v1)", "source_person_id": subj,
                        "source_video_or_image_id": f"{sid}_face.mp4", "split": f"{split} (held-out)",
                        "source_frame": fno, "source_state": state, "face_box_xywh": " ".join(map(str, box)),
                        "face_box_source": box_src}
            m0 = measure(det, rgb, box)
            validation.append({"frame_id": frame_id, "subject": subj, "state": state, "stress_type": "none",
                               "severity": "none", "variant": "", **m0})
            jobs = []
            for stress, levels in SEVERITIES.items():
                if stress == "face_lost":
                    jobs.append((stress, "single_level", None))
                    continue
                for sev in levels:
                    if stress == "occlusion":
                        jobs += [(stress, sev, k) for k in OCCLUSION_KINDS]
                    else:
                        jobs.append((stress, sev, None))
            for stress, sev, kind in jobs:
                region = box
                level = "severe" if sev == "single_level" else sev
                img = apply_stress(orig, stress, level, seed=seed, face_box=region, occlusion_kind=kind or "mask")
                name = f"{stress}_{sev}" + (f"_{kind}" if kind else "") + ".png"
                img.save(folder / name)
                rel = (folder / name).relative_to(args.out).as_posix()
                manifest.append({**base_row, "stress_type": stress, "severity": sev,
                                 "parameter_summary": param_summary(stress, level, kind),
                                 "random_seed": seed, "output_path": f"generated_stress/{rel}", "variant": kind or ""})
                validation.append({"frame_id": frame_id, "subject": subj, "state": state, "stress_type": stress,
                                   "severity": sev, "variant": kind or "", **measure(det, np.asarray(img), box)})
        print(f"subject {subj}: {len(picks)} source frames from {sid}")

    parts.mkdir(parents=True, exist_ok=True)
    tag = "_".join(sorted(args.subjects, key=int)) if args.subjects else "all"
    write_csv(parts / f"manifest_{tag}.csv", TEMPLATE + EXTRA, manifest)
    if validation:
        write_csv(parts / f"validation_{tag}.csv", list(validation[0].keys()), validation)
    print(f"{len(manifest)} stress samples written for subjects {tag}. Run with --merge when all subjects are done.")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def merge_parts(out: Path, parts: Path) -> None:
    manifest = [r for p in sorted(parts.glob("manifest_*.csv")) for r in read_csv(p)]
    validation = [r for p in sorted(parts.glob("validation_*.csv")) for r in read_csv(p)]
    key = lambda r: (int(r["source_person_id"]), r["output_path"])
    manifest.sort(key=key)
    write_csv(out / "stress_test_manifest_dmd.csv", TEMPLATE + EXTRA, manifest)
    write_csv(out / "stress_validation_dmd.csv", list(validation[0].keys()), validation)
    groups: dict[str, list[dict]] = defaultdict(list)
    for v in validation:
        groups[f"{v['stress_type']}|{v['severity']}"].append(v)
    summary = {}
    for k, rows in sorted(groups.items()):
        n = len(rows)
        summary[k] = {"images": n,
                      "face_detect_rate": round(sum(int(r["face_detected"]) for r in rows) / n, 3),
                      "eyes_found_rate": round(sum(int(r["eyes_detected"]) > 0 for r in rows) / n, 3),
                      "face_luma_mean": round(sum(float(r["face_luma_mean"]) for r in rows) / n, 1),
                      "face_saturated_frac": round(sum(float(r["face_saturated_frac"]) for r in rows) / n, 4)}
    (out / "stress_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    people = sorted({r["source_person_id"] for r in manifest}, key=int)
    print(f"merged {len(manifest)} stress samples from people {people}; summary groups {len(summary)}")

if __name__ == "__main__":
    main()
