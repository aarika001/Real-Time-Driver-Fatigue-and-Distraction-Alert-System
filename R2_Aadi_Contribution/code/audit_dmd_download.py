"""Audit a downloaded copy of the Kaggle DMD mirror against the R2 inventory.

Checks, for every file expected by metadata/dmd_kaggle_session_inventory.csv:
- present or missing
- size close to the size listed on Kaggle (within 2 MB)
- for videos: fps, frame count, duration, resolution (ffprobe, or OpenCV if ffprobe is absent)
- face video frame count against face_camera_frames from the annotation

Only files that were downloaded are probed, so a partial download (for example only
drowsiness face videos) is fine. The JSON report contains relative paths only.

Usage (from the R2_Aadi_Contribution folder):
    python code/audit_dmd_download.py --root "<shared Drive>/NeuroNova/datasets/external/dmd/dmd_video_v1" \
        --output "<shared Drive>/NeuroNova/manifests/dmd_download_audit.json"
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
INVENTORY = HERE.parent / "metadata" / "dmd_kaggle_session_inventory.csv"
EXT = {"face": "mp4", "body": "mp4", "hands": "mp4", "mosaic": "avi"}


def probe(path: Path) -> dict:
    if shutil.which("ffprobe"):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_read_packets:format=duration",
             "-of", "json", str(path)], capture_output=True, text=True)
        if out.returncode != 0:
            return {"error": out.stderr.strip()[:200] or "ffprobe failed"}
        j = json.loads(out.stdout)
        s = (j.get("streams") or [{}])[0]
        rate = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/1"
        if rate in ("0/0", "0/1"):
            rate = s.get("r_frame_rate", "0/1")
        n, d = rate.split("/")
        return {"codec": s.get("codec_name"), "width": s.get("width"), "height": s.get("height"),
                "fps": round(float(n) / float(d), 3) if float(d) else None,
                "frames": int(s["nb_read_packets"]) if str(s.get("nb_read_packets", "")).isdigit() else None,
                "duration_s": round(float(j.get("format", {}).get("duration", 0) or 0), 2)}
    try:
        import cv2  # type: ignore
    except ImportError:
        return {"error": "neither ffprobe nor opencv-python is available"}
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    res = {"codec": None, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
           "fps": round(fps, 3) if fps else None, "frames": frames or None,
           "duration_s": round(frames / fps, 2) if fps and frames else None}
    cap.release()
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="Path to dmd_video_v1")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--no-probe", action="store_true", help="Skip video probing (presence and size only)")
    args = ap.parse_args()

    inv = list(csv.DictReader(open(INVENTORY, newline="", encoding="utf-8")))
    files, issues = [], []
    fps_counter: Counter = Counter()
    for r in inv:
        sid, task, split = r["session_id"], r["source_task"], r["kaggle_split"]
        for ann in r["annotation_files"].split(" + "):
            rel = f"annotations/{task}/{sid}/{sid}_{ann}"
            files.append({"rel": rel, "kind": "annotation", "present": (args.root / rel).exists()})
        for view, ext in EXT.items():
            mb = r.get(f"{view}_{ext}_mb", "")
            if not mb:
                continue
            rel = f"videos/{split}/{task}/{sid}/{sid}_{view}.{ext}"
            p = args.root / rel
            entry = {"rel": rel, "kind": f"video_{view}", "session_id": sid, "present": p.exists(), "expected_mb": float(mb)}
            if p.exists():
                entry["actual_mb"] = round(p.stat().st_size / 1e6, 1)
                if abs(entry["actual_mb"] - entry["expected_mb"]) > 2:
                    issues.append(f"size mismatch: {rel}")
                if not args.no_probe:
                    entry.update(probe(p))
                    if entry.get("fps"):
                        fps_counter[entry["fps"]] += 1
                    if view == "face" and entry.get("frames") and r.get("face_camera_frames"):
                        diff = entry["frames"] - int(r["face_camera_frames"])
                        entry["frames_minus_annotation"] = diff
                        if abs(diff) > 2:
                            issues.append(f"face frame count differs from annotation by {diff}: {rel}")
                    if entry.get("error"):
                        issues.append(f"probe failed: {rel}")
            files.append(entry)

    present = [f for f in files if f["present"]]
    report = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "kaggle:momarberliansyah/dmd-video-v1",
        "expected_files": len(files),
        "present_files": len(present),
        "present_by_kind": dict(Counter(f["kind"] for f in present)),
        "missing_by_kind": dict(Counter(f["kind"] for f in files if not f["present"])),
        "subjects_present": sorted({int(r["subject_id"]) for r in inv for f in present if f.get("session_id") == r["session_id"]}),
        "fps_values_seen": {str(k): v for k, v in fps_counter.items()},
        "issues": issues,
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"present {len(present)} of {len(files)} expected files; fps seen {report['fps_values_seen']}; issues {len(issues)}")


if __name__ == "__main__":
    main()
