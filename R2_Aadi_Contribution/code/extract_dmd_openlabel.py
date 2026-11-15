"""Convert DMD OpenLABEL annotation files into a flat event CSV.

Works on the Kaggle DMD mirror layout (dmd_video_v1/annotations/<task>/<session>/<session>_<kind>.json)
and joins session metadata from metadata/dmd_kaggle_session_inventory.csv.

Usage (from the R2_Aadi_Contribution folder):
    python code/extract_dmd_openlabel.py --root "<shared Drive>/NeuroNova/datasets/external/dmd/dmd_video_v1" \
        --out "<shared Drive>/NeuroNova/manifests/dmd_events_all.csv"

Options:
    --groups yawning eyes_state driver_actions   keep only these label groups (default: all)
    --merge-yawn-phases                          merge adjacent "Yawning without hand" and
                                                 "Yawning with hand" segments into one yawn event
    --fps 30                                     also write start_s and end_s using this fps.
                                                 Use the value confirmed by audit_dmd_download.py.

Frame numbers follow the face camera. Body and hands streams are shifted by the
per-session frame_shift recorded in the inventory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INVENTORY = HERE.parent / "metadata" / "dmd_kaggle_session_inventory.csv"
KINDS = ("distraction", "drowsiness", "gaze", "hands")


def load_events(path: Path) -> list[tuple[str, int, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    root = data.get("openlabel") or data.get("vcd") or data
    events = []
    for action in (root.get("actions") or {}).values():
        label = action.get("type") or action.get("name") or ""
        for fi in action.get("frame_intervals") or []:
            events.append((label, int(fi["frame_start"]), int(fi["frame_end"])))
    return events


def merge_yawn_phases(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        prev = out[-1] if out else None
        if (prev and r["native_label"].startswith("yawning/") and prev["native_label"].startswith("yawning/")
                and prev["session_id"] == r["session_id"] and r["start_frame"] <= prev["end_frame"] + 1):
            prev["end_frame"] = max(prev["end_frame"], r["end_frame"])
            prev["native_label"] = "yawning/merged"
            continue
        out.append(dict(r))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="Path to dmd_video_v1")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--merge-yawn-phases", action="store_true")
    ap.add_argument("--fps", type=float, default=None)
    args = ap.parse_args()

    meta = {r["session_id"]: r for r in csv.DictReader(open(INVENTORY, newline="", encoding="utf-8"))}
    rows: list[dict] = []
    missing = []
    for path in sorted((args.root / "annotations").rglob("*.json")):
        stem = path.stem
        kind = stem.rsplit("_", 1)[-1]
        sid = stem[: -(len(kind) + 1)]
        if kind not in KINDS:
            continue
        m = meta.get(sid)
        if m is None:
            missing.append(sid)
            continue
        for label, s, e in load_events(path):
            if args.groups and label.split("/", 1)[0] not in args.groups:
                continue
            rows.append({"session_id": sid, "subject_id": m["subject_id"], "kaggle_split": m["kaggle_split"],
                         "source_task": m["source_task"], "annotation_file": kind, "native_label": label,
                         "start_frame": s, "end_frame": e})
    rows.sort(key=lambda r: (r["session_id"], r["annotation_file"], r["start_frame"], r["native_label"]))
    if args.merge_yawn_phases:
        yawns = [r for r in rows if r["native_label"].startswith("yawning/")]
        others = [r for r in rows if not r["native_label"].startswith("yawning/")]
        rows = sorted(others + merge_yawn_phases(yawns),
                      key=lambda r: (r["session_id"], r["annotation_file"], r["start_frame"], r["native_label"]))

    fields = ["session_id", "subject_id", "kaggle_split", "source_task", "annotation_file", "native_label",
              "start_frame", "end_frame", "n_frames"]
    if args.fps:
        fields += ["start_s", "end_s"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            r["n_frames"] = r["end_frame"] - r["start_frame"] + 1
            if args.fps:
                r["start_s"] = f"{r['start_frame'] / args.fps:.3f}"
                r["end_s"] = f"{(r['end_frame'] + 1) / args.fps:.3f}"
            w.writerow(r)
    print(f"{len(rows)} events written to {args.out.name}")
    if missing:
        print(f"Warning: {len(set(missing))} annotation sessions not in the inventory: {sorted(set(missing))[:5]}")


if __name__ == "__main__":
    main()
