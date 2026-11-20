"""Download selected parts of the Kaggle DMD mirror into a restricted folder.

Source: https://www.kaggle.com/datasets/momarberliansyah/dmd-video-v1
Selection is driven by metadata/dmd_kaggle_session_inventory.csv.

Requirements:
    pip install --upgrade kaggle six
    A Kaggle API token configured by the user (see Kaggle account settings).
    This script never reads, prints, or stores the token.

Examples (run from the R2_Aadi_Contribution folder):
    # Dry run: list what would be downloaded and the total size
    python code/download_dmd_kaggle_subset.py --dest "G:/My Drive/NeuroNova/datasets/external/dmd" --tasks drowsiness --views face --dry-run

    # Annotations and manifests only (about 79 MB)
    python code/download_dmd_kaggle_subset.py --dest "<shared Drive>/NeuroNova/datasets/external/dmd" --annotations-only

    # Drowsiness face videos plus their annotations (about 5.7 GB)
    python code/download_dmd_kaggle_subset.py --dest "<shared Drive>/NeuroNova/datasets/external/dmd" --tasks drowsiness --views face

The destination must be the restricted team Drive, never the Git repository.
Files keep the Kaggle relative paths under <dest>/dmd_video_v1/.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET = "momarberliansyah/dmd-video-v1"
ROOT = "dmd_video_v1"
HERE = Path(__file__).resolve().parent
INVENTORY = HERE.parent / "metadata" / "dmd_kaggle_session_inventory.csv"
TASKS = ("drowsiness", "distraction", "gaze")
VIEWS = ("face", "body", "hands", "mosaic")
EXT = {"face": "mp4", "body": "mp4", "hands": "mp4", "mosaic": "avi"}


# Run the Kaggle tool through the same Python interpreter, so it works even when
# the "kaggle" command is not on PATH (common on Windows).
KAGGLE_CMD = [sys.executable, "-m", "kaggle"]


def preflight() -> None:
    home = Path.home() / ".kaggle"
    has_cred = bool(os.environ.get("KAGGLE_API_TOKEN")) or (home / "access_token").exists() or (home / "kaggle.json").exists()
    if not has_cred:
        sys.exit("No Kaggle credential found. In PowerShell run:  $env:KAGGLE_API_TOKEN = \"<your new token>\"  then try again.")
    check = subprocess.run(KAGGLE_CMD + ["--version"], capture_output=True, text=True)
    if check.returncode != 0:
        err = check.stderr.strip()
        if "No module named 'kaggle'" in err:
            sys.exit(f"The Kaggle tool is not installed for this Python. Run:  {sys.executable} -m pip install --upgrade kaggle six")
        if "No module named 'six'" in err:
            sys.exit(f"The Kaggle tool needs one more package. Run:  {sys.executable} -m pip install six")
        sys.exit("The Kaggle tool could not start:\n" + err[-800:])
    print(check.stdout.strip())


def planned_files(rows: list[dict], tasks: set[str], views: set[str], annotations_only: bool) -> list[tuple[str, float]]:
    files: list[tuple[str, float]] = [
        (f"{ROOT}/README.md", 0.0),
        (f"{ROOT}/manifests/dmd_sessions.csv", 0.0),
        (f"{ROOT}/manifests/dmd_summary.json", 0.0),
    ]
    for r in rows:
        if r["source_task"] not in tasks:
            continue
        sid, task, split = r["session_id"], r["source_task"], r["kaggle_split"]
        for ann in r["annotation_files"].split(" + "):
            files.append((f"{ROOT}/annotations/{task}/{sid}/{sid}_{ann}", 0.0))
        if annotations_only:
            continue
        for v in views:
            mb = r.get(f"{v}_{EXT[v]}_mb", "")
            if not mb:
                continue  # view not present for this session (for example hands in drowsiness)
            files.append((f"{ROOT}/videos/{split}/{task}/{sid}/{sid}_{v}.{EXT[v]}", float(mb)))
    return files


def download_one(rel: str, dest: Path) -> None:
    target = dest / rel
    if target.exists() and target.stat().st_size > 0:
        print(f"skip (exists): {rel}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    # The Kaggle CLI keeps the file's full relative path under -p, so pass the root.
    cmd = KAGGLE_CMD + ["datasets", "download", DATASET, "-f", rel, "-p", str(dest), "--quiet"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed for {rel}:\n{result.stderr.strip()[-800:]}")
    if not target.exists():
        # Fall back: find the file wherever this CLI version put it, then move it into place.
        name = Path(rel).name
        candidates = [p for p in dest.rglob(name) if p.is_file()] + [p for p in dest.rglob(name + ".zip") if p.is_file()]
        if candidates:
            found = max(candidates, key=lambda p: p.stat().st_mtime)
            target.parent.mkdir(parents=True, exist_ok=True)
            if found.suffix == ".zip" and not name.endswith(".zip"):
                with zipfile.ZipFile(found) as zf:
                    member = next(m for m in zf.namelist() if m.endswith(name))
                    with zf.open(member) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out)
                found.unlink()
            else:
                found.replace(target)
    zipped = target.with_name(target.name + ".zip")
    if not target.exists() and zipped.exists():
        with zipfile.ZipFile(zipped) as zf:
            member = next(m for m in zf.namelist() if m.endswith(target.name))
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
        zipped.unlink()
    if not target.exists():
        raise RuntimeError(f"Download finished but file not found: {target}")
    print(f"ok: {rel}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, type=Path, help="Restricted shared Drive folder for DMD")
    ap.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    ap.add_argument("--views", nargs="+", choices=VIEWS, default=["face"])
    ap.add_argument("--annotations-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if (args.dest / ".git").exists() or any((p / ".git").exists() for p in args.dest.parents):
        sys.exit("Refusing to download into a Git repository. Use the restricted team Drive.")

    rows = list(csv.DictReader(open(INVENTORY, newline="", encoding="utf-8")))
    files = planned_files(rows, set(args.tasks), set(args.views), args.annotations_only)
    total_mb = sum(mb for _, mb in files)
    print(f"{len(files)} files planned, about {total_mb / 1000:.2f} GB of video plus small annotation files")
    if args.dry_run:
        for rel, mb in files:
            print(f"{mb:8.0f} MB  {rel}" if mb else f"{'':8}     {rel}")
        return
    preflight()
    failed = []
    for rel, _ in files:
        try:
            download_one(rel, args.dest)
        except RuntimeError as exc:
            print(exc)
            failed.append(rel)
            if len(failed) >= 3 and len(failed) == files.index((rel, _)) + 1:
                sys.exit("The first downloads all failed. Fix the error above, then run the same command again.")
    if failed:
        print(f"{len(failed)} files failed. Run the same command again to retry them; finished files are skipped.")
    print("Done. Record the acquisition date and file count in metadata/dataset_access_log.csv.")


if __name__ == "__main__":
    main()
