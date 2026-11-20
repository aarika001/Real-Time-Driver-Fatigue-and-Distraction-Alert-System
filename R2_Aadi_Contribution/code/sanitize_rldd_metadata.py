"""Create a path-sanitized copy of the processed UTA-RLDD metadata CSV."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def logical_parts(value: str, root: Path) -> Path | None:
    """Return the part of a path from the processed folder name or the first FoldN folder onward."""

    parts = [p for p in Path(value.replace("\\", "/")).parts if p not in (".", "/", "")]
    if root.name in parts:
        rest = parts[len(parts) - parts[::-1].index(root.name):]
        if rest:
            return Path(*rest)
    for i, part in enumerate(parts):
        if re.match(r"(?i)^fold\d", part):
            return Path(*parts[i:])
    return None


def safe_path(value: str, root: Path) -> str:
    """Return a logical relative path without exposing personal absolute paths.

    Order: path relative to the shared root; otherwise the part from the processed
    folder name or the first FoldN folder onward; otherwise the file name only.
    """

    if not value:
        return ""
    candidate = Path(value.replace("\\", "/"))
    if not candidate.is_absolute() and not re.match(r"^[A-Za-z]:", value):
        logical = logical_parts(value, root)
        return logical.as_posix() if logical else candidate.as_posix().lstrip("./")
    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        logical = logical_parts(value, root)
        return logical.as_posix() if logical else (candidate.name or "<unnamed>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanitize RLDD metadata paths")
    parser.add_argument("--metadata", required=True, type=Path, help="Original RLDD metadata CSV")
    parser.add_argument("--output", required=True, type=Path, help="Sanitized CSV output")
    parser.add_argument("--root", type=Path, default=None, help="Shared artifact root for relative paths, default: metadata folder")
    args = parser.parse_args()
    root = (args.root or args.metadata.parent).resolve()

    with args.metadata.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            parser.error("metadata CSV has no header")
        missing = [field for field in ("video_path", "npz_path") if field not in reader.fieldnames]
        if missing:
            parser.error("metadata CSV is missing: " + ", ".join(missing))
        rows = list(reader)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in rows:
            row["video_path"] = safe_path(row.get("video_path", ""), root)
            row["npz_path"] = safe_path(row.get("npz_path", ""), root)
            writer.writerow(row)

    print(f"Sanitized {len(rows)} metadata rows to {args.output}")


if __name__ == "__main__":
    main()
