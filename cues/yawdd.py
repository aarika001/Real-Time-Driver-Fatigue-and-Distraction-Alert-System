"""YawDD helpers for R4's mouth-cue evaluation: file-name parsing, annotation
loading and event-level scoring.

YawDD (Abtahi et al., 2014) mirror-camera clips are named like
``12-MaleGlasses-Yawning.avi`` (subject, gender + eyewear, action); the
dashboard clips have one video per subject with all actions mixed. Parsing is
lenient: anything that does not match still gets a usable id.

Annotation CSV (from R2's yawn labelling), one row per yawn::

    video_id,start_s,end_s
    12-MaleGlasses-Yawning,3.2,8.9

``video_id`` is the file name without extension. Videos that are not in the
CSV are assumed to contain no yawns only if they are Normal or Talking clips;
unannotated Yawning / mixed clips are skipped (they cannot be scored).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

ACTIONS = ("normal", "talking", "yawning")
_SUBJECT_RE = re.compile(r"^(\d+)")


@dataclass(frozen=True)
class YawddVideoInfo:
    video_id: str      # file name without extension
    subject_id: str    # "<camera>/<number>-<gender><eyewear>", unique per person and camera
    action: str        # normal | talking | yawning | mixed
    eyewear: str       # none | glasses | sunglasses | unknown
    camera: str        # mirror | dash | unknown


def parse_yawdd_path(path: str | Path) -> YawddVideoInfo:
    p = Path(path)
    stem = p.stem
    low = stem.lower()
    action = next((a for a in ACTIONS if a in low), "mixed")

    if "sunglasses" in low:
        eyewear = "sunglasses"
    elif "noglasses" in low:
        eyewear = "none"
    elif "glasses" in low:
        eyewear = "glasses"
    else:
        eyewear = "unknown"

    parts_low = [s.lower() for s in p.parts]
    if any("mirror" in s for s in parts_low):
        camera = "mirror"
    elif any("dash" in s for s in parts_low):
        camera = "dash"
    else:
        camera = "unknown"

    person = re.sub(r"[-_ ]*(normal|talking|yawning)\s*$", "", stem, flags=re.IGNORECASE)
    return YawddVideoInfo(video_id=stem, subject_id=f"{camera}/{person}", action=action,
                          eyewear=eyewear, camera=camera)


def load_annotations(csv_path: str | Path) -> dict[str, list[tuple[float, float]]]:
    """{video_id: [(start_s, end_s), ...]} from the annotation CSV."""
    df = pd.read_csv(csv_path)
    required = {"video_id", "start_s", "end_s"}
    if not required <= set(df.columns):
        raise ValueError(f"{csv_path} must have columns {sorted(required)}, got {list(df.columns)}")
    bad = df[df["end_s"] <= df["start_s"]]
    if len(bad):
        raise ValueError(f"{len(bad)} annotation rows have end_s <= start_s")
    out: dict[str, list[tuple[float, float]]] = {}
    for row in df.itertuples(index=False):
        out.setdefault(str(row.video_id), []).append((float(row.start_s), float(row.end_s)))
    return {k: sorted(v) for k, v in out.items()}


def ground_truth_for(info: YawddVideoInfo,
                     annotations: dict[str, list[tuple[float, float]]]) -> Optional[list[tuple[float, float]]]:
    """Yawn intervals for a video, [] for unannotated Normal/Talking, None if unscorable."""
    if info.video_id in annotations:
        return annotations[info.video_id]
    if info.action in ("normal", "talking"):
        return []
    return None


@dataclass
class EventScore:
    n_gt: int = 0                 # annotated yawns
    n_hit: int = 0                # annotated yawns that were detected
    n_false: int = 0              # detections that match no annotated yawn
    delays_s: tuple = ()          # detection delay per hit (detect time - yawn start)

    def __add__(self, other: "EventScore") -> "EventScore":
        return EventScore(self.n_gt + other.n_gt, self.n_hit + other.n_hit,
                          self.n_false + other.n_false, self.delays_s + other.delays_s)


def score_events(detected: Iterable[tuple[float, float, float]],
                 gt: list[tuple[float, float]], tolerance_s: float = 1.0) -> EventScore:
    """Event-level matching.

    A ground-truth yawn is a hit if any detection overlaps it (widened by
    ``tolerance_s`` on both sides). A detection overlapping no yawn is a false
    alarm. One detection may cover two yawns that run together.
    """
    detected = list(detected)
    hit_delays = []
    used = [False] * len(detected)
    for g0, g1 in gt:
        lo, hi = g0 - tolerance_s, g1 + tolerance_s
        match = [i for i, (d0, d1, _) in enumerate(detected) if d0 < hi and d1 > lo]
        if match:
            for i in match:
                used[i] = True
            hit_delays.append(max(0.0, detected[match[0]][2] - g0))
    return EventScore(n_gt=len(gt), n_hit=len(hit_delays),
                      n_false=used.count(False), delays_s=tuple(hit_delays))
