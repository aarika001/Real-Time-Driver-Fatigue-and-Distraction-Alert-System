"""Tests for scripts/r4/eval_dmd.py on a synthetic DMD-like session and stress set.

A stub landmarker reads the "mouth state" from a marker patch in the frame and a
fake eye predictor separates the synthetic open/closed eyes, so the whole
pipeline (video -> cues -> eye labels -> yawn events -> report) runs without
MediaPipe faces or PyTorch.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from cues.inference_r4 import R4Pipeline
from tests.r4.test_eval_and_live import FakePredictor, StubLandmarker, _face_frame

REPO_ROOT = Path(__file__).resolve().parents[2]
FPS = 744 / 25


def _script():
    spec = importlib.util.spec_from_file_location("eval_dmd", REPO_ROOT / "scripts" / "r4" / "eval_dmd.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class MarkerLandmarker(StubLandmarker):
    """Mouth wide open when the top-left marker patch is bright; no face when it is mid-grey."""

    def __init__(self):
        super().__init__([0.0])

    def detect(self, rgb, ts_ms):
        m = rgb[:16, :16].mean()
        if 90 < m < 170:
            return None
        self.seq = [40.0 if m >= 170 else 0.0]
        self.i = 0
        return super().detect(rgb, ts_ms)


def _frame(closed: bool, yawn: bool = False, lost: bool = False) -> np.ndarray:
    f = _face_frame(closed)
    f[:16, :16] = 128 if lost else (255 if yawn else 0)
    return f


def _make_session(root: Path, sid: str, seconds: float = 60.0) -> pd.DataFrame:
    """Open eyes, a closed-eye interval at 40-44 s, yawns at 45-50 s and 52-56 s (second one 'with hand')."""
    path = root / f"{sid}_face.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 240))
    if not vw.isOpened():
        pytest.skip("cannot write mp4 here")
    n = int(seconds * FPS)
    for i in range(n):
        t = i / FPS
        vw.write(_frame(closed=40 <= t < 44, yawn=(45 <= t < 50) or (52 <= t < 56), lost=58 <= t < 58.2))
    vw.release()
    f = lambda s: int(round(s * FPS))   # noqa: E731
    rows = [("eyes_state/open", 0, f(40) - 1), ("eyes_state/close", f(40), f(44) - 1),
            ("eyes_state/open", f(44), n - 1), ("yawning/Yawning without hand", f(45), f(50) - 1),
            ("yawning/Yawning with hand", f(52), f(56) - 1)]
    return pd.DataFrame([{"session_id": sid, "subject_id": "1", "kaggle_split": "test", "source_task": "drowsiness",
                          "annotation_file": "drowsiness", "native_label": lab, "start_frame": s, "end_frame": e}
                         for lab, s, e in rows])


def _factory(mode="video"):
    return R4Pipeline(eye_predictor=FakePredictor(), landmarker=MarkerLandmarker())


def test_frame_eye_labels_and_find_videos(tmp_path):
    ev_mod = _script()
    ev = pd.DataFrame({"native_label": ["eyes_state/open", "eyes_state/close", "eyes_state/closing"],
                       "start_frame": [0, 10, 20], "end_frame": [9, 19, 29]})
    lab = ev_mod.frame_eye_labels(np.arange(35), ev)
    assert (lab[:10] == 0).all() and (lab[10:20] == 1).all() and np.isnan(lab[20:]).all()
    (tmp_path / "a" / "gA_1_s5_x_face.mp4").parent.mkdir()
    (tmp_path / "a" / "gA_1_s5_x_face.mp4").write_bytes(b"")
    assert set(ev_mod.find_face_videos(tmp_path)) == {"gA_1_s5_x"}


def test_eval_dmd_end_to_end_on_synthetic_session(tmp_path):
    vids = tmp_path / "dmd_video_v1"
    vids.mkdir()
    events = _make_session(vids, "gA_1_s5_test")
    events.to_csv(tmp_path / "events.csv", index=False)

    stress_root = tmp_path / "NeuroNova_data"
    man_rows = []
    for k, (state, closed, yawn) in enumerate((("eyes_closed", True, False), ("eyes_open_no_yawn", False, False),
                                                ("yawn", False, True))):
        d = stress_root / "generated_stress" / "dmd" / "subject_01" / f"f{k}_{state}"
        d.mkdir(parents=True)
        cv2.imwrite(str(d / "original.png"), _frame(closed, yawn))
        for st, sev, img in (("glare", "mild", _frame(closed, yawn)), ("face_lost", "single_level", _frame(closed, lost=True))):
            cv2.imwrite(str(d / f"{st}_{sev}.png"), img)
            man_rows.append({"source_person_id": 1, "source_state": state, "stress_type": st, "severity": sev,
                             "output_path": str((d / f"{st}_{sev}.png").relative_to(stress_root)).replace("\\", "/")})
    pd.DataFrame(man_rows).to_csv(tmp_path / "manifest.csv", index=False)

    out = tmp_path / "out"
    mod = _script()
    args = ["--videos_dir", str(vids), "--events", str(tmp_path / "events.csv"), "--stress_root", str(stress_root),
            "--stress_manifest", str(tmp_path / "manifest.csv"), "--out_dir", str(out), "--calib_s", "30",
            "--glasses_subjects", "1"]
    assert mod.main(args, pipeline_factory=_factory) == 0

    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Eye closure, frame level" in report and "Yawns, event level" in report and "Stress set" in report
    eye = pd.read_csv(out / "eye_summary.csv")
    assert eye.iloc[0]["recall_closed"] > 0.9 and eye.iloc[0]["accuracy"] > 0.9
    assert "glasses=yes" in set(eye["group"])
    ys = pd.read_csv(out / "yawn_per_session.csv")
    assert ys.loc[0, "n_yawns"] == 2 and ys.loc[0, "n_hit"] == 2 and ys.loc[0, "false_alarms"] == 0
    st = pd.read_csv(out / "stress_summary.csv").set_index("stress_type")
    assert st.loc["face_lost", "face_rate_stressed"] == 0.0 and st.loc["glare", "eye_acc_stressed"] == 1.0
    assert "indicative" in st.loc["glare", "flag"]

    # second run uses the per-session cache and gives the same numbers
    assert mod.main(args, pipeline_factory=_factory) == 0
    assert pd.read_csv(out / "yawn_per_session.csv").loc[0, "n_hit"] == 2


def test_yawn_events_file_replaces_raw_yawn_phases(tmp_path):
    vids = tmp_path / "v"
    vids.mkdir()
    ev = _make_session(vids, "gA_1_s5_test", seconds=60)
    raw = pd.concat([ev, ev[ev["native_label"].str.startswith("yawning/")]], ignore_index=True)  # duplicated phases
    raw.to_csv(tmp_path / "all.csv", index=False)
    merged = ev[ev["native_label"].str.startswith("yawning/")].copy()
    merged.to_csv(tmp_path / "yawns.csv", index=False)
    out = tmp_path / "o"
    assert _script().main(["--videos_dir", str(vids), "--events", str(tmp_path / "all.csv"),
                           "--yawn_events", str(tmp_path / "yawns.csv"), "--out_dir", str(out)],
                          pipeline_factory=_factory) == 0
    assert pd.read_csv(out / "yawn_per_session.csv").loc[0, "n_yawns"] == 2      # not 4
