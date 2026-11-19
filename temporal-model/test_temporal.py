"""Sanity tests.  Run from this folder:  python test_temporal.py   (or: pytest -q)"""
import numpy as np

import features as ft


def _rec(subject, label, n=700, fold="Fold1", pitch=178.0, seed=0, eye=False):
    rng = np.random.RandomState(seed)
    pose = np.stack([pitch + rng.randn(n), rng.randn(n), rng.randn(n)], 1).astype(np.float32)
    pose[:, 0] = ((pose[:, 0] + 180) % 360) - 180          # wraps to about -179 / +179
    return ft.VideoRec(subject=subject, fold=fold, kss="0", label=label,
                       mar=(0.05 + 0.01 * rng.randn(n)).astype(np.float32), pose=pose,
                       quality=np.tile([0.5, 0.2, 0.01, 300.0], (n, 1)).astype(np.float32),
                       det=np.ones(n, np.uint8), eye=(rng.rand(n).astype(np.float32) if eye else None),
                       name=f"{subject}_{label}")


def test_wrap_deg():
    assert abs(float(ft.wrap_deg(179.0 - (-179.0))) - (-2.0)) < 1e-4
    assert abs(float(ft.wrap_deg(-179.0 - 179.0)) - 2.0) < 1e-4


def test_pitch_near_180_gives_small_z():
    r = _rec("a", "alert", pitch=179.0)
    b = ft.fit_baseline(r.mar[:450], r.pose[:450], r.det[:450])
    X = ft.frame_features(r, b)
    assert np.abs(X[:, 2]).max() < Z_LIMIT, "pitch wrap-around leaked into the z-score"
    assert X.shape[1] == ft.N_FEATURES


Z_LIMIT = 6.0


def test_no_face_means_no_cue():
    r = _rec("a", "alert")
    r.det[10:20] = 0
    b = ft.fit_baseline(r.mar, r.pose, r.det)
    X = ft.frame_features(r, b)
    assert np.all(X[10:20, :ft.N_CUES] == 0) and np.all(X[10:20, ft.REL_START + 4] == 0)


def test_windows_stay_inside_video_and_after_calibration():
    recs = [_rec("a", "alert", n=700, seed=1), _rec("b", "fatigue", n=620, seed=2)]
    base = ft.fit_baseline(recs[0].mar, recs[0].pose, recs[0].det)
    feats = [ft.frame_features(r, base) for r in recs]
    ws = ft.WindowSet(feats, [r.det for r in recs], recs, {"alert": 0, "fatigue": 1})
    assert len(ws) > 0
    idx = np.arange(len(ws))
    local = ws.local_start(idx)
    lens = np.array([recs[v].n for v in ws.vid])
    assert local.min() >= ft.CALIB_FRAMES
    assert (local + ft.WINDOW <= lens).all()
    X, y = ws.batch(idx[:5])
    assert X.shape == (5, ft.WINDOW, ft.N_FEATURES)


def test_splits_have_no_subject_overlap_and_cover_everyone():
    recs = [_rec(f"Fold{f}_{s}", "alert", fold=f"Fold{f}", seed=f * 10 + s) for f in range(1, 6) for s in range(3)]
    buckets = ft.assign_buckets(recs, "fold")
    tested = []
    for sp in ft.rotating_splits(buckets):
        tr, va, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
        assert not (tr & va) and not (tr & te) and not (va & te)
        assert tr and va and te
        tested += sp["test"]
    assert sorted(tested) == sorted({r.subject for r in recs})


def test_cue_dropout_and_model_shapes():
    import torch
    from model import TemporalGRU, cue_dropout
    x = torch.randn(64, ft.WINDOW, ft.N_FEATURES)
    x[:, :, ft.FLAG_START:] = 1.0
    out = cue_dropout(x, 0.9, [1, 2])
    assert out is not x and torch.equal(x[:, :, ft.FLAG_START:], torch.ones(64, ft.WINDOW, 3)), "input must not be modified"
    f_mouth, f_head = out[:, 0, ft.FLAG_START + 1], out[:, 0, ft.FLAG_START + 2]
    assert ((f_mouth + f_head) >= 1).all(), "at least one cue group must always survive"
    assert (out[f_mouth == 0][:, :, 1] == 0).all()                      # dropped mouth cue is zeroed
    assert (out[f_head == 0][:, :, 2:5] == 0).all()                     # dropped head cues are zeroed
    assert cue_dropout(x, 0.0, [1, 2]) is x
    m = TemporalGRU(n_classes=3)
    assert m(x).shape == (64, 3)
    # streaming one frame at a time gives the same answer as the full window
    m.eval()
    full = m(x[:2])
    h = None
    for t in range(ft.WINDOW):
        step, h = m(x[:2, t:t + 1], h0=h, return_hidden=True)
    assert torch.allclose(full, step, atol=1e-5)


def test_eye_sidecar_is_loaded_and_length_checked(tmp_path=None):
    import os, tempfile
    import pandas as pd
    root = tmp_path or tempfile.mkdtemp()
    d = os.path.join(root, "Fold1", "subject_01")
    os.makedirs(d)
    n = 100
    npz = os.path.join(d, "kss_0_alert.npz")
    np.savez(npz, mar=np.zeros(n, np.float32), head_pose=np.zeros((n, 3), np.float32),
             quality=np.zeros((n, 4), np.float32), face_detected=np.ones(n, np.uint8))
    pd.DataFrame([{"fold_id": "Fold1", "subject_id": "01", "raw_kss": "0", "label": "alert",
                   "npz_path": npz}]).to_csv(os.path.join(root, "rldd_metadata_index.csv"), index=False)
    assert ft.load_videos(root)[0].eye is None
    np.save(ft.eye_sidecar_path(npz), np.full(n, 0.3, np.float32))
    r = ft.load_videos(root)[0]
    assert r.eye is not None and abs(float(r.eye[0]) - 0.3) < 1e-6
    assert ft.present_groups([r]) == [0, 1, 2]
    X = ft.frame_features(r, ft.fit_baseline(r.mar, r.pose, r.det))
    assert np.allclose(X[:, 0], 0.3) and np.all(X[:, ft.FLAG_START] == 1)
    np.save(ft.eye_sidecar_path(npz), np.zeros(n + 5, np.float32))      # wrong length must fail loudly
    try:
        ft.load_videos(root)
        raise AssertionError("length mismatch was not detected")
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
