"""Run R4's eye-state CNN over the eye crops in the processed RLDD archives and store
P(eye closed) per frame, so the temporal model can use it as its eye cue.

    python temporal-model/add_eye_probs.py --data_dir data/processed_rldd \\
        --eye_model runs/eye_cnn/eye_cnn.onnx

* Uses R4's own loader (``cues.predictors.load_eye_predictor``), so the model sees
  exactly what it was trained on and the output is the calibrated P(closed)
  (temperature + sigmoid included).  Accepts R4's ``best.pt`` checkpoint or the
  ``.onnx`` made by ``scripts/r4/export_onnx.py`` (ONNX is usually faster on CPU).
* The 64x64 CLAHE crops already stored in the ``.npz`` files are the same crops R4's
  live pipeline produces (R4's tests check this), so no re-cropping is needed.
* The two eyes are averaged.  Frames where no face was found are written as NaN (R4's
  convention); train.py ignores every cue on those frames.
* For each ``kss_<x>_<label>.npz`` it writes ``kss_<x>_<label>.eye.npy`` next to it
  (float32, one value per frame).  The archives themselves are not modified.

Needs the ``cues/`` package from R4's branch (``r4/interface-contract`` or
``integration``).  By default it is looked up in the repo root above this folder; use
``--cues_root`` if your checkout of it is elsewhere.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import features as ft


def find_cues_root(arg):
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in ([arg] if arg else []) + [os.path.dirname(here), os.getcwd()]:
        if cand and os.path.isfile(os.path.join(cand, "cues", "predictors.py")):
            return os.path.abspath(cand)
    raise SystemExit(
        "Could not find R4's `cues/` package. Check out the branch that has it\n"
        "(r4/interface-contract or integration), put temporal-model/ inside that checkout,\n"
        "or pass --cues_root <folder that contains cues/>.")


def load_predictor(path, device, threads):
    from cues.predictors import load_eye_predictor
    if device != "cpu" and path.lower().endswith(".pt"):
        from cues.eye_model import EyeStatePredictor
        return EyeStatePredictor(path, device=device, num_threads=threads)
    return load_eye_predictor(path, num_threads=threads)


def predict_batched(pred, crops, batch):
    out = [np.asarray(pred.predict_crops(crops[i:i + batch]), dtype=np.float32).reshape(-1)
           for i in range(0, len(crops), batch)]
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def eye_probs_for_video(pred, left, right, det, stride, batch):
    """(N,) float32 P(closed): both eyes averaged, NaN where no face was detected.
    With stride > 1 only every stride-th detected frame is run through the CNN and the
    frames in between are linearly interpolated (a speed knob)."""
    n = len(det)
    det_b = det.astype(bool)
    pick = np.arange(0, n, stride)
    pick = pick[det_b[pick]]
    p = np.full(n, np.nan, dtype=np.float32)
    if pick.size == 0:
        return p
    ps = 0.5 * (predict_batched(pred, left[pick], batch) + predict_batched(pred, right[pick], batch))
    p[:] = np.interp(np.arange(n), pick, ps)
    p[~det_b] = np.nan
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="folder with rldd_metadata_index.csv (R3's --output_dir)")
    ap.add_argument("--eye_model", required=True, help="R4's best.pt checkpoint or eye_cnn.onnx")
    ap.add_argument("--cues_root", default=None, help="folder containing R4's cues/ package")
    ap.add_argument("--stride", type=int, default=1, help="run the CNN on every Nth frame and interpolate (faster)")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", default="cpu", help="cpu or cuda (cuda only for .pt checkpoints)")
    ap.add_argument("--overwrite", action="store_true", help="recompute files that already exist")
    args = ap.parse_args()

    sys.path.insert(0, find_cues_root(args.cues_root))
    pred = load_predictor(args.eye_model, args.device, args.threads)

    meta = pd.read_csv(os.path.join(args.data_dir, "rldd_metadata_index.csv"),
                       dtype={"subject_id": str, "raw_kss": str, "fold_id": str, "label": str})
    sums, counts = {}, {}
    for k, (_, row) in enumerate(meta.iterrows(), 1):
        path = ft._resolve_npz(args.data_dir, row)
        side = ft.eye_sidecar_path(path)
        if not os.path.isfile(path):
            print(f"[warn] missing {path}")
            continue
        with np.load(path) as z:
            det = z["face_detected"].astype(bool)
            if os.path.isfile(side) and not args.overwrite:
                p = np.load(side)
            else:
                p = eye_probs_for_video(pred, z["left_eye_crops"], z["right_eye_crops"], det,
                                        max(1, args.stride), args.batch_size)
                np.save(side, p)
        ok = det & np.isfinite(p)
        if ok.any():
            sums[row["label"]] = sums.get(row["label"], 0.0) + float(p[ok].sum())
            counts[row["label"]] = counts.get(row["label"], 0) + int(ok.sum())
        mean = f"{p[ok].mean():.3f}" if ok.any() else "n/a"
        print(f"[{k}/{len(meta)}] {os.path.relpath(side, args.data_dir)}  mean P(closed) = {mean}", flush=True)

    print("\nmean P(eye closed) on detected frames, per label:")
    means = {lab: sums[lab] / counts[lab] for lab in sums}
    for lab in ("alert", "low_vigilant", "fatigue"):
        if lab in means:
            print(f"  {lab:13s} {means[lab]:.3f}")
    if "alert" in means and "fatigue" in means and means["alert"] >= means["fatigue"]:
        print("\n[WARNING] alert videos do not have a lower closed-eye probability than fatigue videos.\n"
              "          Check that the right model file was used (and that its crops match RLDD's).")


if __name__ == "__main__":
    main()
