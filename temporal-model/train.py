"""Train and evaluate the temporal (GRU) model on preprocessed UTA-RLDD.

    python train.py --data_dir ../processed_rldd --out_dir runs/temporal

Protocol (no subject ever appears in two of train / val / test):
  * buckets = official folds Fold1..Fold5 (or --cv subject for GroupKFold-style buckets)
  * for each bucket i: test = i, val = i+1 (early stopping + temperature scaling),
    train = remaining buckets
  * every subject is tested exactly once; pooled metrics use those test predictions
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import features as ft
from metrics import summarize
from model import TemporalGRU, cue_dropout, fit_temperature


def parse_args():
    p = argparse.ArgumentParser(description="Temporal GRU model on preprocessed UTA-RLDD")
    p.add_argument("--data_dir", required=True, help="--output_dir of RLDD-preprocess (holds rldd_metadata_index.csv)")
    p.add_argument("--out_dir", default="runs/temporal")
    p.add_argument("--label_mode", default="binary", choices=list(ft.LABEL_MODES),
                   help="binary: alert vs fatigue (KSS5 dropped) | binary_lv_fatigue: KSS5 counted as fatigue | three")
    p.add_argument("--cv", default="fold", choices=["fold", "subject"], help="bucket subjects by official fold or randomly")
    p.add_argument("--n_splits", type=int, default=5, help="buckets for --cv subject")
    p.add_argument("--only", nargs="*", default=None, help="run only these test buckets, e.g. Fold2 Fold4")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--cue_dropout", type=float, default=0.3, help="P(hide a cue group) per window; 0 = off")
    p.add_argument("--stride", type=int, default=ft.STRIDE)
    p.add_argument("--min_face_rate", type=float, default=0.2, help="drop windows where the face is found in fewer frames")
    p.add_argument("--no_calibration", action="store_true", help="ablation: one pooled baseline instead of per-driver")
    p.add_argument("--no_reliability", action="store_true", help="ablation: hide quality/face-detected channels")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p.parse_args()


# ----------------------------------------------------------------------------- helpers
def make_set(by_subject, subjects, base_fn, label_map, args):
    recs, feats, dets = [], [], []
    for s in subjects:
        for r in by_subject[s]:
            feats.append(ft.frame_features(r, base_fn(s), use_reliability=not args.no_reliability))
            dets.append(r.det)
            recs.append(r)
    return ft.WindowSet(feats, dets, recs, label_map, stride=args.stride, min_face_rate=args.min_face_rate)


@torch.no_grad()
def predict_logits(model, ws, device, bs=1024):
    model.eval()
    out = []
    for i in range(0, len(ws), bs):
        X, _ = ws.batch(np.arange(i, min(i + bs, len(ws))))
        out.append(model(torch.from_numpy(X).to(device)).cpu())
    return torch.cat(out) if out else torch.zeros(0, 1)


def train_fold(train_ws, val_ws, groups, n_classes, args, device, seed):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = TemporalGRU(hidden=args.hidden, n_classes=n_classes, dropout=args.dropout).to(device)
    counts = np.bincount(train_ws.y, minlength=n_classes)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    y_val = torch.from_numpy(val_ws.y)

    best, best_state, bad, history = float("inf"), None, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(len(train_ws))
        tot, nb = 0.0, 0
        for i in range(0, len(perm), args.batch_size):
            X, y = train_ws.batch(perm[i:i + args.batch_size])
            X = cue_dropout(torch.from_numpy(X).to(device), args.cue_dropout, groups)
            loss = loss_fn(model(X), torch.from_numpy(y).to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        val_logits = predict_logits(model, val_ws, device)
        val_loss = float(F.cross_entropy(val_logits, y_val))
        val_acc = float((val_logits.argmax(1) == y_val).float().mean())
        history.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_nll": val_loss, "val_acc": val_acc})
        print(f"    epoch {ep:02d}  train {tot / max(nb, 1):.4f}  val_nll {val_loss:.4f}  val_acc {val_acc:.3f}", flush=True)
        if val_loss < best - 1e-4:
            best, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("    early stop")
                break
    model.load_state_dict(best_state)
    return model, history


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    os.makedirs(args.out_dir, exist_ok=True)
    label_map = ft.LABEL_MODES[args.label_mode]
    class_names = ft.CLASS_NAMES[args.label_mode]
    n_classes = len(class_names)

    recs = ft.load_videos(args.data_dir, args.label_mode)
    calib, dropped = ft.build_subject_calibration(recs)
    for s, why in dropped:
        print(f"[warn] subject {s} dropped: {why}")
    recs = [r for r in recs if r.subject in calib]
    by_subject = {}
    for r in recs:
        by_subject.setdefault(r.subject, []).append(r)
    groups = ft.present_groups(recs)
    print(f"{len(by_subject)} subjects, {len(recs)} videos | cue groups: {[ft.GROUP_NAMES[g] for g in groups]} "
          f"| label_mode={args.label_mode} | device={device}")
    if 0 not in groups:
        print("[note] no eye_closed_prob in the .npz files -> model uses mouth + head + quality cues only")

    buckets = ft.assign_buckets(recs, args.cv, args.n_splits, args.seed)
    per_subject_base = ft.baselines_per_subject(calib)

    fold_results, all_pred = {}, []
    for split in ft.rotating_splits(buckets):
        if args.only and split["name"] not in args.only:
            continue
        t0 = time.time()
        print(f"\n=== test {split['name']} | train {len(split['train'])} subj, val {len(split['val'])}, "
              f"test {len(split['test'])} ===")
        if args.no_calibration:
            pooled = ft.baseline_pooled(calib, split["train"])
            base_fn = lambda s, b=pooled: b
        else:
            base_fn = lambda s: per_subject_base[s]
        tr = make_set(by_subject, split["train"], base_fn, label_map, args)
        va = make_set(by_subject, split["val"], base_fn, label_map, args)
        te = make_set(by_subject, split["test"], base_fn, label_map, args)
        print(f"  windows: train {len(tr)}  val {len(va)}  test {len(te)}  "
              f"| train class counts {np.bincount(tr.y, minlength=n_classes).tolist()}")
        if min(len(tr), len(va), len(te)) == 0:
            print("  [skip] empty split")
            continue

        model, hist = train_fold(tr, va, groups, n_classes, args, device, args.seed + len(fold_results))
        val_logits = predict_logits(model, va, device)
        T = fit_temperature(val_logits, torch.from_numpy(va.y))
        te_logits = predict_logits(model, te, device)
        p_raw = F.softmax(te_logits, 1).numpy()
        p_cal = F.softmax(te_logits / T, 1).numpy()
        raw, cal = summarize(p_raw, te.y, te.subject, class_names), summarize(p_cal, te.y, te.subject, class_names)
        print(f"  T={T:.2f} | test acc {cal['accuracy']:.3f}  bal_acc {cal['balanced_accuracy']:.3f}  "
              f"macro_f1 {cal['macro_f1']:.3f}  ECE {raw['ece']:.3f} -> {cal['ece']:.3f}  ({time.time() - t0:.0f}s)")
        fold_results[split["name"]] = {"temperature": T, "raw": raw, "calibrated": cal, "history": hist,
                                       "n_train_subjects": len(split["train"])}

        idx = np.arange(len(te))
        df = pd.DataFrame({
            "test_bucket": split["name"],
            "subject": te.subject,
            "video": [te.recs[v].name for v in te.vid],
            "start_frame": te.local_start(idx),
            "y": te.y,
        })
        for k, n in enumerate(class_names):
            df[f"p_{n}"] = p_cal[:, k]
        all_pred.append(df)

        torch.save({
            "state_dict": model.state_dict(), "temperature": T, "class_names": class_names,
            "hidden": args.hidden, "n_classes": n_classes, "groups": groups,
            "layout": {"cues": ft.CUE_NAMES, "reliability": ft.REL_NAMES, "flags": ft.GROUP_NAMES},
            "window": ft.WINDOW, "fps": ft.FPS, "args": vars(args),
            "pooled_baseline": ({"mar_mu": pooled.mar_mu, "mar_sd": pooled.mar_sd,
                                 "pose_mu": pooled.pose_mu.tolist(), "pose_sd": pooled.pose_sd.tolist()}
                                if args.no_calibration else None),
        }, os.path.join(args.out_dir, f"model_{split['name']}.pt"))

    if not all_pred:
        raise SystemExit("nothing was evaluated")

    pred = pd.concat(all_pred, ignore_index=True)
    pred.to_csv(os.path.join(args.out_dir, "predictions_windows.csv"), index=False)
    pcols = [f"p_{n}" for n in class_names]
    probs, y, subj = pred[pcols].to_numpy(), pred["y"].to_numpy(), pred["subject"].to_numpy()

    # video-level: average the window probabilities of each test video
    vid = pred.groupby(["subject", "video"]).agg({**{c: "mean" for c in pcols}, "y": "first"}).reset_index()
    vid["pred"] = vid[pcols].to_numpy().argmax(1)
    vid.to_csv(os.path.join(args.out_dir, "predictions_videos.csv"), index=False)

    pooled_res = summarize(probs, y, subj, class_names)
    pooled_res["video_level_accuracy"] = float((vid["pred"] == vid["y"]).mean())
    pooled_res["n_videos"] = int(len(vid))
    out = {"args": vars(args), "class_names": class_names, "cue_groups": [ft.GROUP_NAMES[g] for g in groups],
           "pooled_test_calibrated": pooled_res, "folds": fold_results}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    ps = pooled_res["per_subject_accuracy"]
    print("\n================ pooled test results (calibrated) ================")
    print(f"windows {pooled_res['n_windows']} | accuracy {pooled_res['accuracy']:.3f} "
          f"(95% CI over subjects {pooled_res['accuracy_ci95_over_subjects'][0]:.3f}-{pooled_res['accuracy_ci95_over_subjects'][1]:.3f})")
    print(f"balanced acc {pooled_res['balanced_accuracy']:.3f} | macro-F1 {pooled_res['macro_f1']:.3f} | "
          f"ECE {pooled_res['ece']:.3f}" + (f" | AUC {pooled_res['roc_auc']:.3f}" if "roc_auc" in pooled_res else ""))
    print(f"per-subject accuracy min/median/max: {ps['min']:.3f} / {ps['median']:.3f} / {ps['max']:.3f} over {ps['n_subjects']} subjects")
    print(f"video-level accuracy {pooled_res['video_level_accuracy']:.3f} over {pooled_res['n_videos']} videos")
    print(f"saved to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
