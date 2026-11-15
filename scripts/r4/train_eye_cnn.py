"""Train the MobileNetV2 eye-state CNN (R4, Milestone 4).

Steps
-----
1. Index MRL Eye (+ optional extra labelled crops via --extra_index CSVs).
2. Split by PERSON into train / val / test (saved to <out_dir>/splits.csv).
3. Load all crops into RAM as 64x64 uint8.
4. Fine-tune ImageNet MobileNetV2: head-only warm-up epoch, then the whole
   network; AdamW + warm-up + cosine LR; class-balanced BCE (pos_weight);
   mixed precision on GPU; early stopping on validation AUROC.
5. Fit temperature scaling on the validation people -> calibrated P(closed).
6. Evaluate on the unseen test people: overall, per condition (glasses,
   reflection, lighting, sensor), per person, bootstrap CIs over people.

Outputs in --out_dir: best.pt, last.pt, history.csv, splits.csv,
test_predictions.csv, test_report.md, test_metrics.json.

Examples:
    # 2-minute check that everything runs (tiny subset, no pretrained download)
    python scripts/r4/train_eye_cnn.py --mrl_root /kaggle/input/mrl-eye --out_dir runs/smoke --smoke
    # full training
    python scripts/r4/train_eye_cnn.py --mrl_root /kaggle/input/mrl-eye --out_dir runs/eye_cnn
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cues.augment import AugmentConfig  # noqa: E402
from cues.config import load_config  # noqa: E402
from cues.eye_dataset import (TAG_COLUMNS, EyeCropDataset, build_mrl_index, load_csv_index,  # noqa: E402
                              load_raw_crops, split_by_subject)
from cues.eye_metrics import binary_metrics, summarize_predictions, to_markdown, worst_condition  # noqa: E402
from cues.eye_model import EyeModelConfig, EyeNet, fit_temperature, load_checkpoint, save_checkpoint  # noqa: E402


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    freeze_epochs: int = 1
    patience: int = 4
    val_subjects: int = 5
    test_subjects: int = 7
    seed: int = 42

    @classmethod
    def from_dict(cls, d) -> "TrainConfig":
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise KeyError(f"Unknown eye_train keys: {sorted(unknown)}")
        return cls(**d)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):           # older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.inference_mode()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, idx = [], []
    for x, _, i in loader:
        logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        idx.append(i)
    return torch.cat(logits).numpy(), torch.cat(idx).numpy()


def subject_mean_f1(p: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> float:
    vals = [binary_metrics(p[subjects == s], y[subjects == s])["f1_closed"] for s in np.unique(subjects)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def build_index(args) -> pd.DataFrame:
    frames = [build_mrl_index(args.mrl_root)]
    for csv in args.extra_index or []:
        frames.append(load_csv_index(csv))
    df = pd.concat(frames, ignore_index=True)
    for col in TAG_COLUMNS:
        df[col] = df[col].fillna("unknown") if col in df else "unknown"
    if args.smoke:   # 6 people x 40 images: enough to exercise every code path in ~1 minute
        keep = sorted(df["subject_id"].unique())[:6]
        df = (df[df["subject_id"].isin(keep)].sample(frac=1.0, random_state=0)
              .groupby("subject_id").head(40).reset_index(drop=True))
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mrl_root", required=True, help="MRL Eye folder or mrlEyes_2018_01.zip")
    ap.add_argument("--extra_index", nargs="*", default=[], help="CSV indexes of extra labelled crops")
    ap.add_argument("--out_dir", default="runs/eye_cnn")
    ap.add_argument("--config", default=None, help="YAML config (default configs/r4.yaml)")
    ap.add_argument("--epochs", type=int, default=None, help="override eye_train.epochs")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to check the pipeline")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    tc = TrainConfig.from_dict(cfg.get("eye_train"))
    mc = EyeModelConfig.from_dict(cfg.get("eye_model"))
    ac = AugmentConfig.from_dict(cfg.get("eye_augment"))
    if args.epochs is not None:
        tc.epochs = args.epochs
    if args.batch_size is not None:
        tc.batch_size = args.batch_size
    if args.smoke:
        tc.epochs, tc.batch_size, tc.patience = 2, 16, 2
        mc.pretrained = False
    seed_everything(tc.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = device.type == "cuda"
    workers = args.num_workers if args.num_workers is not None else (0 if platform.system() == "Windows"
                                                                     else min(4, os.cpu_count() or 1))
    print(f"Device: {device} | AMP: {use_amp} | workers: {workers} | torch {torch.__version__}")

    # ---- data ------------------------------------------------------------------
    t0 = time.time()
    df = build_index(args)
    df["split"] = split_by_subject(df, tc.val_subjects, tc.test_subjects, tc.seed)
    df.to_csv(out / "splits.csv", index=False)
    for name, part in df.groupby("split"):
        print(f"  {name:5s}: {len(part):6d} images, {part['subject_id'].nunique():3d} people, "
              f"{part['label_closed'].mean():.1%} closed")
    raw = load_raw_crops(df)
    print(f"Loaded {len(df)} crops in {time.time() - t0:.0f}s")

    idx = {s: np.flatnonzero(df["split"].to_numpy() == s) for s in ("train", "val", "test")}
    labels = df["label_closed"].to_numpy()
    subjects = df["subject_id"].to_numpy()
    ds = {s: EyeCropDataset(raw[idx[s]], labels[idx[s]], ac if s == "train" else None, tc.seed)
          for s in idx}
    pin = device.type == "cuda"
    train_loader = DataLoader(ds["train"], batch_size=tc.batch_size, shuffle=True, drop_last=len(idx["train"]) > tc.batch_size,
                              num_workers=workers, pin_memory=pin)
    eval_loaders = {s: DataLoader(ds[s], batch_size=tc.batch_size * 2, shuffle=False,
                                  num_workers=workers, pin_memory=pin) for s in ("val", "test")}

    # ---- model, loss, optimiser ------------------------------------------------
    model = EyeNet(mc).to(device)
    n_pos = int(labels[idx["train"]].sum())
    n_neg = len(idx["train"]) - n_pos
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = tc.epochs * steps_per_epoch
    warmup = max(1, tc.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = make_scaler(use_amp)
    print(f"Train: {len(idx['train'])} images, pos_weight={pos_weight.item():.3f}, "
          f"{steps_per_epoch} steps/epoch, {tc.epochs} epochs")

    # ---- training loop ---------------------------------------------------------
    history, best_auc, best_f1, bad_epochs = [], -1.0, -1.0, 0
    for epoch in range(tc.epochs):
        frozen = epoch < tc.freeze_epochs
        model.set_backbone_trainable(not frozen)
        model.train()
        if frozen:
            model.features.eval()       # keep pretrained BatchNorm statistics while the head warms up
        ds["train"].set_epoch(epoch)
        t_ep, run_loss, n_seen = time.time(), 0.0, 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss += float(loss.item()) * len(y)
            n_seen += len(y)

        z_val, i_val = predict_logits(model, eval_loaders["val"], device)
        p_val = sigmoid(z_val)
        y_val = labels[idx["val"]][i_val]
        m = binary_metrics(p_val, y_val)
        s_f1 = subject_mean_f1(p_val, y_val, subjects[idx["val"]][i_val])
        row = {"epoch": epoch + 1, "train_loss": run_loss / max(1, n_seen), "val_auroc": m["auroc"],
               "val_f1": m["f1_closed"], "val_subject_f1": s_f1, "val_acc": m["accuracy"],
               "lr": optimizer.param_groups[0]["lr"], "frozen_backbone": frozen,
               "seconds": round(time.time() - t_ep, 1)}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"Epoch {epoch + 1:2d}/{tc.epochs} loss {row['train_loss']:.4f} | val AUROC {m['auroc']:.4f} "
              f"F1 {m['f1_closed']:.4f} subject-F1 {s_f1:.4f} acc {m['accuracy']:.4f} | {row['seconds']}s")

        extra = {"epoch": epoch + 1, "val": m, "train_config": asdict(tc), "augment": ac.to_dict()}
        save_checkpoint(out / "last.pt", model, extra)
        auc = m["auroc"] if np.isfinite(m["auroc"]) else m["accuracy"]
        if auc > best_auc or (auc == best_auc and m["f1_closed"] > best_f1):
            best_auc, best_f1, bad_epochs = auc, m["f1_closed"], 0
            save_checkpoint(out / "best.pt", model, extra)
            print("  -> new best, saved best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= tc.patience:
                print(f"Early stopping: no validation improvement for {tc.patience} epochs")
                break

    # ---- temperature scaling on validation people --------------------------------
    model, ckpt = load_checkpoint(out / "best.pt")
    model.to(device)
    z_val, i_val = predict_logits(model, eval_loaders["val"], device)
    y_val = labels[idx["val"]][i_val]
    temperature = fit_temperature(z_val, y_val)
    ece_before = binary_metrics(sigmoid(z_val), y_val)["ece"]
    ece_after = binary_metrics(sigmoid(z_val / temperature), y_val)["ece"]
    model.temperature.fill_(temperature)
    extra = dict(ckpt["extra"], temperature_fit={"T": temperature, "val_ece_before": ece_before,
                                                 "val_ece_after": ece_after})
    save_checkpoint(out / "best.pt", model, extra)
    print(f"Temperature T={temperature:.3f}: validation ECE {ece_before:.4f} -> {ece_after:.4f}")

    # ---- final test on unseen people ---------------------------------------------
    z_te, i_te = predict_logits(model, eval_loaders["test"], device)
    test_df = df.iloc[idx["test"][i_te]].copy()
    test_df["p_closed"] = sigmoid(z_te / temperature)
    keep = ["dataset", "subject_id", "path", "label_closed", "p_closed", *TAG_COLUMNS]
    test_df[keep].to_csv(out / "test_predictions.csv", index=False)
    summary = summarize_predictions(test_df, group_cols=["dataset", *TAG_COLUMNS],
                                    n_boot=200 if args.smoke else 1000)
    worst = worst_condition(summary)
    report = ["# Eye-state CNN: test on unseen people", "",
              f"Best epoch {ckpt['extra'].get('epoch')} | temperature {temperature:.3f} | "
              f"{test_df['subject_id'].nunique()} test people, {len(test_df)} crops", "",
              to_markdown(summary), ""]
    if worst is not None:
        report.append(f"Worst condition (>= 5 people): {worst['group']} "
                      f"(recall_closed {worst['recall_closed']:.3f})")
    (out / "test_report.md").write_text("\n".join(report), encoding="utf-8")
    overall = {k: (v.item() if hasattr(v, "item") else v) for k, v in summary.iloc[0].to_dict().items()}
    overall = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in overall.items()}
    overall["temperature"] = temperature
    (out / "test_metrics.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    summary.to_csv(out / "test_summary.csv", index=False)
    print("\n".join(report))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Outputs in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
