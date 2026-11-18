
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import joblib
from tqdm import tqdm

from .common import (
    LABELS,
    LABEL_TO_ID,
    load_metadata,
    load_baselines,
    extract_feature_matrix,
    make_windows,
    save_json,
)


class WindowDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class TemporalGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=3):
        super().__init__()

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(0.25)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(self.dropout(h[-1]))


def collect_windows(metadata, baselines, window, stride):
    all_x, all_y, all_groups, names = [], [], [], None
    skipped = []

    for _, rec in tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Building sequences",
    ):
        sid = str(rec.subject_id)

        if sid not in baselines.index:
            skipped.append({
                "subject_id": sid,
                "npz_path": rec.npz_path,
                "reason": "no_neutral_baseline",
            })
            continue

        try:
            feats, feat_names, detected = extract_feature_matrix(
                rec.npz_path,
                baselines.loc[sid],
            )

            if names is None:
                names = feat_names
            elif names != feat_names:
                raise ValueError(
                    "Inconsistent optional feature arrays across NPZ "
                    "files. Ensure upstream R4/R7 fields exist "
                    "consistently."
                )

            wins = make_windows(feats, window, stride)

            if len(wins) == 0:
                continue

            starts = range(
                0,
                len(feats) - window + 1,
                stride,
            )

            keep = []

            for j, s in enumerate(starts):
                if np.mean(detected[s:s + window]) >= 0.60:
                    keep.append(j)

            if not keep:
                continue

            wins = wins[keep]

            all_x.append(wins)
            all_y.extend(
                [LABEL_TO_ID[str(rec.label)]] * len(wins)
            )
            all_groups.extend([sid] * len(wins))

        except Exception as exc:
            skipped.append({
                "subject_id": sid,
                "npz_path": rec.npz_path,
                "reason": str(exc),
            })

    if not all_x:
        raise RuntimeError(
            "No valid training windows were created. "
            "Check baseline coverage, NPZ paths and frame counts."
        )

    return (
        np.concatenate(all_x),
        np.asarray(all_y),
        np.asarray(all_groups),
        names,
        pd.DataFrame(skipped),
    )


def group_split(groups, y, seed=42):
    """
    Create subject-independent train, validation and test splits.

    All three R6 classes must occur in every split:
    alert, fatigue and distraction.
    """

    groups = np.asarray(groups)
    y = np.asarray(y)

    unique_subjects = np.unique(groups)

    if len(unique_subjects) < 3:
        raise RuntimeError(
            f"Need at least 3 distinct subjects for subject-wise "
            f"train/validation/test evaluation; found "
            f"{len(unique_subjects)}. "
            "Do not replace this with a random frame split."
        )

    expected_classes = set(range(len(LABELS)))
    observed_classes = set(np.unique(y).tolist())

    if observed_classes != expected_classes:
        raise RuntimeError(
            f"Dataset must contain all R6 classes {LABELS}. "
            f"Observed class IDs: {sorted(observed_classes)}."
        )

    for attempt in range(100):
        split_seed = seed + attempt

        # Reserve 20% of subjects for testing.
        gss1 = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=split_seed,
        )

        trv, te = next(
            gss1.split(np.zeros(len(y)), y, groups)
        )

        # Divide remaining subjects into training and validation.
        gss2 = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=split_seed + 1000,
        )

        tr_rel, va_rel = next(
            gss2.split(
                np.zeros(len(trv)),
                y[trv],
                groups[trv],
            )
        )

        tr = trv[tr_rel]
        va = trv[va_rel]

        # Require every class in all three splits.
        if (
            set(np.unique(y[tr])) != expected_classes
            or set(np.unique(y[va])) != expected_classes
            or set(np.unique(y[te])) != expected_classes
        ):
            continue

        train_subjects = set(groups[tr])
        val_subjects = set(groups[va])
        test_subjects = set(groups[te])

        # Verify that no subject appears in multiple splits.
        if (
            train_subjects.isdisjoint(val_subjects)
            and train_subjects.isdisjoint(test_subjects)
            and val_subjects.isdisjoint(test_subjects)
        ):
            return tr, va, te

    raise RuntimeError(
        "Could not create subject-wise train/validation/test splits "
        "containing all three R6 classes after 100 attempts. "
        "Add correctly labelled subjects or recordings. "
        "Do not use a random frame split."
    )


def apply_cue_dropout(x, p=0.15):
    # Current feature groups:
    # mouth, pose + relative pose, image quality.
    # Face detection is retained.
    if p <= 0:
        return x

    x = x.clone()

    groups = [
        (0,),
        (1, 2, 3, 4, 5),
        tuple(range(6, min(10, x.shape[-1]))),
    ]

    for group in groups:
        if not group:
            continue

        mask = (
            torch.rand(
                (x.shape[0], 1, 1),
                device=x.device,
            ) < p
        )

        x[:, :, list(group)] = torch.where(
            mask,
            torch.zeros_like(x[:, :, list(group)]),
            x[:, :, list(group)],
        )

    return x


def fit_temperature(logits, labels):
    # Scalar temperature optimized on validation logits.
    log_t = torch.zeros(
        1,
        requires_grad=True,
        device=logits.device,
    )

    opt = torch.optim.LBFGS(
        [log_t],
        lr=0.05,
        max_iter=100,
    )

    criterion = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()

        t = torch.exp(log_t).clamp(0.05, 20.0)
        loss = criterion(logits / t, labels)

        loss.backward()
        return loss

    opt.step(closure)

    return float(
        torch.exp(log_t)
        .clamp(0.05, 20.0)
        .detach()
        .cpu()
    )


def main():
    ap = argparse.ArgumentParser(
        description="Train NeuroNova R6 temporal GRU"
    )

    ap.add_argument(
        "--metadata",
        required=True,
        help="R3 metadata CSV",
    )
    ap.add_argument(
        "--baselines",
        required=True,
        help="R5 driver baseline CSV",
    )
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--window", type=int, default=75)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cue-dropout", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(a.metadata)
    baselines = load_baselines(a.baselines)

    x, y, groups, feature_names, skipped = collect_windows(
        metadata,
        baselines,
        a.window,
        a.stride,
    )

    skipped.to_csv(
        out / "r6_skipped_records.csv",
        index=False,
    )

    tr, va, te = group_split(groups, y, a.seed)

    # Fit scaling on training frames only.
    scaler = StandardScaler()

    scaler.fit(
        x[tr].reshape(-1, x.shape[-1])
    )

    x_scaled = (
        scaler.transform(
            x.reshape(-1, x.shape[-1])
        )
        .reshape(x.shape)
        .astype(np.float32)
    )

    train_ds = WindowDataset(x_scaled[tr], y[tr])
    val_ds = WindowDataset(x_scaled[va], y[va])
    test_ds = WindowDataset(x_scaled[te], y[te])

    train_loader = DataLoader(
        train_ds,
        batch_size=a.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=a.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=a.batch_size,
        shuffle=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = TemporalGRU(
        x.shape[-1],
        64,
        len(LABELS),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=a.lr,
        weight_decay=1e-4,
    )

    loss_fn = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(1, a.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            xb = apply_cue_dropout(
                xb,
                a.cue_dropout,
            )

            opt.zero_grad()

            logits = model(xb)
            loss = loss_fn(logits, yb)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            opt.step()

            train_losses.append(
                float(loss.detach().cpu())
            )

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                val_losses.append(
                    float(loss_fn(model(xb), yb).cpu())
                )

        v = (
            float(np.mean(val_losses))
            if val_losses
            else float("inf")
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": v,
        })

        print(
            f"Epoch {epoch:03d}/{a.epochs} "
            f"train_loss={history[-1]['train_loss']:.4f} "
            f"val_loss={v:.4f}"
        )

        if v < best_val:
            best_val = v

            best_state = {
                k: t.detach().cpu().clone()
                for k, t in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a model state."
        )

    model.load_state_dict(best_state)
    model.eval()

    def logits_for(loader):
        ls, ys = [], []

        with torch.no_grad():
            for xb, yb in loader:
                ls.append(
                    model(xb.to(device)).cpu()
                )
                ys.append(yb)

        return torch.cat(ls), torch.cat(ys)

    val_logits, val_y = logits_for(val_loader)

    temperature = fit_temperature(
        val_logits.to(device),
        val_y.to(device),
    )

    test_logits, test_y = logits_for(test_loader)

    probs = torch.softmax(
        test_logits / temperature,
        dim=1,
    ).numpy()

    pred = probs.argmax(axis=1)
    true = test_y.numpy()

    report = classification_report(
        true,
        pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "test_accuracy": float(
            accuracy_score(true, pred)
        ),
        "test_macro_f1": float(
            f1_score(
                true,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "temperature": temperature,
        "test_windows": int(len(true)),
        "test_subjects": sorted(
            set(groups[te].tolist())
        ),
        "train_subjects": sorted(
            set(groups[tr].tolist())
        ),
        "validation_subjects": sorted(
            set(groups[va].tolist())
        ),
        "labels": LABELS,
        "warning": (
            "Metrics are meaningful only if each subject and class "
            "is correctly labelled and the dataset is representative."
        ),
    }

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": int(x.shape[-1]),
            "hidden_size": 64,
            "num_classes": len(LABELS),
            "labels": LABELS,
            "window": a.window,
            "fps": a.fps,
            "feature_names": feature_names,
            "temperature": temperature,
        },
        out / "r6_gru.pt",
    )

    joblib.dump(
        scaler,
        out / "r6_preprocessing.joblib",
    )

    save_json(
        out / "r6_model_config.json",
        {
            "input_size": int(x.shape[-1]),
            "hidden_size": 64,
            "num_classes": len(LABELS),
            "labels": LABELS,
            "window": a.window,
            "stride": a.stride,
            "fps": a.fps,
            "feature_names": feature_names,
            "architecture": "one-layer GRU, hidden_size=64",
        },
    )

    save_json(
        out / "r6_temperature.json",
        {
            "temperature": temperature,
            "method": "scalar validation-set temperature scaling",
        },
    )

    save_json(
        out / "r6_metrics.json",
        metrics,
    )

    save_json(
        out / "r6_classification_report.json",
        report,
    )

    pd.DataFrame(history).to_csv(
        out / "r6_training_history.csv",
        index=False,
    )

    pd.DataFrame(
        confusion_matrix(
            true,
            pred,
            labels=list(range(len(LABELS))),
        ),
        index=LABELS,
        columns=LABELS,
    ).to_csv(
        out / "r6_confusion_matrix.csv"
    )

    pred_df = pd.DataFrame({
        "true_label": [LABELS[i] for i in true],
        "predicted_label": [LABELS[i] for i in pred],
        "confidence": probs.max(axis=1),
    })

    for i, name in enumerate(LABELS):
        pred_df[f"prob_{name}"] = probs[:, i]

    pred_df.to_csv(
        out / "r6_test_predictions.csv",
        index=False,
    )

    save_json(
        out / "r6_training_summary.json",
        {
            **metrics,
            "feature_names": feature_names,
            "total_windows": int(len(y)),
            "split_windows": {
                "train": len(tr),
                "validation": len(va),
                "test": len(te),
            },
            "skipped_records": int(len(skipped)),
        },
    )

    print(
        "\nTraining complete. R6 artefacts saved in:",
        out.resolve(),
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
