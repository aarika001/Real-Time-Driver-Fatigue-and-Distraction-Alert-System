import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from tqdm import tqdm

from .common import extract_feature_matrix, load_baselines, make_windows


LABELS = ["Normal", "Talking", "Yawning"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}


class WindowDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class ActivityGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers=1, batch_first=True
        )
        self.dropout = nn.Dropout(0.25)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(self.dropout(h[-1]))


def build_dataset(metadata, baselines, window=75, stride=15):
    all_x, all_y, all_groups, all_records = [], [], [], []
    skipped = []
    feature_names = None

    for _, rec in tqdm(metadata.iterrows(), total=len(metadata),
                       desc="Extracting activity windows"):
        sid = str(rec.subject_id).strip()
        label = str(rec.source_label).strip()

        if label not in LABEL_TO_ID:
            continue

        if sid not in baselines.index:
            skipped.append({
                "subject_id": sid,
                "npz_path": rec.npz_path,
                "reason": "no_neutral_baseline",
            })
            continue

        try:
            feats, names, detected = extract_feature_matrix(
                rec.npz_path, baselines.loc[sid]
            )

            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise ValueError("Inconsistent feature schema")

            wins = make_windows(feats, window, stride)
            starts = range(0, len(feats) - window + 1, stride)

            keep = [
                i for i, start in enumerate(starts)
                if np.mean(detected[start:start + window]) >= 0.60
            ]

            if not keep:
                continue

            wins = wins[keep]
            all_x.append(wins)
            all_y.extend([LABEL_TO_ID[label]] * len(wins))
            all_groups.extend([sid] * len(wins))
            all_records.extend([str(rec.source_filename)] * len(wins))

        except Exception as exc:
            skipped.append({
                "subject_id": sid,
                "npz_path": rec.npz_path,
                "reason": str(exc),
            })

    if not all_x:
        raise RuntimeError("No valid activity windows were generated.")

    return (
        np.concatenate(all_x),
        np.asarray(all_y, dtype=np.int64),
        np.asarray(all_groups),
        np.asarray(all_records),
        feature_names,
        pd.DataFrame(skipped),
    )


def subject_split(groups, y, seed=42):
    expected = set(range(len(LABELS)))

    for attempt in range(500):
        s = seed + attempt

        first = GroupShuffleSplit(
            n_splits=1, test_size=0.20, random_state=s
        )
        trainval, test = next(
            first.split(np.zeros(len(y)), y, groups)
        )

        second = GroupShuffleSplit(
            n_splits=1, test_size=0.20, random_state=s + 1000
        )
        train_rel, val_rel = next(
            second.split(
                np.zeros(len(trainval)),
                y[trainval],
                groups[trainval],
            )
        )

        train = trainval[train_rel]
        val = trainval[val_rel]

        if all(set(np.unique(y[idx])) == expected
               for idx in (train, val, test)):
            train_subjects = set(groups[train])
            val_subjects = set(groups[val])
            test_subjects = set(groups[test])

            if (train_subjects.isdisjoint(val_subjects)
                and train_subjects.isdisjoint(test_subjects)
                and val_subjects.isdisjoint(test_subjects)):
                return train, val, test

    raise RuntimeError(
        "Could not create subject-independent splits containing all "
        "three classes. No random-window split will be substituted."
    )


def evaluate(model, loader, device):
    model.eval()
    actual, predicted = [], []

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            predicted.extend(logits.argmax(1).cpu().numpy())
            actual.extend(yb.numpy())

    return np.asarray(actual), np.asarray(predicted)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        default="data/processed/mirror_sqpnp_full/mirror_metadata.csv"
    )
    parser.add_argument(
        "--baselines",
        default="data/processed/mirror_sqpnp_full/calibration/driver_baselines_r5.csv"
    )
    parser.add_argument(
        "--output", default="outputs/r6_activity_experiment"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata)
    required = {"subject_id", "source_label", "npz_path", "status"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")

    metadata["subject_id"] = metadata["subject_id"].astype(str).str.strip()
    metadata["source_label"] = metadata["source_label"].astype(str).str.strip()

    # Exclude mixed recordings and failed preprocessing records.
    metadata = metadata[
        metadata["source_label"].isin(LABELS)
        & (metadata["status"].astype(str).str.lower() == "success")
    ].copy()

    print("\nRecording counts:")
    print(metadata["source_label"].value_counts().to_string())

    baselines = load_baselines(args.baselines)

    x, y, groups, records, feature_names, skipped = build_dataset(
        metadata, baselines, window=75, stride=15
    )

    skipped.to_csv(out / "skipped_records.csv", index=False)

    print(f"\nWindows: {len(y)}")
    print(f"Subjects with windows: {len(np.unique(groups))}")
    print(f"Features: {len(feature_names)}")

    train_idx, val_idx, test_idx = subject_split(
        groups, y, args.seed
    )

    split_summary = {}
    for name, idx in [
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ]:
        split_summary[name] = {
            "windows": int(len(idx)),
            "subjects": sorted(set(groups[idx].tolist())),
            "class_counts": {
                LABELS[i]: int(np.sum(y[idx] == i))
                for i in range(len(LABELS))
            },
        }

    with open(out / "subject_splits.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    # Fit scaler only on training data.
    scaler = StandardScaler()
    scaler.fit(x[train_idx].reshape(-1, x.shape[-1]))

    x_scaled = scaler.transform(
        x.reshape(-1, x.shape[-1])
    ).reshape(x.shape).astype(np.float32)
    joblib.dump(scaler, out / "activity_scaler.joblib")
    print("Preprocessing scaler saved.")

    train_ds = WindowDataset(x_scaled[train_idx], y[train_idx])
    val_ds = WindowDataset(x_scaled[val_idx], y[val_idx])
    test_ds = WindowDataset(x_scaled[test_idx], y[test_idx])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"\nTraining device: {device}")

    model = ActivityGRU(
        input_size=x.shape[-1],
        hidden_size=64,
        num_classes=len(LABELS),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    history = []

    def save_checkpoint(path, epoch, val_loss):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "input_size": int(x.shape[-1]),
            "hidden_size": 64,
            "num_classes": len(LABELS),
            "labels": LABELS,
            "feature_names": feature_names,
            "window_frames": 75,
            "stride_frames": 15,
            "validation_loss": float(val_loss),
            "best_validation_loss": float(best_val_loss),
            "experiment": "Mirror activity recognition",
            "not_official_r6_vigilance_checkpoint": True,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                val_losses.append(
                    loss_fn(logits, yb.to(device)).item()
                )

        train_loss = float(np.mean(losses))
        val_loss = float(np.mean(val_losses))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
        })

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            save_checkpoint(
                out / "activity_gru_best.pt", epoch, val_loss
            )

        save_checkpoint(
            out / "activity_gru_last.pt", epoch, val_loss
        )
        pd.DataFrame(history).to_csv(
            out / "training_history.csv", index=False
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    actual, predicted = evaluate(model, test_loader, device)

    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        actual, predicted, labels=list(range(len(LABELS)))
    )

    pd.DataFrame(
        matrix, index=LABELS, columns=LABELS
    ).to_csv(out / "confusion_matrix.csv")

    predictions = pd.DataFrame({
        "subject_id": groups[test_idx],
        "source_filename": records[test_idx],
        "actual": [LABELS[i] for i in actual],
        "predicted": [LABELS[i] for i in predicted],
    })
    predictions.to_csv(out / "test_predictions.csv", index=False)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_size": int(x.shape[-1]),
        "hidden_size": 64,
        "labels": LABELS,
        "feature_names": feature_names,
        "window_frames": 75,
        "stride_frames": 15,
        "experiment": "YawDD Mirror activity recognition",
        "not_official_r6_vigilance_checkpoint": True,
    }
    # Best checkpoint was already saved during training; do not overwrite it.
    joblib.dump(scaler, out / "activity_scaler.joblib")

    with open(out / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    results = {
        "experiment": "Exploratory YawDD Mirror activity recognition",
        "labels": LABELS,
        "excluded_label": "TalkingYawning",
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(
            actual, predicted, average="macro", zero_division=0
        )),
        "classification_report": report,
        "split_subject_counts": {
            k: len(v["subjects"]) for k, v in split_summary.items()
        },
        "window_counts": {
            k: v["windows"] for k, v in split_summary.items()
        },
        "warning": (
            "These are video-level activity labels, not validated "
            "alert/fatigue/distraction vigilance labels. This checkpoint "
            "is not the official R6 model."
        ),
    }

    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n========== TEST RESULTS ==========")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1:  {results['macro_f1']:.4f}")
    print("\nClassification report:")
    print(classification_report(
        actual, predicted, target_names=LABELS, zero_division=0
    ))
    print(f"\nArtifacts saved in: {out.resolve()}")


if __name__ == "__main__":
    main()
