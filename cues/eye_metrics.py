"""Metrics for the eye-state CNN, reported the way Section 3.3 of the proposal asks:
overall, per tagged condition, per person (min/median/max), with bootstrap 95%
confidence intervals over PEOPLE (not images), and groups with fewer than 5
people flagged rather than averaged away.

Positive class = eye CLOSED (label 1).
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MIN_PEOPLE = 5


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Average |confidence - accuracy| over equal-width probability bins (weighted by count).
    Low ECE means "P(closed) = 0.8" really is right about 80% of the time."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    conf = np.where(p >= 0.5, p, 1 - p)
    pred = (p >= 0.5).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - (pred[m] == y[m]).mean())
    return float(ece)


def binary_metrics(p: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    p, y = np.asarray(p, float), np.asarray(y, int)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "accuracy": float((tp + tn) / max(1, len(y))),
        "precision_closed": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "recall_closed": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "f1_closed": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan"),
        "auroc": float(roc_auc_score(y, p)) if both else float("nan"),
        "ece": expected_calibration_error(p, y),
    }


def per_subject_accuracy(p: np.ndarray, y: np.ndarray, subjects: Iterable[str],
                         threshold: float = 0.5) -> pd.Series:
    df = pd.DataFrame({"correct": ((np.asarray(p) >= threshold).astype(int) == np.asarray(y)),
                       "subject": list(subjects)})
    return df.groupby("subject")["correct"].mean()


def bootstrap_ci_over_subjects(p: np.ndarray, y: np.ndarray, subjects: Iterable[str],
                               metric: str = "accuracy", n_boot: int = 1000,
                               seed: int = 0) -> tuple[float, float]:
    """95% CI by resampling PEOPLE with replacement (images of one person are correlated)."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    subjects = np.asarray(list(subjects))
    uniq = np.unique(subjects)
    if len(uniq) < 2:
        return float("nan"), float("nan")
    idx_by_subj = {s: np.flatnonzero(subjects == s) for s in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_subj[s] for s in pick])
        v = binary_metrics(p[idx], y[idx])[metric]
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_predictions(df: pd.DataFrame, group_cols: Iterable[str] = (),
                          threshold: float = 0.5, n_boot: int = 1000) -> pd.DataFrame:
    """One row for 'overall' plus one per value of each group column.

    ``df`` needs columns p_closed, label_closed, subject_id (+ the group columns).
    """
    rows = []

    def add(name: str, part: pd.DataFrame) -> None:
        m = binary_metrics(part["p_closed"], part["label_closed"], threshold)
        acc_s = per_subject_accuracy(part["p_closed"], part["label_closed"], part["subject_id"], threshold)
        lo, hi = bootstrap_ci_over_subjects(part["p_closed"].to_numpy(), part["label_closed"].to_numpy(),
                                            part["subject_id"], "accuracy", n_boot)
        n_people = int(part["subject_id"].nunique())
        rows.append({"group": name, **m, "acc_ci_low": lo, "acc_ci_high": hi, "n_people": n_people,
                     "person_acc_min": float(acc_s.min()), "person_acc_median": float(acc_s.median()),
                     "person_acc_max": float(acc_s.max()),
                     "flag": "fewer than 5 people" if n_people < MIN_PEOPLE else ""})

    add("overall", df)
    for col in group_cols:
        if col not in df.columns:
            continue
        for val, part in df.groupby(col):
            if str(val) != "unknown":
                add(f"{col}={val}", part)
    return pd.DataFrame(rows)


def worst_condition(summary: pd.DataFrame, metric: str = "recall_closed") -> Optional[pd.Series]:
    """Worst group (by ``metric``) among groups with >= 5 people, as the proposal requires."""
    ok = summary[(summary["group"] != "overall") & (summary["n_people"] >= MIN_PEOPLE)]
    ok = ok[np.isfinite(ok[metric])]
    return ok.sort_values(metric).iloc[0] if len(ok) else None


def to_markdown(summary: pd.DataFrame, cols: Optional[list[str]] = None) -> str:
    cols = cols or ["group", "n", "n_people", "accuracy", "acc_ci_low", "acc_ci_high", "f1_closed",
                    "recall_closed", "auroc", "ece", "person_acc_min", "person_acc_median", "flag"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in summary.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
