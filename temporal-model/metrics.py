"""Evaluation helpers: calibration error, per-subject spread, subject-level bootstrap."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error on the top-1 confidence."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def per_subject_accuracy(subjects: np.ndarray, correct: np.ndarray):
    """subject -> (n_correct, n_windows)."""
    out = {}
    for s in np.unique(subjects):
        m = subjects == s
        out[str(s)] = (int(correct[m].sum()), int(m.sum()))
    return out


def bootstrap_ci_by_subject(counts: dict, n_boot: int = 1000, seed: int = 0):
    """95% CI of pooled window accuracy, resampling *subjects* (not windows)."""
    keys = list(counts)
    c = np.array([counts[k][0] for k in keys], dtype=float)
    n = np.array([counts[k][1] for k in keys], dtype=float)
    rng = np.random.RandomState(seed)
    accs = []
    for _ in range(n_boot):
        i = rng.randint(0, len(keys), len(keys))
        accs.append(c[i].sum() / max(n[i].sum(), 1.0))
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def summarize(probs: np.ndarray, y: np.ndarray, subjects: np.ndarray, class_names) -> dict:
    pred = probs.argmax(1)
    correct = pred == y
    res = {
        "n_windows": int(len(y)),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_per_class": {n: float(f) for n, f in zip(
            class_names, f1_score(y, pred, average=None, labels=range(len(class_names)), zero_division=0))},
        "nll": nll(probs, y),
        "ece": ece(probs, y),
    }
    if probs.shape[1] == 2 and len(np.unique(y)) == 2:
        res["roc_auc"] = float(roc_auc_score(y, probs[:, 1]))
    counts = per_subject_accuracy(subjects, correct)
    accs = np.array([c / n for c, n in counts.values()])
    res["per_subject_accuracy"] = {
        "min": float(accs.min()), "median": float(np.median(accs)), "max": float(accs.max()),
        "n_subjects": int(len(accs)),
    }
    res["accuracy_ci95_over_subjects"] = bootstrap_ci_by_subject(counts)
    return res
