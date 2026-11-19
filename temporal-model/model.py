"""Reliability-aware temporal model: one-layer GRU over 5-second windows, random cue
dropout during training, and temperature scaling for calibrated probabilities."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from features import FLAG_START, GROUP_CUE_IDX, GROUP_NAMES, N_FEATURES


class TemporalGRU(nn.Module):
    """x: (B, T, N_FEATURES) -> logits (B, n_classes).

    The prediction is read from the GRU's last hidden state, so the same weights can
    run frame-by-frame in the live demo: pass the previous hidden state as ``h0`` and
    one frame at a time."""

    def __init__(self, n_features: int = N_FEATURES, hidden: int = 64, n_classes: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None, return_hidden: bool = False):
        _, h = self.gru(x, h0)                       # h: (1, B, hidden)
        logits = self.fc(self.drop(h[-1]))
        return (logits, h) if return_hidden else logits


def cue_dropout(x: torch.Tensor, p: float, groups: Sequence[int]) -> torch.Tensor:
    """Randomly hide whole cue groups (eye / mouth / head) for a whole window.

    A hidden group has its cue channels zeroed and its availability flag set to 0, so
    the network sees "this cue is missing" and must lean on the others.  At least one
    group is always kept.  Returns a modified copy; ``x`` is left untouched."""
    if p <= 0 or len(groups) < 2:
        return x
    x = x.clone()
    B = x.shape[0]
    drop = torch.rand(B, len(groups), device=x.device) < p
    all_drop = drop.all(dim=1)
    if all_drop.any():
        rows = all_drop.nonzero(as_tuple=True)[0]
        keep = torch.randint(len(groups), (len(rows),), device=x.device)
        drop[rows, keep] = False
    for j, g in enumerate(groups):
        m = drop[:, j]
        if not m.any():
            continue
        for c in GROUP_CUE_IDX[GROUP_NAMES[g]]:
            x[m, :, c] = 0.0
        x[m, :, FLAG_START + g] = 0.0
    return x


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Single temperature T minimising NLL of softmax(logits / T) on held-out data
    (Guo et al., 2017).  Does not change the predicted class, only the confidence."""
    # Bounded 1-D search on a log grid.  (LBFGS diverges to T -> infinity when the model
    # is no better than chance on the validation set, because the NLL-optimal answer is
    # then "predict uniform".)
    with torch.no_grad():
        ts = torch.logspace(-1.3, 1.3, 261)                      # T from 0.05 to 20
        nlls = torch.stack([F.cross_entropy(logits / t, labels) for t in ts])
        return float(ts[nlls.argmin()])
