"""Pytest root config: put the repo root first on sys.path so `cues` and
`tests.r4` always resolve to this repo (some environments, e.g. Kaggle,
have another package called `tests` installed)."""
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
