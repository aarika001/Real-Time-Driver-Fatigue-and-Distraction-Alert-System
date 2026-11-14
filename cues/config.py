"""Load the R4 YAML config (configs/r4.yaml).

Every script takes ``--config``; values in the file override the defaults in
the dataclasses, so a missing file or section falls back to sensible values.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "r4.yaml"


def load_config(path: Optional[str | Path] = None) -> dict[str, Any]:
    """Return the parsed YAML as a dict ({} if no path and no default file)."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.exists():
        if path:
            raise FileNotFoundError(f"Config not found: {p}")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top level of {p} must be a mapping")
    return data
