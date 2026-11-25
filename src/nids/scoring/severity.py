"""Severity scoring (U2). Weights live in configs/severity.yaml, not in code."""

from __future__ import annotations

import math
from functools import lru_cache

import yaml

from nids.config import ROOT

SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]


@lru_cache
def load_severity_config(path: str | None = None) -> dict:
    with open(path or ROOT / "configs" / "severity.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def score(label: str, confidence: float, recent_same_type: int, cfg: dict | None = None
          ) -> tuple[float, str]:
    """Return (score, band). recent_same_type counts this alert too (>= 1)."""
    cfg = cfg or load_severity_config()
    base = cfg["base"].get(label, cfg["base"]["default"])
    burst = 1 + cfg["burst_weight"] * math.log10(max(recent_same_type, 1))
    s = min(base * confidence * burst, 1.0)
    for band in reversed(SEVERITY_ORDER):           # Critical first
        if s >= cfg["bands"][band]:
            return s, band
    return s, "Low"


def max_band(a: str | None, b: str) -> str:
    if a is None:
        return b
    return max(a, b, key=SEVERITY_ORDER.index)
