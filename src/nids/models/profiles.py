"""Data profiles saved at training time and used at serving time.

- baseline:     percentiles of *benign* traffic per feature -> plain-English alert stories (U1)
- reference:    training histograms per feature              -> drift monitor, PSI (U7)
- fingerprints: per-class median, normalised to 0..1          -> attack fingerprint radar (U9)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PERCENTILES = {"p1": 1, "p5": 5, "p25": 25, "p50": 50, "p75": 75, "p95": 95, "p99": 99}


def benign_baseline(X_benign: pd.DataFrame) -> dict:
    q = X_benign.quantile([p / 100 for p in PERCENTILES.values()])
    out = {}
    for f in X_benign.columns:
        out[f] = {name: float(q.loc[p / 100, f]) for name, p in PERCENTILES.items()}
        out[f]["min"] = float(X_benign[f].min())
        out[f]["max"] = float(X_benign[f].max())
    return out


def drift_reference(X: pd.DataFrame, bins: int = 10) -> dict:
    out = {}
    for f in X.columns:
        edges = np.unique(np.quantile(X[f], np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:          # (near-)constant feature: PSI is meaningless
            continue
        counts, _ = np.histogram(np.clip(X[f], edges[0], edges[-1]), bins=edges)
        out[f] = {"edges": edges.tolist(), "props": (counts / counts.sum()).tolist()}
    return out


def psi(ref: dict, values: np.ndarray, eps: float = 1e-4) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.2 moderate shift, >0.2 drifted."""
    edges = np.asarray(ref["edges"])
    counts, _ = np.histogram(np.clip(values, edges[0], edges[-1]), bins=edges)
    actual = counts / max(counts.sum(), 1)
    expected = np.asarray(ref["props"])
    a, e = np.clip(actual, eps, None), np.clip(expected, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def fingerprints(X: pd.DataFrame, y: pd.Series, features: list[str]) -> dict:
    lo, hi = X[features].quantile(0.01), X[features].quantile(0.99)
    span = (hi - lo).replace(0, 1)
    norm = ((X[features] - lo) / span).clip(0, 1)
    med = norm.groupby(y.values).median()
    return {"features": features,
            "classes": {c: [float(v) for v in med.loc[c, features]] for c in med.index}}


def save_profiles(root: Path, **profiles) -> None:
    d = root / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    for name, obj in profiles.items():
        (d / f"{name}.json").write_text(json.dumps(obj), encoding="utf-8")


def load_profile(root: Path, name: str) -> dict:
    p = root / "profiles" / f"{name}.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else {}
