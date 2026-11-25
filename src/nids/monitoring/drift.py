"""Drift monitor (U7): is live traffic still like the training data?

Computes the Population Stability Index for each of the model's selected features over the
most recent flows. PSI > 0.2 on a feature means its distribution has shifted noticeably.
"""

from __future__ import annotations

import numpy as np

from nids.models.profiles import psi

DRIFT_PSI = 0.2
WARN_PSI = 0.1
MIN_FLOWS = 100


def check(recent_flows: list[dict], reference: dict, features: list[str]) -> dict:
    if len(recent_flows) < MIN_FLOWS:
        return {"status": "insufficient_data", "n_flows": len(recent_flows),
                "min_flows": MIN_FLOWS, "features": {}, "drifted": []}
    per_feature = {}
    for f in features:
        if f not in reference:
            continue
        values = np.array([r[f] for r in recent_flows if f in r], dtype=float)
        if len(values):
            per_feature[f] = psi(reference[f], values)
    drifted = sorted([f for f, v in per_feature.items() if v > DRIFT_PSI],
                     key=lambda f: -per_feature[f])
    warn = [f for f, v in per_feature.items() if WARN_PSI < v <= DRIFT_PSI]
    status = "drift" if drifted else ("warning" if warn else "stable")
    return {"status": status, "n_flows": len(recent_flows), "features": per_feature,
            "drifted": drifted, "warning": warn}
