"""Plain-English alert stories (U1).

SHAP says *which* features mattered. A story says *why*, by comparing the flow's raw values
against normal (benign) traffic percentiles saved at training time. Template-based on
purpose: deterministic, fast, and testable.
"""

from __future__ import annotations

import math

# Friendlier names for common CICIDS features (fallback: the raw column name)
FRIENDLY = {
    "Flow Packets/s": "packet rate",
    "Flow Bytes/s": "byte rate",
    "Flow Duration": "flow duration",
    "Destination Port": "destination port",
    "Total Fwd Packets": "packets sent",
    "Total Backward Packets": "packets received",
    "Total Length of Fwd Packets": "bytes sent",
    "Fwd Packet Length Mean": "average outgoing packet size",
    "Bwd Packet Length Mean": "average reply packet size",
    "Average Packet Size": "average packet size",
    "Flow IAT Mean": "average gap between packets",
    "SYN Flag Count": "SYN flags",
    "RST Flag Count": "RST flags",
    "ACK Flag Count": "ACK flags",
    "PSH Flag Count": "PSH flags",
    "Init_Win_bytes_forward": "initial TCP window",
}

UNITS = {"Flow Packets/s": " packets/s", "Flow Bytes/s": " bytes/s", "Flow Duration": " µs",
         "Flow IAT Mean": " µs"}


def friendly(feature: str) -> str:
    return FRIENDLY.get(feature, feature)


def _fmt(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:,.1f}"
    return f"{v:.3g}"


def describe(feature: str, value: float, base: dict | None) -> str:
    """One clause comparing a raw value to benign traffic."""
    name, unit = friendly(feature), UNITS.get(feature, "")
    if feature == "Destination Port":
        return f"it targeted port {int(value)}"
    if not base:
        return f"{name} was {_fmt(value)}{unit}"
    p50 = base["p50"]
    if value > base["p99"]:
        if p50 > 0 and value / p50 >= 2:
            return (f"{name} was {_fmt(value)}{unit}, {_fmt(value / p50)}× the normal median "
                    f"(higher than 99% of normal traffic)")
        return f"{name} was {_fmt(value)}{unit}, higher than 99% of normal traffic"
    if value < base["p1"]:
        return (f"{name} was unusually low at {_fmt(value)}{unit} "
                f"(normal median {_fmt(p50)}{unit})")
    if value > base["p95"]:
        return f"{name} was high at {_fmt(value)}{unit} (top 5% of normal traffic)"
    if value < base["p5"]:
        return f"{name} was low at {_fmt(value)}{unit} (bottom 5% of normal traffic)"
    return f"{name} was {_fmt(value)}{unit}, which is within the normal range"


def unusual_features(row: dict, baseline: dict, top_k: int = 3) -> list[dict]:
    """For anomalies there is no class to explain, so rank features by how far they sit
    outside the benign range (log-distance from the benign median)."""
    scored = []
    for f, v in row.items():
        b = baseline.get(f)
        if not b or b["p1"] <= v <= b["p99"]:
            continue
        ref = b["p50"] if b["p50"] != 0 else (b["p99"] or 1.0)
        dist = abs(math.log10((abs(v) + 1e-9) / (abs(ref) + 1e-9)))
        scored.append({"feature": f, "value": float(v), "contribution": float(dist)})
    return sorted(scored, key=lambda d: -d["contribution"])[:top_k]


def _is_unusual(feature: str, value: float, base: dict | None) -> bool:
    if feature == "Destination Port" or not base:
        return True
    return value > base["p95"] or value < base["p5"]


def story(label: str, confidence: float, top_features: list[dict], baseline: dict,
          note: str | None = None) -> str:
    text = _story(label, confidence, top_features, baseline)
    return f"{text} {note}" if note else text


def _story(label: str, confidence: float, top_features: list[dict], baseline: dict) -> str:
    top = top_features[:4]
    # Prefer reasons that are actually unusual; "within the normal range" explains nothing.
    strong = [f for f in top if _is_unusual(f["feature"], f["value"],
                                            baseline.get(f["feature"]))]
    chosen = (strong or top[:1])[:3]
    reasons = [describe(f["feature"], f["value"], baseline.get(f["feature"])) for f in chosen]
    if label == "Benign":
        return f"Looks like normal traffic ({confidence:.0%} confident)."
    if label == "Unknown-Anomaly":
        head = "Doesn't match any known attack, but is unlike normal traffic"
    else:
        head = f"Flagged as {label} ({confidence:.0%} confident)"
    if not reasons:
        return head + "."
    if not strong:
        # Nothing is individually unusual: say so instead of listing a "normal" value
        f = top[0]
        return (f"{head}: no single feature is out of the ordinary; the verdict comes from "
                f"the combination, led by {friendly(f['feature'])} "
                f"({_fmt(f['value'])}{UNITS.get(f['feature'], '')}).")
    if len(reasons) == 1:
        return f"{head}: {reasons[0]}."
    return f"{head}: {'; '.join(reasons[:-1])}; and {reasons[-1]}."
