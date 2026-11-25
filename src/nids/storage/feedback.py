"""Analyst feedback loop (U6).

Analysts mark alerts as confirmed or false positive. That gives a precision measured on real
traffic (not just the test set), and labelled flows that can be added to the next training run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nids.storage.db import iso, now

VERDICTS = {"confirmed", "false_positive"}


def set_verdict(conn, pred_id: int, verdict: str, note: str | None = None) -> None:
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}")
    conn.execute(
        "INSERT INTO feedback (prediction_id, verdict, note, ts) VALUES (?,?,?,?) "
        "ON CONFLICT(prediction_id) DO UPDATE SET verdict=excluded.verdict, "
        "note=excluded.note, ts=excluded.ts",
        (pred_id, verdict, note, iso(now())))
    conn.commit()


def summary(conn) -> dict:
    rows = conn.execute(
        "SELECT p.label, f.verdict, COUNT(*) AS n FROM predictions p "
        "JOIN feedback f ON f.prediction_id = p.id GROUP BY p.label, f.verdict").fetchall()
    per_label: dict[str, dict] = {}
    for r in rows:
        per_label.setdefault(r["label"], {"confirmed": 0, "false_positive": 0})[r["verdict"]] = r["n"]
    for v in per_label.values():
        reviewed = v["confirmed"] + v["false_positive"]
        v["precision"] = v["confirmed"] / reviewed if reviewed else None
    confirmed = sum(v["confirmed"] for v in per_label.values())
    fp = sum(v["false_positive"] for v in per_label.values())
    alerts = conn.execute("SELECT COUNT(*) FROM predictions WHERE is_attack=1").fetchone()[0]
    return {
        "reviewed": confirmed + fp,
        "unreviewed": alerts - confirmed - fp,
        "confirmed": confirmed,
        "false_positive": fp,
        "precision": confirmed / (confirmed + fp) if confirmed + fp else None,
        "per_label": per_label,
    }


def export(conn, path: Path) -> int:
    """Write reviewed flows with their corrected label (false positive -> Benign)."""
    rows = conn.execute(
        "SELECT p.label, p.features, f.verdict FROM predictions p "
        "JOIN feedback f ON f.prediction_id = p.id").fetchall()
    if not rows:
        return 0
    records = []
    for r in rows:
        rec = json.loads(r["features"])
        rec["Label"] = r["label"] if r["verdict"] == "confirmed" else "Benign"
        records.append(rec)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)
    return len(records)
