"""SQLite store for predictions, incidents and analyst feedback.

Swap for Postgres (SQLAlchemy) if you deploy this for real.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nids.config import db_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    label       TEXT NOT NULL,
    confidence  REAL NOT NULL,
    is_attack   INTEGER NOT NULL,
    model       TEXT,
    explanation TEXT,
    features    TEXT
);
CREATE TABLE IF NOT EXISTS incidents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    label         TEXT NOT NULL,
    start_ts      TEXT NOT NULL,
    end_ts        TEXT NOT NULL,
    n_alerts      INTEGER NOT NULL,
    max_severity  TEXT NOT NULL,
    max_score     REAL NOT NULL,
    story         TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    prediction_id INTEGER PRIMARY KEY REFERENCES predictions(id) ON DELETE CASCADE,
    verdict       TEXT NOT NULL CHECK (verdict IN ('confirmed', 'false_positive')),
    note          TEXT,
    ts            TEXT NOT NULL
);
"""

# Columns added after the first version. Added in place so old databases keep working.
MIGRATIONS = {
    "predictions": {
        "severity": "TEXT", "severity_score": "REAL", "story": "TEXT",
        "incident_id": "INTEGER", "anomaly_score": "REAL",
    },
}

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(ts);
CREATE INDEX IF NOT EXISTS idx_pred_attack ON predictions(is_attack, label, ts);
CREATE INDEX IF NOT EXISTS idx_pred_incident ON predictions(incident_id);
CREATE INDEX IF NOT EXISTS idx_inc_label ON incidents(label, end_ts);
"""


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.isoformat()


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    for table, cols in MIGRATIONS.items():
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        for col, typ in cols.items():
            if col not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
    conn.executescript(INDEXES)
    conn.commit()
    return conn


def save_prediction(conn, *, label: str, confidence: float, is_attack: bool, model: str,
                    explanation: list, features: dict, severity: str | None = None,
                    severity_score: float | None = None, story: str | None = None,
                    anomaly_score: float | None = None, ts: datetime | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO predictions (ts, label, confidence, is_attack, model, explanation, features,"
        " severity, severity_score, story, anomaly_score) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (iso(ts or now()), label, confidence, int(is_attack), model, json.dumps(explanation),
         json.dumps(features), severity, severity_score, story, anomaly_score),
    )
    conn.commit()
    return cur.lastrowid


def count_recent_alerts(conn, label: str, seconds: int, at: datetime | None = None) -> int:
    since = iso((at or now()) - timedelta(seconds=seconds))
    return conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE is_attack=1 AND label=? AND ts>=?",
        (label, since)).fetchone()[0]


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["explanation"] = json.loads(d.get("explanation") or "[]")
    d["features"] = json.loads(d.get("features") or "{}")
    d["is_attack"] = bool(d["is_attack"])
    return d


def get_prediction(conn, pred_id: int) -> dict | None:
    r = conn.execute(
        "SELECT p.*, f.verdict FROM predictions p LEFT JOIN feedback f "
        "ON f.prediction_id = p.id WHERE p.id=?", (pred_id,)).fetchone()
    return _row(r) if r else None


def recent(conn, limit: int = 200, attacks_only: bool = False,
           incident_id: int | None = None) -> list[dict]:
    q = ("SELECT p.*, f.verdict FROM predictions p "
         "LEFT JOIN feedback f ON f.prediction_id = p.id WHERE 1=1")
    args: list = []
    if attacks_only:
        q += " AND p.is_attack = 1"
    if incident_id is not None:
        q += " AND p.incident_id = ?"
        args.append(incident_id)
    q += " ORDER BY p.id DESC LIMIT ?"
    args.append(limit)
    return [_row(r) for r in conn.execute(q, args)]


def recent_features(conn, limit: int = 500) -> list[dict]:
    return [json.loads(r[0]) for r in conn.execute(
        "SELECT features FROM predictions ORDER BY id DESC LIMIT ?", (limit,))]


def stats(conn) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    attacks = conn.execute("SELECT COUNT(*) FROM predictions WHERE is_attack=1").fetchone()[0]
    by_label = {r["label"]: r["n"] for r in conn.execute(
        "SELECT label, COUNT(*) AS n FROM predictions GROUP BY label ORDER BY n DESC")}
    by_severity = {r["severity"]: r["n"] for r in conn.execute(
        "SELECT severity, COUNT(*) AS n FROM predictions WHERE is_attack=1 "
        "GROUP BY severity")}
    # per-minute timeline (ts is ISO-8601, first 16 chars = YYYY-MM-DDTHH:MM)
    timeline = [dict(r) for r in conn.execute(
        "SELECT substr(ts, 1, 16) AS minute, label, COUNT(*) AS n FROM predictions "
        "GROUP BY minute, label ORDER BY minute")]
    return {"total": total, "attacks": attacks, "by_label": by_label,
            "by_severity": by_severity, "timeline": timeline}


def clear(conn) -> None:
    conn.executescript("DELETE FROM feedback; DELETE FROM predictions; DELETE FROM incidents;")
    conn.commit()
