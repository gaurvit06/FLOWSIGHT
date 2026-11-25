"""Incident grouping (U3): turn thousands of alerts into a handful of incidents.

Rule: an alert joins the latest incident of the same attack type if that incident's last
alert was less than GAP_S seconds ago; otherwise it opens a new incident. With live capture
you would also group by target host.
"""

from __future__ import annotations

from datetime import datetime

from nids.scoring.severity import max_band
from nids.storage.db import iso, now

GAP_S = 120


def attach(conn, pred_id: int, label: str, ts: datetime, severity: str, score: float,
           story: str) -> int:
    row = conn.execute("SELECT * FROM incidents WHERE label=? ORDER BY end_ts DESC LIMIT 1",
                       (label,)).fetchone()
    if row and (ts - datetime.fromisoformat(row["end_ts"])).total_seconds() <= GAP_S:
        inc_id = row["id"]
        better = score > row["max_score"]
        conn.execute(
            "UPDATE incidents SET end_ts=?, n_alerts=n_alerts+1, max_severity=?, "
            "max_score=?, story=? WHERE id=?",
            (iso(ts), max_band(row["max_severity"], severity), max(score, row["max_score"]),
             story if better else row["story"], inc_id))
    else:
        inc_id = conn.execute(
            "INSERT INTO incidents (label, start_ts, end_ts, n_alerts, max_severity, max_score,"
            " story) VALUES (?,?,?,?,?,?,?)",
            (label, iso(ts), iso(ts), 1, severity, score, story)).lastrowid
    conn.execute("UPDATE predictions SET incident_id=? WHERE id=?", (inc_id, pred_id))
    conn.commit()
    return inc_id


def _with_status(d: dict) -> dict:
    age = (now() - datetime.fromisoformat(d["end_ts"])).total_seconds()
    d["status"] = "active" if age <= GAP_S else "closed"
    d["duration_s"] = (datetime.fromisoformat(d["end_ts"])
                       - datetime.fromisoformat(d["start_ts"])).total_seconds()
    return d


def list_incidents(conn, limit: int = 200) -> list[dict]:
    return [_with_status(dict(r)) for r in conn.execute(
        "SELECT * FROM incidents ORDER BY end_ts DESC LIMIT ?", (limit,))]


def get_incident(conn, inc_id: int) -> dict | None:
    r = conn.execute("SELECT * FROM incidents WHERE id=?", (inc_id,)).fetchone()
    return _with_status(dict(r)) if r else None
