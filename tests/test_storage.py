import sqlite3
from datetime import timedelta

import pandas as pd
import pytest

from nids.storage import db, feedback, incidents


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


def _alert(conn, label, ts, sev="High", score=0.6):
    pid = db.save_prediction(conn, label=label, confidence=0.9, is_attack=True, model="m",
                             explanation=[], features={"a": 1.0}, severity=sev,
                             severity_score=score, story=f"{label} story", ts=ts)
    return pid, incidents.attach(conn, pid, label, ts, sev, score, f"{label} story")


def test_alerts_close_in_time_join_one_incident(conn):
    t0 = db.now()
    _, i1 = _alert(conn, "DDoS", t0)
    _, i2 = _alert(conn, "DDoS", t0 + timedelta(seconds=30), sev="Critical", score=0.9)
    assert i1 == i2
    inc = incidents.get_incident(conn, i1)
    assert inc["n_alerts"] == 2 and inc["max_severity"] == "Critical"


def test_gap_or_different_type_opens_new_incident(conn):
    t0 = db.now()
    _, a = _alert(conn, "DDoS", t0)
    _, b = _alert(conn, "DDoS", t0 + timedelta(seconds=incidents.GAP_S + 5))
    _, c = _alert(conn, "PortScan", t0 + timedelta(seconds=incidents.GAP_S + 6))
    assert len({a, b, c}) == 3
    assert len(db.recent(conn, incident_id=b)) == 1


def test_count_recent_alerts(conn):
    t0 = db.now()
    for i in range(3):
        _alert(conn, "DoS", t0 - timedelta(seconds=i * 10))
    _alert(conn, "DoS", t0 - timedelta(seconds=600))
    assert db.count_recent_alerts(conn, "DoS", 60, t0) == 3


def test_feedback_precision_and_export(conn, tmp_path):
    t0 = db.now()
    p1, _ = _alert(conn, "DDoS", t0)
    p2, _ = _alert(conn, "DDoS", t0)
    _alert(conn, "DDoS", t0)
    feedback.set_verdict(conn, p1, "confirmed")
    feedback.set_verdict(conn, p2, "false_positive")
    s = feedback.summary(conn)
    assert s["precision"] == 0.5 and s["unreviewed"] == 1
    feedback.set_verdict(conn, p2, "confirmed")          # analysts can change their mind
    assert feedback.summary(conn)["precision"] == 1.0
    out = tmp_path / "fb.csv"
    assert feedback.export(conn, out) == 2
    assert set(pd.read_csv(out)["Label"]) == {"DDoS"}


def test_bad_verdict_rejected(conn):
    with pytest.raises(ValueError):
        feedback.set_verdict(conn, 1, "maybe")


def test_old_database_is_migrated(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE predictions (id INTEGER PRIMARY KEY, ts TEXT, label TEXT, "
                "confidence REAL, is_attack INTEGER, model TEXT, explanation TEXT, features TEXT)")
    old.commit()
    old.close()
    c = db.connect(path)
    cols = {r["name"] for r in c.execute("PRAGMA table_info(predictions)")}
    assert {"severity", "story", "incident_id", "anomaly_score"} <= cols
