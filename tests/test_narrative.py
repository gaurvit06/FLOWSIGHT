from nids.explain.narrative import describe, story, unusual_features

BASE = {"Flow Packets/s": {"p1": 1, "p5": 5, "p50": 50, "p95": 400, "p99": 1000},
        "Flow Duration": {"p1": 100, "p5": 1000, "p50": 8e5, "p95": 5e6, "p99": 9e6},
        "Average Packet Size": {"p1": 40, "p5": 60, "p50": 500, "p95": 1200, "p99": 1400}}


def test_describe_high_value_reports_ratio():
    s = describe("Flow Packets/s", 50_000, BASE["Flow Packets/s"])
    assert "1,000×" in s and "packet rate" in s and "99%" in s


def test_describe_low_value():
    s = describe("Flow Duration", 10, BASE["Flow Duration"])
    assert "unusually low" in s


def test_describe_port():
    assert describe("Destination Port", 22, None) == "it targeted port 22"


def test_story_prefers_unusual_reasons():
    top = [{"feature": "Average Packet Size", "value": 500, "contribution": 2.0},   # normal
           {"feature": "Flow Packets/s", "value": 50_000, "contribution": 1.0}]     # extreme
    s = story("DDoS", 0.97, top, BASE)
    assert s.startswith("Flagged as DDoS (97% confident)")
    assert "packet rate" in s and "average packet size" not in s


def test_story_benign_and_unknown():
    assert "normal traffic" in story("Benign", 0.99, [], BASE)
    s = story("Unknown-Anomaly", 0.7,
              [{"feature": "Flow Packets/s", "value": 1e6, "contribution": 3}], BASE)
    assert "Doesn't match any known attack" in s


def test_unusual_features_ranks_by_distance():
    row = {"Flow Packets/s": 1e6, "Flow Duration": 9e5, "Average Packet Size": 5}
    out = unusual_features(row, BASE)
    assert [o["feature"] for o in out] == ["Flow Packets/s", "Average Packet Size"]
