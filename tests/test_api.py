import pytest
from fastapi.testclient import TestClient

from nids.data.clean import LABEL_COL


@pytest.fixture()
def client(trained_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("NIDS_MODELS_DIR", str(trained_dir))
    monkeypatch.setenv("NIDS_DB_PATH", str(tmp_path / "test.db"))
    from nids.api.main import app
    with TestClient(app) as c:
        yield c


def _flow(clean_df, label):
    row = clean_df[clean_df[LABEL_COL] == label].iloc[0].drop(LABEL_COL)
    return {"features": row.astype(float).to_dict()}


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_attack_has_explanation(client, clean_df):
    r = client.post("/predict", json=_flow(clean_df, "DDoS"))
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "DDoS"
    assert body["is_attack"] and body["explanation"]
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-4


def test_missing_features_rejected(client):
    r = client.post("/predict", json={"features": {"Flow Duration": 1.0}})
    assert r.status_code == 422


def test_predictions_are_stored(client, clean_df):
    client.post("/predict", json=_flow(clean_df, "PortScan"))
    s = client.get("/stats").json()
    assert s["total"] >= 1 and s["attacks"] >= 1
    assert client.get("/predictions", params={"attacks_only": True}).json()


def test_alert_has_story_severity_and_incident(client, clean_df):
    body = client.post("/predict", json=_flow(clean_df, "DDoS")).json()
    assert body["story"].startswith("Flagged as DDoS")
    assert body["severity"] in {"Low", "Medium", "High", "Critical"}
    assert body["incident_id"] is not None
    inc = client.get(f"/incidents/{body['incident_id']}").json()
    assert inc["label"] == "DDoS" and inc["alerts"][0]["id"] == body["id"]


def test_benign_has_no_severity(client, clean_df):
    body = client.post("/predict", json=_flow(clean_df, "Benign")).json()
    assert body["label"] == "Benign" and body["severity"] is None
    assert body["incident_id"] is None


def test_unknown_anomaly(client, clean_df):
    flow = _flow(clean_df, "Benign")
    flow["features"]["Flow Duration"] *= 1e5
    body = client.post("/predict", json=flow).json()
    assert body["label"] == "Unknown-Anomaly" and body["is_attack"]
    assert "Doesn't match any known attack" in body["story"]


def test_store_false_does_not_persist(client, clean_df):
    before = client.get("/stats").json()["total"]
    body = client.post("/predict", params={"store": False}, json=_flow(clean_df, "DoS")).json()
    assert body["id"] is None
    assert client.get("/stats").json()["total"] == before


def test_feedback_flow(client, clean_df):
    pid = client.post("/predict", json=_flow(clean_df, "BruteForce")).json()["id"]
    assert client.post(f"/feedback/{pid}", json={"verdict": "false_positive"}).status_code == 200
    s = client.get("/feedback/summary").json()
    assert s["false_positive"] == 1 and s["precision"] == 0.0
    assert client.post("/feedback/999999", json={"verdict": "confirmed"}).status_code == 404
    assert client.post(f"/feedback/{pid}", json={"verdict": "nope"}).status_code == 422


def test_drift_and_model_endpoints(client, clean_df):
    assert client.get("/drift").json()["status"] == "insufficient_data"
    m = client.get("/models").json()
    assert {x["name"] for x in m["models"]} == {"rf", "xgb"}
    assert m["anomaly"]["metrics"]["attack_flag_rate"] >= 0
    assert m["fingerprints"]["classes"]
    assert "Flow Packets/s" in client.get("/baseline").json()
