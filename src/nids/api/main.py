"""FlowSight prediction service.

    uvicorn nids.api.main:app --reload --port 8000
    open http://localhost:8000/docs for interactive API docs

For every flow: classify -> (if benign) check the unknown-attack detector -> explain with SHAP
-> write a plain-English story -> score severity -> store -> group into an incident.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from nids.api.schemas import BatchIn, FeedbackIn, FlowIn, Prediction
from nids.config import ROOT, models_dir
from nids.explain.narrative import story as make_story
from nids.explain.narrative import unusual_features
from nids.explain.shap_explainer import FlowExplainer
from nids.models.anomaly import UNKNOWN, AnomalyDetector
from nids.models.profiles import load_profile
from nids.models.registry import list_models, load_model
from nids.monitoring import drift
from nids.scoring.severity import load_severity_config
from nids.scoring.severity import score as severity_score
from nids.storage import db, feedback, incidents

BENIGN = "Benign"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    root = models_dir()
    pipe, meta = load_model(root=root)
    state.update(
        pipe=pipe, meta=meta, explainer=FlowExplainer(pipe, meta["features"]),
        detector=AnomalyDetector.load(root / "anomaly"),
        baseline=load_profile(root, "baseline"),
        reference=load_profile(root, "reference"),
        fingerprints=load_profile(root, "fingerprints"),
        severity_cfg=load_severity_config(),
        conn=db.connect(),
    )
    print(f"Loaded model '{meta['name']}' (macro-F1 {meta['metrics']['macro_f1']:.4f}); "
          f"anomaly detector: {'on' if state['detector'] else 'off'}")
    yield
    state["conn"].close()


app = FastAPI(title="FlowSight NIDS API", version="0.2.0", lifespan=lifespan)


def _to_frame(flows: list[FlowIn]) -> pd.DataFrame:
    features = state["meta"]["features"]
    for fl in flows:
        missing = set(features) - set(fl.features)
        if missing:
            raise HTTPException(422, f"Missing features: {sorted(missing)[:10]}"
                                     f"{' ...' if len(missing) > 10 else ''}")
    return pd.DataFrame([{f: fl.features[f] for f in features} for fl in flows],
                        columns=features).astype("float32")


def _predict(flows: list[FlowIn], store: bool = True) -> list[Prediction]:
    pipe, meta, explainer = state["pipe"], state["meta"], state["explainer"]
    detector, baseline, conn = state["detector"], state["baseline"], state["conn"]
    classes = meta["classes"]
    min_conf = meta["config"].get("anomaly", {}).get("min_benign_confidence")
    X = _to_frame(flows)
    proba = pipe.predict_proba(X)
    anomaly = detector.anomaly_score(X) if detector else None

    out = []
    for i in range(len(X)):
        idx = int(proba[i].argmax())
        label, confidence = classes[idx], float(proba[i, idx])
        row = X.iloc[[i]]
        a_score = float(anomaly[i]) if anomaly is not None else None
        note = None

        odd = a_score is not None and a_score > 0             # unlike any benign flow
        unsure = min_conf is not None and confidence < min_conf  # classifier doubts "Benign"
        if label == BENIGN and (odd or unsure):
            label = UNKNOWN
            p_benign = confidence
            confidence = max(1 - p_benign, min(0.5 + a_score * 5, 0.99) if odd else 0.0)
            explanation = unusual_features(row.iloc[0].to_dict(), baseline)
            if not explanation:
                # nothing out of range: explain what pulls it towards the runner-up class
                explanation = explainer.explain(row, int(proba[i].argsort()[-2]))
            if unsure:
                note = f"The model is only {p_benign:.0%} sure this is normal traffic."
        elif label != BENIGN:
            # SHAP only for alerts: it costs a few ms, and analysts only read alerts
            explanation = explainer.explain(row, idx)
        else:
            explanation = []

        is_attack = label != BENIGN
        text = make_story(label, confidence, explanation, baseline, note=note)
        pred = Prediction(
            label=label, confidence=confidence, is_attack=is_attack,
            probabilities={c: float(p) for c, p in zip(classes, proba[i])},
            explanation=explanation, story=text, anomaly_score=a_score, model=meta["name"],
        )
        if is_attack:
            ts = db.now()
            recent = db.count_recent_alerts(conn, label, state["severity_cfg"]["burst_window_s"],
                                            ts) + 1
            pred.severity_score, pred.severity = severity_score(label, confidence, recent,
                                                                state["severity_cfg"])
        if store:
            pred.id = db.save_prediction(
                conn, label=label, confidence=confidence, is_attack=is_attack,
                model=meta["name"], explanation=explanation, features=row.iloc[0].to_dict(),
                severity=pred.severity, severity_score=pred.severity_score, story=text,
                anomaly_score=a_score)
            if is_attack:
                pred.incident_id = incidents.attach(conn, pred.id, label, db.now(),
                                                    pred.severity, pred.severity_score, text)
        out.append(pred)
    return out


# ------------------------------------------------------------------ service info
@app.get("/health")
def health():
    return {"status": "ok", "model": state["meta"]["name"],
            "anomaly_detector": state["detector"] is not None}


@app.get("/model")
def model_info():
    m = state["meta"]
    return {k: m[k] for k in ("name", "classes", "features", "selected_features", "metrics",
                              "trained_at")}


@app.get("/models")
def all_models():
    """Comparison of every trained model, plus the anomaly detector's metrics."""
    import json
    anomaly_path = models_dir() / "anomaly" / "anomaly.json"
    return {
        "models": [{"name": m["name"], "metrics": m["metrics"], "classes": m["classes"],
                    "selected_features": m["selected_features"],
                    "best": m["name"] == state["meta"]["name"]} for m in list_models()],
        "anomaly": json.loads(anomaly_path.read_text("utf-8")) if anomaly_path.exists()
        else None,
        "fingerprints": state["fingerprints"],
    }


@app.get("/baseline")
def baseline():
    """Benign percentiles per feature (used for what-if slider ranges)."""
    return state["baseline"]


@app.get("/reports")
def reports():
    """Experiment results written by the scripts in scripts/ (if they have been run)."""
    out = {}
    for name in ("FINAL_REPORT", "results", "zero_day", "cross_dataset", "feature_selection",
                 "imbalance", "port_ablation",
                 "shap_global", "DATASET", "MODEL_CARD"):
        p = ROOT / "reports" / f"{name}.md"
        if p.exists():
            out[name] = p.read_text("utf-8")
    return out


# ------------------------------------------------------------------ prediction
@app.post("/predict", response_model=Prediction)
def predict(flow: FlowIn, store: bool = True):
    return _predict([flow], store)[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(batch: BatchIn, store: bool = True):
    if not batch.flows:
        return []
    return _predict(batch.flows, store)


@app.get("/predictions")
def predictions(limit: int = 200, attacks_only: bool = False, incident_id: int | None = None):
    return db.recent(state["conn"], limit, attacks_only, incident_id)


@app.get("/predictions/{pred_id}")
def prediction(pred_id: int):
    p = db.get_prediction(state["conn"], pred_id)
    if not p:
        raise HTTPException(404, "Prediction not found")
    return p


@app.get("/stats")
def stats():
    return db.stats(state["conn"])


@app.delete("/predictions")
def clear_predictions():
    db.clear(state["conn"])
    return {"cleared": True}


# ------------------------------------------------------------------ incidents (U3)
@app.get("/incidents")
def list_incidents(limit: int = 200):
    return incidents.list_incidents(state["conn"], limit)


@app.get("/incidents/{inc_id}")
def get_incident(inc_id: int):
    inc = incidents.get_incident(state["conn"], inc_id)
    if not inc:
        raise HTTPException(404, "Incident not found")
    inc["alerts"] = db.recent(state["conn"], 500, incident_id=inc_id)
    return inc


# ------------------------------------------------------------------ feedback (U6)
@app.post("/feedback/{pred_id}")
def give_feedback(pred_id: int, body: FeedbackIn):
    if not db.get_prediction(state["conn"], pred_id):
        raise HTTPException(404, "Prediction not found")
    feedback.set_verdict(state["conn"], pred_id, body.verdict, body.note)
    return {"ok": True}


@app.get("/feedback/summary")
def feedback_summary():
    return feedback.summary(state["conn"])


@app.post("/feedback/export")
def feedback_export():
    path = ROOT / "data" / "feedback" / "feedback_labelled.csv"
    n = feedback.export(state["conn"], path)
    return {"rows": n, "path": str(path)}


# ------------------------------------------------------------------ drift (U7)
@app.get("/drift")
def drift_status(window: int = 500):
    return drift.check(db.recent_features(state["conn"], window), state["reference"],
                       state["meta"]["selected_features"])
