from nids.data.clean import split_xy
from nids.explain.shap_explainer import FlowExplainer
from nids.models.registry import best_name, load_model


def test_artifacts_saved(trained_dir):
    assert (trained_dir / "best.json").exists()
    for name in ("rf", "xgb"):
        assert (trained_dir / name / "model.joblib").exists()


def test_selected_feature_count(trained_dir, tiny_cfg):
    _, meta = load_model(root=trained_dir)
    assert len(meta["selected_features"]) == tiny_cfg["features"]["k"]


def test_prediction_is_deterministic(trained_dir, clean_df):
    pipe, meta = load_model(root=trained_dir)
    X, _ = split_xy(clean_df)
    X = X[meta["features"]].head(50)
    assert (pipe.predict(X) == pipe.predict(X)).all()


def test_explainer_returns_top_features(trained_dir, clean_df):
    pipe, meta = load_model(best_name(trained_dir), root=trained_dir)
    X, _ = split_xy(clean_df)
    row = X[meta["features"]].head(1)
    idx = int(pipe.predict_proba(row)[0].argmax())
    ex = FlowExplainer(pipe, meta["features"]).explain(row, idx, top_k=3)
    assert len(ex) == 3
    assert all(e["feature"] in meta["selected_features"] for e in ex)
