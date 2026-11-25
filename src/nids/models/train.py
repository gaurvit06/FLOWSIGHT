"""Train RF / XGBoost / SVM on the same split, evaluate, save, and pick the best.
Also builds the unknown-attack detector and the data profiles used by stories, drift and
fingerprints.

    python -m nids.models.train --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from nids.config import load_config, models_dir, resolve
from nids.data.clean import clean, split_xy
from nids.data.load import load_raw
from nids.features.pipeline import build_pipeline, selected_features
from nids.models.anomaly import AnomalyDetector
from nids.models.evaluate import BENIGN, evaluate
from nids.models.factory import make_model
from nids.models.profiles import benign_baseline, drift_reference, fingerprints, save_profiles
from nids.models.registry import mark_best, save_model


def load_dataset(cfg: dict, rebuild: bool = False) -> pd.DataFrame:
    processed = resolve(cfg["data"]["processed_path"])
    if processed.exists() and not rebuild:
        print(f"Using cached processed data: {processed}")
        return pd.read_parquet(processed)
    print("Loading raw CSVs...")
    raw = load_raw(resolve(cfg["data"]["raw_dir"]), cfg["data"].get("sample_frac", 1.0),
                   cfg["seed"])
    df = clean(raw, drop_other=cfg["data"].get("drop_other", True))
    processed.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed, index=False)
    print(f"Cleaned: {len(raw):,} -> {len(df):,} rows. Saved {processed}")
    return df


def split(df: pd.DataFrame, cfg: dict):
    """The one train/test split every script uses, so results are comparable."""
    X, y = split_xy(df)
    le = LabelEncoder().fit(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, le.transform(y), test_size=cfg["data"]["test_size"], stratify=y,
        random_state=cfg["seed"])
    return X_train, X_test, y_train, y_test, list(le.classes_)


def fit_one(name: str, X_train, y_train, cfg: dict, n_features: int):
    seed = cfg["seed"]
    max_rows = cfg.get("svm_max_rows")
    if name == "svm" and max_rows and len(X_train) > max_rows:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=max_rows,
                                                  stratify=y_train, random_state=seed)
    pipe = build_pipeline(make_model(name, seed), k=cfg["features"]["k"],
                          n_features=n_features, method=cfg["features"]["method"], seed=seed)
    fit_params = {}
    if name == "xgb":
        fit_params["clf__sample_weight"] = compute_sample_weight("balanced", y_train)
    pipe.fit(X_train, y_train, **fit_params)
    return pipe, len(X_train)


def train_all(df: pd.DataFrame, cfg: dict, out_root: Path | None = None) -> dict:
    out_root = out_root or models_dir()
    X_train, X_test, y_train, y_test, classes = split(df, cfg)
    feature_names = list(X_train.columns)
    print("Class balance:\n" + df["Label"].value_counts().to_string())

    results, pipes = {}, {}
    for name in cfg["models"]:
        print(f"\nTraining {name}...")
        t0 = time.perf_counter()
        pipe, n_rows = fit_one(name, X_train, y_train, cfg, len(feature_names))
        train_s = time.perf_counter() - t0

        metrics = evaluate(pipe, X_test, y_test, classes)
        metrics["train_seconds"] = train_s
        save_model(name, pipe, {
            "classes": classes,
            "features": feature_names,
            "selected_features": selected_features(pipe, feature_names),
            "metrics": metrics,
            "config": cfg,
            "train_rows": n_rows,
        }, root=out_root)
        results[name], pipes[name] = metrics, pipe
        print(f"  macro-F1={metrics['macro_f1']:.4f}  FPR={metrics['false_positive_rate']}  "
              f"latency={metrics['latency_ms_per_flow']:.2f}ms  train={train_s:.1f}s")

    best = max(results, key=lambda n: results[n]["macro_f1"])
    mark_best(best, root=out_root)

    # ---- Unknown-attack detector (U4): benign training flows only
    benign_idx = classes.index(BENIGN) if BENIGN in classes else None
    if benign_idx is not None:
        X_benign = X_train[y_train == benign_idx]
        det = AnomalyDetector(cfg.get("anomaly", {}).get("quantile", 0.005), cfg["seed"])
        det.fit(X_benign)
        flagged = det.is_anomaly(X_test)
        is_benign = y_test == benign_idx
        det_metrics = {
            "benign_false_alarm_rate": float(flagged[is_benign].mean()),
            "attack_flag_rate": float(flagged[~is_benign].mean()),
        }
        det.save(out_root / "anomaly", det_metrics)
        print(f"\nAnomaly detector: flags {det_metrics['attack_flag_rate']:.1%} of attacks, "
              f"{det_metrics['benign_false_alarm_rate']:.2%} of benign")

        # ---- Profiles for stories (U1), drift (U7) and fingerprints (U9)
        sel = selected_features(pipes[best], feature_names)
        y_train_lbl = pd.Series([classes[i] for i in y_train], index=X_train.index)
        save_profiles(out_root,
                      baseline=benign_baseline(X_benign),
                      reference=drift_reference(X_train),
                      fingerprints=fingerprints(X_train, y_train_lbl, sel[:8]))

    print(f"\nBest model: {best}")
    print(pd.DataFrame({n: {k: r[k] for k in ("macro_f1", "accuracy", "false_positive_rate",
                                                "latency_ms_per_flow", "train_seconds")}
                        for n, r in results.items()}).T.round(4).to_string())
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--rebuild", action="store_true", help="re-clean raw CSVs")
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = load_dataset(cfg, rebuild=args.rebuild)
    out = resolve(cfg.get("artifacts_dir", models_dir()))
    train_all(df, cfg, out_root=out)


if __name__ == "__main__":
    main()
