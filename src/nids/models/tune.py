"""Cross-validation and hyperparameter search (Phase 2).

Rule: every modelling decision is made with cross-validation or a validation split carved
out of the *training* data. The held-out test set from `train.split` is touched only by
the final evaluation in train.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from nids.features.pipeline import build_pipeline
from nids.models.factory import make_model
from nids.models.train import fit_one

PARAM_SPACES = {
    "rf": {
        "clf__n_estimators": randint(100, 400),
        "clf__max_depth": [None, 10, 20, 30],
        "clf__min_samples_leaf": randint(1, 5),
        "select__k": [10, 15, 20, 30],
    },
    "xgb": {
        "clf__n_estimators": randint(100, 500),
        "clf__max_depth": randint(4, 12),
        "clf__learning_rate": uniform(0.03, 0.27),
        "clf__subsample": uniform(0.7, 0.3),
        "clf__colsample_bytree": uniform(0.6, 0.4),
        "select__k": [10, 15, 20, 30],
    },
}


def subsample(X: pd.DataFrame, y: np.ndarray, max_rows: int | None, seed: int):
    if max_rows and len(X) > max_rows:
        X, _, y, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=seed)
    return X, y


def val_split(X_train: pd.DataFrame, y_train: np.ndarray, cfg: dict, size: float = 0.2):
    return train_test_split(X_train, y_train, test_size=size, stratify=y_train,
                            random_state=cfg["seed"])


def fold_metrics(y_true, y_pred, classes: list[str]) -> dict:
    labels = list(range(len(classes)))
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    attack_rec = [r for c, r in zip(classes, rec) if c != "Benign"]
    out = {"macro_f1": f1_score(y_true, y_pred, average="macro"),
           "min_attack_recall": float(min(attack_rec)) if attack_rec else float("nan")}
    if "Benign" in classes:
        b = classes.index("Benign")
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        out["fpr"] = float((cm[b].sum() - cm[b, b]) / max(cm[b].sum(), 1))
    return out


def cv_evaluate(X: pd.DataFrame, y: np.ndarray, classes: list[str], cfg: dict,
                names: list[str], n_splits: int = 5) -> dict:
    """Stratified k-fold on training data -> {model: {metric: [per-fold values]}}."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])
    results = {n: {} for n in names}
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        for n in names:
            pipe, _ = fit_one(n, X.iloc[tr], y[tr], cfg, X.shape[1])
            m = fold_metrics(y[va], pipe.predict(X.iloc[va]), classes)
            for k, v in m.items():
                results[n].setdefault(k, []).append(v)
        print(f"  fold {fold}/{n_splits} done")
    return results


def random_search(name: str, X: pd.DataFrame, y: np.ndarray, cfg: dict, n_iter: int = 20,
                  cv: int = 3):
    pipe = build_pipeline(make_model(name, cfg["seed"]), k=cfg["features"]["k"],
                          n_features=X.shape[1], method=cfg["features"]["method"],
                          seed=cfg["seed"])
    search = RandomizedSearchCV(pipe, PARAM_SPACES[name], n_iter=n_iter, cv=cv,
                                scoring="f1_macro", random_state=cfg["seed"], n_jobs=1,
                                verbose=1)
    fit_params = {"clf__sample_weight": compute_sample_weight("balanced", y)} \
        if name == "xgb" else {}
    search.fit(X, y, **fit_params)
    return search.best_params_, float(search.best_score_)


def mean_std_table(results: dict) -> pd.DataFrame:
    rows = {}
    for n, metrics in results.items():
        rows[n] = {k: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in metrics.items()}
    return pd.DataFrame(rows).T
