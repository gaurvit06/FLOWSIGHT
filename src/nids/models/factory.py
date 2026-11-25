"""Model definitions. Hyperparameters here are sensible starting points — tune them in Phase 3."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

TREE_MODELS = {"rf", "xgb"}


def make_model(name: str, seed: int = 42):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        )
    if name == "xgb":
        # class imbalance is handled with sample_weight at fit time (see train.py)
        return XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist",
            eval_metric="mlogloss", n_jobs=-1, random_state=seed,
        )
    if name == "svm":
        # A kernel (RBF) SVM is O(n^2) and won't finish on millions of flows, so we use a
        # linear SVM and calibrate it to get probabilities. Worth discussing in your report.
        return CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", dual=False, max_iter=5000, random_state=seed),
            cv=3,
        )
    raise ValueError(f"Unknown model: {name}")
