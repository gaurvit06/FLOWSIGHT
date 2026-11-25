"""The ONE preprocessing pipeline used for both training and serving.

Scaling and feature selection live inside the sklearn Pipeline, so they are fitted on
training data only (no leakage) and the API applies exactly the same transforms.

Feature selection methods:
- tree        (default) top-k by ExtraTrees importance. Captures non-linear signals such as
              Destination Port, and avoids picking k near-duplicate packet-size features.
- f_classif   ANOVA F-test: fast, but linear; on CIC-IDS2017 it drops Destination Port and
              keeps redundant features (BruteForce F1 0.49 vs 0.998 with `tree`, XGBoost).
- mutual_info slower, non-linear, univariate.
"""

from __future__ import annotations

from functools import partial

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, SelectorMixin, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TreeTopK(SelectorMixin, BaseEstimator):
    """Keep the k features ExtraTrees ranks most important.

    Unlike SelectFromModel, the mask is computed once at fit time and the forest is then
    discarded, so prediction costs nothing extra (SelectFromModel re-derives importances
    from the whole forest on every transform: ~35 ms per flow) and the saved model is small.
    """

    def __init__(self, k: int = 20, n_estimators: int = 50, random_state: int = 42):
        self.k = k
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        forest = ExtraTreesClassifier(n_estimators=self.n_estimators, n_jobs=-1,
                                      random_state=self.random_state).fit(X, y)
        self.importances_ = forest.feature_importances_
        self.mask_ = np.zeros(X.shape[1], dtype=bool)
        self.mask_[np.argsort(-self.importances_)[: self.k]] = True
        self.n_features_in_ = X.shape[1]
        return self

    def _get_support_mask(self):
        return self.mask_


def make_selector(method: str, k: int, seed: int = 42):
    if method == "tree":
        return TreeTopK(k=k, random_state=seed)
    if method == "mutual_info":
        return SelectKBest(partial(mutual_info_classif, random_state=seed), k=k)
    if method == "f_classif":
        return SelectKBest(f_classif, k=k)
    raise ValueError(f"Unknown feature selection method: {method}")


def build_pipeline(clf, k: int, n_features: int, method: str = "tree",
                   seed: int = 42) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("select", make_selector(method, min(k, n_features), seed)),
        ("clf", clf),
    ])


def selected_features(pipe: Pipeline, feature_names: list[str]) -> list[str]:
    mask = pipe.named_steps["select"].get_support()
    return [f for f, keep in zip(feature_names, mask) if keep]
