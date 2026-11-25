"""The ONE preprocessing pipeline used for both training and serving.

Scaling and feature selection live inside the sklearn Pipeline, so they are fitted on
training data only (no leakage) and the API applies exactly the same transforms.
"""

from __future__ import annotations

from functools import partial

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(clf, k: int, n_features: int, method: str = "f_classif",
                   seed: int = 42) -> Pipeline:
    if method == "mutual_info":
        score = partial(mutual_info_classif, random_state=seed)
    elif method == "f_classif":
        score = f_classif
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    return Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(score, k=min(k, n_features))),
        ("clf", clf),
    ])


def selected_features(pipe: Pipeline, feature_names: list[str]) -> list[str]:
    mask = pipe.named_steps["select"].get_support()
    return [f for f, keep in zip(feature_names, mask) if keep]
