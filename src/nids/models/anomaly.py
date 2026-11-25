"""Unknown-attack detector (U4).

An Isolation Forest trained on *benign* flows only. A classifier can only output classes it
was trained on; this model instead answers "does this look like normal traffic?". When the
classifier says Benign but this says "very abnormal", the flow becomes `Unknown-Anomaly`.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

UNKNOWN = "Unknown-Anomaly"


class AnomalyDetector:
    def __init__(self, quantile: float = 0.005, seed: int = 42):
        self.quantile = quantile    # share of benign training flows we accept as false alarms
        self.seed = seed
        self.model: IsolationForest | None = None
        self.threshold: float | None = None
        self.features: list[str] = []

    def fit(self, X_benign: pd.DataFrame) -> AnomalyDetector:
        self.features = list(X_benign.columns)
        self.model = IsolationForest(n_estimators=200, contamination="auto",
                                     random_state=self.seed, n_jobs=-1).fit(X_benign)
        scores = self.model.score_samples(X_benign)          # higher = more normal
        self.threshold = float(np.quantile(scores, self.quantile))
        # Range rule: Isolation Forest averages over all features, so one wildly extreme
        # feature can be diluted. Also remember the benign magnitude range per feature.
        self.log_max = np.log10(np.abs(X_benign).max().to_numpy(dtype=float) + 1)
        return self

    def range_score(self, X: pd.DataFrame) -> np.ndarray:
        """> 0 when some feature is more than 10x beyond anything seen in benign training."""
        logv = np.log10(np.abs(X[self.features].to_numpy(dtype=float)) + 1)
        return (logv - self.log_max).max(axis=1) - 1.0

    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """<= 0 = consistent with benign traffic; > 0 = anomalous (larger = stranger)."""
        forest = self.threshold - self.model.score_samples(X[self.features])
        return np.maximum(forest, self.range_score(X))

    def is_anomaly(self, X: pd.DataFrame) -> np.ndarray:
        return self.anomaly_score(X) > 0

    def save(self, d: Path, metrics: dict | None = None) -> None:
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, d / "detector.joblib")
        (d / "anomaly.json").write_text(json.dumps({
            "threshold": self.threshold, "quantile": self.quantile, "metrics": metrics or {}},
            indent=2), encoding="utf-8")

    @staticmethod
    def load(d: Path) -> AnomalyDetector | None:
        p = d / "detector.joblib"
        return joblib.load(p) if p.exists() else None
