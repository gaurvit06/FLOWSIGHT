"""Per-prediction explanations with SHAP.

For tree models (RF, XGBoost) TreeExplainer is exact and fast enough to explain every flow.
The linear SVM is not explained per-flow here; extend with shap.LinearExplainer if needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nids.features.pipeline import selected_features


class FlowExplainer:
    def __init__(self, pipe, feature_names: list[str]):
        self.pre = pipe[:-1]                       # scale + select
        self.clf = pipe[-1]
        self.names = selected_features(pipe, feature_names)
        self._explainer = None
        if type(self.clf).__name__ in {"RandomForestClassifier", "XGBClassifier"}:
            import shap  # imported lazily: slow import, optional for SVM
            self._explainer = shap.TreeExplainer(self.clf)

    @property
    def available(self) -> bool:
        return self._explainer is not None

    def explain(self, X_row: pd.DataFrame, class_idx: int, top_k: int = 5) -> list[dict]:
        """Top features pushing this single flow towards the predicted class."""
        if not self.available:
            return []
        Xt = self.pre.transform(X_row)
        sv = self._explainer.shap_values(Xt)
        # SHAP's return shape differs between versions and model types
        if isinstance(sv, list):                 # list[n_classes] of (n, features)
            vals = np.asarray(sv[class_idx])[0]
        else:
            sv = np.asarray(sv)
            vals = sv[0, :, class_idx] if sv.ndim == 3 else sv[0]
        order = np.argsort(-np.abs(vals))[:top_k]
        return [
            {
                "feature": self.names[i],
                "value": float(X_row.iloc[0][self.names[i]]),   # raw, unscaled value
                "contribution": float(vals[i]),
            }
            for i in order
        ]
