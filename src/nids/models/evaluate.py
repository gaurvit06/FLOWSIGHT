"""Metrics that matter for intrusion detection (accuracy alone is misleading)."""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

BENIGN = "Benign"


def evaluate(pipe, X_test, y_test_enc: np.ndarray, classes: list[str]) -> dict:
    y_pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)
    labels = list(range(len(classes)))

    report = classification_report(
        y_test_enc, y_pred, labels=labels, target_names=classes,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test_enc, y_pred, labels=labels)

    # False positive rate: benign flows wrongly flagged as any attack
    fpr = None
    if BENIGN in classes:
        b = classes.index(BENIGN)
        benign_total = cm[b].sum()
        fpr = float((benign_total - cm[b, b]) / benign_total) if benign_total else None

    try:
        auc = float(roc_auc_score(y_test_enc, proba, multi_class="ovr", labels=labels))
    except ValueError:
        auc = None

    # Inference latency per flow (single-row predictions, like live traffic)
    sample = X_test.iloc[:200]
    t0 = time.perf_counter()
    for i in range(len(sample)):
        pipe.predict(sample.iloc[[i]])
    latency_ms = (time.perf_counter() - t0) / max(len(sample), 1) * 1000

    return {
        "accuracy": float(accuracy_score(y_test_enc, y_pred)),
        "macro_f1": float(f1_score(y_test_enc, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test_enc, y_pred, average="weighted")),
        "roc_auc_ovr": auc,
        "false_positive_rate": fpr,
        "per_class": {
            c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score", "support")}
            for c in classes
        },
        "confusion_matrix": cm.tolist(),
        "latency_ms_per_flow": float(latency_ms),
    }
