"""Leave-one-attack-out experiment for the unknown-attack detector -> reports/zero_day.md

For each attack class A:
  1. train the classifier WITHOUT any A flows (A is a "zero-day")
  2. send the test-set A flows through classifier + anomaly detector
  3. report how many are caught: by the classifier (as some other attack), by the anomaly
     detector (classifier said Benign), or missed.

    python scripts/zero_day_experiment.py [--model xgb]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _common import base_args, setup, write_report

from nids.models.anomaly import AnomalyDetector
from nids.models.train import fit_one
from nids.models.tune import subsample


def main() -> None:
    ap = base_args(__doc__)
    ap.add_argument("--model", default="xgb")
    args = ap.parse_args()
    cfg, _, X_train, X_test, y_train, y_test, classes = setup(args)
    X_train, y_train = subsample(X_train, y_train, args.max_rows, cfg["seed"])
    benign = classes.index("Benign")

    det = AnomalyDetector(cfg.get("anomaly", {}).get("quantile", 0.005), cfg["seed"])
    det.fit(X_train[y_train == benign])
    benign_test = X_test[y_test == benign]
    benign_fa = float(det.is_anomaly(benign_test).mean())

    rows = []
    for held_out in (c for c in classes if c != "Benign"):
        h = classes.index(held_out)
        keep = y_train != h
        remaining = [i for i in range(len(classes)) if i != h]
        remap = {old: new for new, old in enumerate(remaining)}     # contiguous labels
        y_tr = np.array([remap[v] for v in y_train[keep]])
        pipe, _ = fit_one(args.model, X_train[keep], y_tr, cfg, X_train.shape[1])

        X_zd = X_test[y_test == h]
        pred = pipe.predict(X_zd)
        said_benign = pred == remap[benign]
        anomalous = det.is_anomaly(X_zd)
        by_clf = float((~said_benign).mean())
        by_det = float((said_benign & anomalous).mean())
        rows.append({
            "held-out attack": held_out, "test flows": len(X_zd),
            "caught by classifier (as another attack)": by_clf,
            "caught only by anomaly detector": by_det,
            "total caught": by_clf + by_det,
            "missed": 1 - by_clf - by_det,
        })
        print(f"{held_out:<12} classifier {by_clf:.1%} + detector {by_det:.1%} "
              f"= {by_clf + by_det:.1%}")

    t = pd.DataFrame(rows).set_index("held-out attack")
    pct = t.drop(columns="test flows").map(lambda v: f"{v:.1%}")
    pct.insert(0, "test flows", t["test flows"])
    write_report("zero_day", "\n".join([
        "# Zero-day experiment (leave one attack out)", "",
        (f"Classifier: {args.model}. The anomaly detector is trained on benign flows only and "
        f"raises false alarms on **{benign_fa:.2%}** of benign test flows."), "",
        pct.to_markdown(), "",
        ("*Caught by classifier* means the model had never seen this attack but still called "
        "it an attack (a different type). *Caught only by anomaly detector* is what the "
        "unknown-attack detector adds on top."),
    ]))


if __name__ == "__main__":
    main()
