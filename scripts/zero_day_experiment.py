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
    benign_odd = det.is_anomaly(benign_test)
    min_conf = cfg.get("anomaly", {}).get("min_benign_confidence", 0.0)
    fa_rates = []

    rows = []
    for held_out in (c for c in classes if c != "Benign"):
        h = classes.index(held_out)
        keep = y_train != h
        remaining = [i for i in range(len(classes)) if i != h]
        remap = {old: new for new, old in enumerate(remaining)}     # contiguous labels
        y_tr = np.array([remap[v] for v in y_train[keep]])
        pipe, _ = fit_one(args.model, X_train[keep], y_tr, cfg, X_train.shape[1])

        b = remap[benign]

        def unknown_rule(X, odd, pipe=pipe, b=b):
            p = pipe.predict_proba(X)
            said_benign = p.argmax(axis=1) == b
            return said_benign, said_benign & (odd | (p[:, b] < min_conf))

        X_zd = X_test[y_test == h]
        said_benign, flagged = unknown_rule(X_zd, det.is_anomaly(X_zd))
        _, benign_flagged = unknown_rule(benign_test, benign_odd)
        fa_rates.append(float(benign_flagged.mean()))
        by_clf = float((~said_benign).mean())
        by_det = float(flagged.mean())
        rows.append({
            "held-out attack": held_out, "test flows": len(X_zd),
            "caught by classifier (as another attack)": by_clf,
            "caught only by unknown-attack rule": by_det,
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
        (f"Classifier: {args.model}. Unknown-attack rule: the classifier says Benign but "
         f"either the anomaly detector (Isolation Forest + range rule, benign-only) fires or the "
         f"classifier is less than {min_conf:.0%} sure. It raises false alarms on "
         f"**{np.mean(fa_rates):.2%}** of benign test flows (mean over the runs)."), "",
        pct.to_markdown(), "",
        ("*Caught by classifier* means the model had never seen this attack but still called "
         "it an attack (a different type). *Caught only by unknown-attack rule* is what the "
         "rule adds on top."),
    ]))


if __name__ == "__main__":
    main()
