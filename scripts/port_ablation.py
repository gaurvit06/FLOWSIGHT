"""Does FlowSight depend on Destination Port? Train with and without it -> reports/port_ablation.md

    python scripts/port_ablation.py

SHAP shows Destination Port is the top feature for BruteForce and DoS. A real attacker can
change ports, so this measures how much each class's detection relies on that shortcut.
Both runs use the same training sample and are scored on the validation split (the test set
is not touched).
"""

from __future__ import annotations

import pandas as pd
from _common import base_args, setup, write_report
from sklearn.metrics import f1_score, precision_score, recall_score

from nids.features.pipeline import selected_features
from nids.models.train import fit_one
from nids.models.tune import subsample, val_split

PORT = "Destination Port"


def main() -> None:
    ap = base_args(__doc__)
    ap.add_argument("--model", default="xgb")
    args = ap.parse_args()
    cfg, _, X_train, _, y_train, _, classes = setup(args)
    X, y = subsample(X_train, y_train, args.max_rows, cfg["seed"])
    X_tr, X_va, y_tr, y_va = val_split(X, y, cfg)
    labels = list(range(len(classes)))

    rows, macro, used = {}, {}, {}
    for variant, cols in [("with port", list(X.columns)),
                          ("without port", [c for c in X.columns if c != PORT])]:
        pipe, _ = fit_one(args.model, X_tr[cols], y_tr, cfg, len(cols))
        pred = pipe.predict(X_va[cols])
        used[variant] = PORT in selected_features(pipe, cols)
        macro[variant] = f1_score(y_va, pred, average="macro")
        rec = recall_score(y_va, pred, labels=labels, average=None, zero_division=0)
        prec = precision_score(y_va, pred, labels=labels, average=None, zero_division=0)
        for c, r, p in zip(classes, rec, prec):
            rows.setdefault(c, {})[f"recall {variant}"] = r
            rows.setdefault(c, {})[f"precision {variant}"] = p
        print(f"{variant:<13} macro-F1={macro[variant]:.4f}  port selected={used[variant]}")

    t = pd.DataFrame(rows).T
    t["recall drop"] = t["recall with port"] - t["recall without port"]
    worst = t["recall drop"].idxmax()
    md = [
        "# Destination Port ablation", "",
        (f"Model {args.model}, same {len(X_tr):,}-row training sample, scored on a "
         f"{len(X_va):,}-row validation split. Destination Port was "
         f"{'selected' if used['with port'] else 'not selected'} by feature selection when "
         "available."), "",
        "| | macro-F1 |", "|---|---|",
        f"| with Destination Port | {macro['with port']:.4f} |",
        f"| without Destination Port | {macro['without port']:.4f} |", "",
        t.round(4).to_markdown(), "",
        (f"Largest recall drop: **{worst}** ({t.loc[worst, 'recall drop']:+.4f}). A small drop "
         "means the class is recognised from traffic behaviour, not just the port; a large "
         "drop means an attacker using a non-standard port could evade detection."),
    ]
    write_report("port_ablation", "\n".join(md))


if __name__ == "__main__":
    main()
