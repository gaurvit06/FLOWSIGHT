"""Class-imbalance ablation: nothing vs class weights vs SMOTE -> reports/imbalance.md

    python scripts/imbalance_ablation.py

SMOTE is applied *inside* the training split only (imblearn pipeline), never before splitting.
"""

from __future__ import annotations

import pandas as pd
from _common import base_args, setup, write_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

from nids.features.pipeline import build_pipeline
from nids.models.factory import make_model
from nids.models.tune import subsample, val_split


def main() -> None:
    args = base_args(__doc__).parse_args()
    cfg, _, X_train, _, y_train, _, classes = setup(args)
    X, y = subsample(X_train, y_train, args.max_rows, cfg["seed"])
    X_tr, X_va, y_tr, y_va = val_split(X, y, cfg)
    seed, k = cfg["seed"], cfg["features"]["k"]

    rows = {}
    for strategy in ("none", "class weights", "SMOTE"):
        pipe = build_pipeline(make_model("xgb", seed), k=k, n_features=X.shape[1], seed=seed)
        fit = {}
        if strategy == "class weights":
            fit["clf__sample_weight"] = compute_sample_weight("balanced", y_tr)
        if strategy == "SMOTE":
            pipe = ImbPipeline([pipe.steps[0], pipe.steps[1],
                                ("smote", SMOTE(random_state=seed, k_neighbors=3)),
                                pipe.steps[2]])
        pipe.fit(X_tr, y_tr, **fit)
        pred = pipe.predict(X_va)
        rec = recall_score(y_va, pred, labels=range(len(classes)), average=None,
                           zero_division=0)
        rows[strategy] = {"macro-F1": f1_score(y_va, pred, average="macro"),
                          **{f"recall {c}": r for c, r in zip(classes, rec)}}
        print(strategy, f"macro-F1={rows[strategy]['macro-F1']:.4f}")

    table = pd.DataFrame(rows).T.round(4)
    write_report("imbalance", "# Class-imbalance ablation (XGBoost, validation split)\n\n"
                 + table.to_markdown() + "\n\nLook at the smallest classes' recall, not "
                 "macro-F1 alone.\n")


if __name__ == "__main__":
    main()
