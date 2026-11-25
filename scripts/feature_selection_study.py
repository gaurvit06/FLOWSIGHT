"""Macro-F1 vs number of features for three selection methods -> reports/figures/

    python scripts/feature_selection_study.py

Rankings are computed on the training split and scored on a validation split carved from it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _common import FIGURES, base_args, setup, write_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from nids.models.tune import subsample, val_split


def rankings(X, y, seed):
    f, _ = f_classif(X, y)
    Xm, ym = subsample(X, y, 20_000, seed)       # mutual information is slow
    mi = mutual_info_classif(Xm, ym, random_state=seed)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed).fit(X, y)
    cols = np.array(X.columns)
    return {
        "ANOVA F (f_classif)": list(cols[np.argsort(-np.nan_to_num(f))]),
        "mutual information": list(cols[np.argsort(-mi)]),
        "RF importance": list(cols[np.argsort(-rf.feature_importances_)]),
    }


def main() -> None:
    args = base_args(__doc__).parse_args()
    cfg, _, X_train, _, y_train, _, _classes = setup(args)
    X, y = subsample(X_train, y_train, args.max_rows, cfg["seed"])
    X_tr, X_va, y_tr, y_va = val_split(X, y, cfg)
    ranks = rankings(X_tr, y_tr, cfg["seed"])

    n = X.shape[1]
    ks = sorted({k for k in (3, 5, 10, 15, 20, 30, 40, n) if k <= n})
    scores = {}
    for method, order in ranks.items():
        scores[method] = []
        for k in ks:
            cols = order[:k]
            clf = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                                tree_method="hist", n_jobs=-1, random_state=cfg["seed"])
            clf.fit(X_tr[cols], y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
            scores[method].append(f1_score(y_va, clf.predict(X_va[cols]), average="macro"))
        print(method, [f"{s:.3f}" for s in scores[method]])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, s in scores.items():
        ax.plot(ks, s, marker="o", label=method)
    ax.set_xlabel("number of features (k)")
    ax.set_ylabel("validation macro-F1")
    ax.set_title("Feature selection study")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_selection.png", dpi=150)

    md = ["# Feature selection study", "", "![](figures/feature_selection.png)", "",
          "| k | " + " | ".join(scores) + " |", "|---|" + "---|" * len(scores)]
    for i, k in enumerate(ks):
        md.append(f"| {k} | " + " | ".join(f"{scores[m][i]:.4f}" for m in scores) + " |")
    md += ["", "Top 10 by RF importance: " + ", ".join(ranks["RF importance"][:10])]
    write_report("feature_selection", "\n".join(md))


if __name__ == "__main__":
    main()
