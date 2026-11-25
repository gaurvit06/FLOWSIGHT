"""Global SHAP importance for the served model -> reports/figures/shap_*.png

    python scripts/shap_global.py [--n 2000]
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from _common import FIGURES, base_args, setup, write_report

from nids.features.pipeline import selected_features
from nids.models.registry import load_model


def main() -> None:
    ap = base_args(__doc__)
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    cfg, _, _, X_test, _, _, _ = setup(args)
    pipe, meta = load_model()
    if meta["name"] == "svm":
        raise SystemExit("Global SHAP here needs a tree model (rf/xgb) as the served model.")
    classes = meta["classes"]
    names = selected_features(pipe, meta["features"])
    sample = X_test[meta["features"]].sample(min(args.n, len(X_test)), random_state=cfg["seed"])
    Xt = pd.DataFrame(pipe[:-1].transform(sample), columns=names)

    sv = shap.TreeExplainer(pipe[-1]).shap_values(Xt)
    sv = [np.asarray(s) for s in sv] if isinstance(sv, list) else \
        [np.asarray(sv)[:, :, i] for i in range(np.asarray(sv).shape[2])]

    # mean |SHAP| per feature per class (one bar chart, stacked by class)
    imp = pd.DataFrame({c: np.abs(s).mean(axis=0) for c, s in zip(classes, sv)}, index=names)
    imp = imp.loc[imp.sum(axis=1).sort_values().index]
    ax = imp.plot.barh(stacked=True, figsize=(9, 7))
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"Global feature importance by class ({meta['name']})")
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_global.png", dpi=150)
    plt.close()

    md = ["# Global SHAP importance", "", "![](figures/shap_global.png)", ""]
    for c, s in zip(classes, sv):
        if c == "Benign":
            continue
        shap.summary_plot(s, Xt, show=False, max_display=10)
        plt.title(f"{c}: how each feature pushes towards this class")
        plt.tight_layout()
        plt.savefig(FIGURES / f"shap_{c}.png", dpi=150)
        plt.close()
        top = imp[c].sort_values(ascending=False).index[:3]
        md += [f"## {c}", f"Top features: {', '.join(top)}", "",
               f"![](figures/shap_{c}.png)", ""]
    write_report("shap_global", "\n".join(md))


if __name__ == "__main__":
    main()
