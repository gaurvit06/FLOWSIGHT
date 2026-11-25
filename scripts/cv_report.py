"""Stratified 5-fold CV for all models (+ optional hyperparameter search) -> reports/results.md

    python scripts/cv_report.py [--folds 5] [--tune --n-iter 20]
"""

from __future__ import annotations

from _common import base_args, setup, write_report

from nids.models.tune import cv_evaluate, mean_std_table, random_search, subsample


def main() -> None:
    ap = base_args(__doc__)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--n-iter", type=int, default=20)
    args = ap.parse_args()
    cfg, _, X_train, _, y_train, _, classes = setup(args)
    X, y = subsample(X_train, y_train, args.max_rows, cfg["seed"])

    print(f"Cross-validating {cfg['models']} on {len(X):,} training rows...")
    res = cv_evaluate(X, y, classes, cfg, cfg["models"], args.folds)
    table = mean_std_table(res)
    print(table.to_string())

    md = ["# Cross-validated results", "",
          (f"Stratified {args.folds}-fold CV on {len(X):,} training rows "
          "(held-out test set not used). Mean ± std over folds."), "",
          table.to_markdown(), ""]

    if args.tune:
        md += ["## Hyperparameter search (RandomizedSearchCV, 3-fold, macro-F1)", ""]
        for name in ("rf", "xgb"):
            if name in cfg["models"]:
                params, score = random_search(name, X, y, cfg, args.n_iter)
                print(f"{name}: best CV macro-F1 {score:.4f} with {params}")
                md += [f"**{name}**: macro-F1 {score:.4f}", "", "```", str(params), "```", ""]
        md += ["Copy the best parameters into `src/nids/models/factory.py` and retrain."]
    write_report("results", "\n".join(md))


if __name__ == "__main__":
    main()
