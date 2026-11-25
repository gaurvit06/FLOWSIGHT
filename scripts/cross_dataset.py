"""Generalization report card (U8): evaluate the served model on a *different* dataset.

    python scripts/cross_dataset.py --test-dir data/cse-cic-ids2018 [--max-rows 200000]

Typical use: train on CICIDS2017 (data/raw), test on CSE-CIC-IDS2018 CSVs. Column names are
cleaned the same way; features missing from the new dataset are reported and the run stops.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from _common import REPORTS, write_report
from sklearn.metrics import f1_score, recall_score

from nids.config import resolve
from nids.data.clean import LABEL_COL, clean
from nids.data.columns import RENAMES
from nids.data.load import load_raw
from nids.models.registry import load_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--max-rows", type=int, default=200_000)
    ap.add_argument("--name", default="second dataset")
    args = ap.parse_args()

    pipe, meta = load_model()
    raw = load_raw(resolve(args.test_dir))
    raw.columns = raw.columns.str.strip()
    df = clean(raw.rename(columns=RENAMES))
    missing = [f for f in meta["features"] if f not in df.columns]
    if missing:
        raise SystemExit(f"{len(missing)} model features are missing from the new dataset, "
                         f"e.g. {missing[:8]}. Add them to RENAMES.")
    df = df[df[LABEL_COL].isin(meta["classes"])]
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=0)

    classes = meta["classes"]
    y = df[LABEL_COL].map(classes.index).to_numpy()
    pred = pipe.predict(df[meta["features"]])
    rec = recall_score(y, pred, labels=range(len(classes)), average=None, zero_division=0)
    present = np.isin(range(len(classes)), np.unique(y))

    rows = []
    for i, c in enumerate(classes):
        same = meta["metrics"]["per_class"][c]["recall"]
        other = rec[i] if present[i] else float("nan")
        rows.append({"class": c, "recall (same dataset)": same,
                     f"recall ({args.name})": other,
                     "drop": same - other if present[i] else float("nan"),
                     f"flows in {args.name}": int((y == i).sum())})
    t = pd.DataFrame(rows).set_index("class")
    f1_same = meta["metrics"]["macro_f1"]
    f1_other = f1_score(y, pred, labels=np.where(present)[0], average="macro")

    md = ["# Generalization report card", "",
          (f"Model **{meta['name']}**, trained on the original dataset, evaluated unchanged on "
          f"**{args.name}** ({len(df):,} flows)."), "",
          f"| | same dataset | {args.name} |", "|---|---|---|",
          f"| macro-F1 | {f1_same:.4f} | {f1_other:.4f} |", "",
          t.round(4).to_markdown(), "",
          "## Why it drops", "",
          ("*Write this yourself, it's one of the most valuable parts of the report.* "
          "Typical causes: a different network and traffic mix, different attack tools and "
          "settings, a different CICFlowMeter version computing features slightly "
          "differently, and labelling differences between the datasets.")]
    REPORTS.mkdir(exist_ok=True)
    write_report("cross_dataset", "\n".join(md))
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
