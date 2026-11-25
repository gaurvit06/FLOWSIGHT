"""Shared setup for experiment scripts: load config + data + the standard split."""

from __future__ import annotations

import argparse
from pathlib import Path

from nids.config import ROOT, load_config
from nids.models.train import load_dataset, split

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def base_args(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-rows", type=int, default=200_000,
                    help="subsample training data for speed (0 = all)")
    return ap


def setup(args):
    cfg = load_config(args.config)
    df = load_dataset(cfg)
    X_train, X_test, y_train, y_test, classes = split(df, cfg)
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    return cfg, df, X_train, X_test, y_train, y_test, classes


def write_report(name: str, text: str) -> Path:
    p = REPORTS / f"{name}.md"
    p.write_text(text, encoding="utf-8")
    print(f"\nWrote {p}")
    return p
