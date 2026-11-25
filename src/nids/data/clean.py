"""Cleaning rules. Every rule here exists because it changes results if skipped."""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_COL = "Label"

# Identifiers that let a model memorise *who* attacked instead of *how*.
LEAKY_COLUMNS = {
    "Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP",
    "Source Port", "Src Port", "Timestamp",
}


def map_label(raw: str) -> str:
    """Collapse the ~15 CICIDS labels into the classes this project models."""
    s = str(raw).strip()
    u = s.upper()
    if u == "BENIGN":
        return "Benign"
    if u == "DDOS":
        return "DDoS"
    if u.startswith("DOS"):
        return "DoS"
    if u == "PORTSCAN":
        return "PortScan"
    if u in {"FTP-PATATOR", "SSH-PATATOR"}:
        return "BruteForce"
    return "Other"  # Bot, Web Attack *, Infiltration, Heartbleed


def clean(df: pd.DataFrame, drop_other: bool = True) -> pd.DataFrame:
    df = df.copy()

    # 1. CICIDS column names have stray leading spaces (" Flow Duration")
    df.columns = df.columns.str.strip()
    if LABEL_COL not in df.columns:
        raise ValueError(f"No '{LABEL_COL}' column found. Columns: {list(df.columns)[:10]}...")

    # 2. Drop identifier columns (data leakage)
    df = df.drop(columns=[c for c in df.columns if c in LEAKY_COLUMNS])

    # 3. Map labels
    df[LABEL_COL] = df[LABEL_COL].map(map_label)
    if drop_other:
        df = df[df[LABEL_COL] != "Other"]

    # 4. Keep numeric features only, replace inf (from divide-by-zero rates) and drop NaN
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # 5. Drop constant columns (zero information, break some scalers)
    nunique = df[feature_cols].nunique()
    df = df.drop(columns=nunique[nunique <= 1].index.tolist())

    # 6. Drop exact duplicates — CICIDS has many, and they inflate test scores
    #    because identical rows end up in both train and test.
    df = df.drop_duplicates()

    # float32 halves memory on the full 2.8M-row dataset
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    df[feature_cols] = df[feature_cols].astype("float32")
    return df.reset_index(drop=True)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[LABEL_COL]), df[LABEL_COL]
