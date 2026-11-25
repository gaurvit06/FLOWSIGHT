"""Read raw CICIDS-style CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw(raw_dir: str | Path, sample_frac: float = 1.0, seed: int = 42) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files in {raw_dir}. Download CICIDS2017 (MachineLearningCVE CSVs) "
            "into it, or run `python scripts/make_sample_data.py` for a synthetic dataset."
        )
    frames = []
    for f in files:
        # CICIDS files contain some non-UTF8 bytes in the Web Attack labels
        df = pd.read_csv(f, low_memory=False, encoding="latin-1")
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=seed)
        frames.append(df)
        print(f"  loaded {f.name}: {len(df):,} rows")
    return pd.concat(frames, ignore_index=True)
