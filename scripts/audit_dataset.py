"""Dataset audit and provenance record -> reports/DATASET.md

For every raw CSV: checksum, rows, raw label counts, inf/NaN cells and duplicates. Then what
cleaning does to the whole dataset, and the final class balance. Run it once after
fetching the data; the numbers go straight into your report's dataset section.

    python scripts/audit_dataset.py
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from _common import REPORTS, write_report

from nids.config import load_config, resolve
from nids.data.clean import LABEL_COL, LEAKY_COLUMNS, clean, map_label


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    cfg = load_config()
    raw_dir = resolve(cfg["data"]["raw_dir"])
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSVs in {raw_dir}. Run scripts/fetch_dataset.py first.")
    synthetic = any(f.name == "sample_synthetic.csv" for f in files)

    per_file, frames = [], []
    for f in files:
        df = pd.read_csv(f, low_memory=False, encoding="latin-1")
        df.columns = df.columns.str.strip()
        num = df.select_dtypes("number")
        per_file.append({
            "file": f.name,
            "MB": round(f.stat().st_size / 1e6, 1),
            "rows": len(df),
            "inf cells": int(np.isinf(num.to_numpy()).sum()),
            "NaN cells": int(df.isna().sum().sum()),
            "duplicate rows": int(df.duplicated().sum()),
            "labels": ", ".join(f"{k} ({v:,})" for k, v in
                                df[LABEL_COL].astype(str).str.strip().value_counts().items()),
            "sha256": sha256(f)[:16] + "…",
        })
        frames.append(df)
        print(f"  audited {f.name}: {len(df):,} rows")

    raw = pd.concat(frames, ignore_index=True)
    mapped = raw[LABEL_COL].map(map_label).value_counts()
    cleaned = clean(raw, drop_other=cfg["data"].get("drop_other", True))
    leaky = sorted(c for c in raw.columns if c in LEAKY_COLUMNS)

    md = [
        "# Dataset audit & provenance", "",
        (f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by "
        "`scripts/audit_dataset.py`."), "",
    ]
    if synthetic:
        md += [("> ⚠️ **This audit includes the synthetic sample CSV.** Its numbers are made up; "
               "fetch the real dataset before using this in a report."), ""]
    md += [
        ("**Source:** CIC-IDS2017, Canadian Institute for Cybersecurity, University of New "
        "Brunswick (unb.ca/cic/datasets/ids-2017.html), MachineLearningCSV files. "
        "Full checksums: `data/raw/CHECKSUMS.sha256`."), "",
        "## Files", "", pd.DataFrame(per_file).to_markdown(index=False), "",
        "## Label mapping (raw → FlowSight classes)", "",
        mapped.rename("flows").to_frame().to_markdown(), "",
        "## Cleaning", "",
        "| step | effect |", "|---|---|",
        f"| raw rows | {len(raw):,} |",
        f"| identifier columns dropped (leakage) | {', '.join(leaky) or 'none'} |",
        f"| classes dropped (`drop_other`) | {int(mapped.get('Other', 0)):,} rows |",
        f"| rows after cleaning (inf/NaN dropped, duplicates removed) | {len(cleaned):,} |",
        (f"| removed in total | {len(raw) - len(cleaned):,} "
        f"({1 - len(cleaned) / len(raw):.1%}) |"),
        f"| features kept | {cleaned.shape[1] - 1} |", "",
        "## Final class balance", "",
        cleaned[LABEL_COL].value_counts().rename("flows").to_frame()
        .assign(share=lambda d: (d["flows"] / d["flows"].sum()).map("{:.3%}".format))
        .to_markdown(), "",
    ]
    REPORTS.mkdir(exist_ok=True)
    write_report("DATASET", "\n".join(md))


if __name__ == "__main__":
    main()
