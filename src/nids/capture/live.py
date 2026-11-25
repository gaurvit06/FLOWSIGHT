"""Live capture bridge (Phase 10, stretch goal).

CICFlowMeter turns live packets into exactly the flow features this model was trained on, and
writes them as CSV rows. This script tails its output folder and sends each new flow to the API.

    1. Run CICFlowMeter in real-time mode on your lab interface, writing to e.g. data/live/
    2. Start the API:          uvicorn nids.api.main:app --port 8000
    3. Run this bridge:        python -m nids.capture.live --watch data/live

LAB ONLY: generate attack traffic (nmap, hping3, hydra) only between VMs you own, on an
isolated host-only network. Never point attack tools at networks you don't own or have
written permission to test.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from nids.data.columns import RENAMES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", required=True, help="folder where CICFlowMeter writes CSVs")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--poll", type=float, default=1.0, help="seconds between checks")
    args = ap.parse_args()

    folder = Path(args.watch)
    folder.mkdir(parents=True, exist_ok=True)
    seen_rows: dict[Path, int] = {}
    client = httpx.Client(base_url=args.api, timeout=30)
    features = client.get("/model").json()["features"]
    print(f"Watching {folder} for CICFlowMeter CSVs... (Ctrl+C to stop)")

    while True:
        for csv in sorted(folder.glob("*.csv")):
            try:
                df = pd.read_csv(csv, low_memory=False, encoding="latin-1")
            except (pd.errors.EmptyDataError, PermissionError):
                continue                      # file is being written; try next poll
            start = seen_rows.get(csv, 0)
            new = df.iloc[start:]
            seen_rows[csv] = len(df)
            if new.empty:
                continue
            new.columns = new.columns.str.strip()
            new = new.rename(columns=RENAMES)
            missing = [f for f in features if f not in new.columns]
            if missing:
                raise SystemExit(f"CICFlowMeter output lacks features {missing[:8]}; extend "
                                 "RENAMES in src/nids/data/columns.py")
            X = new[features].apply(pd.to_numeric, errors="coerce")
            X = X.replace([np.inf, -np.inf], np.nan).dropna()
            flows = [{"features": r} for r in X.astype(float).to_dict("records")]
            if flows:
                preds = client.post("/predict/batch", json={"flows": flows}).json()
                for p in preds:
                    if p["is_attack"]:
                        print(f"[{p.get('severity')}] {p['story']}")
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
