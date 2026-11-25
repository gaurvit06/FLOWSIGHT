"""Replay flows from the processed dataset into the API as if they were live traffic.

    python -m nids.capture.replay --n 500 --delay 0.02 --attack-frac 0.3

Start the API first (make serve). Watch the dashboard update while this runs.
Live capture (Phase 7) would replace this with CICFlowMeter/Scapy producing the same features.
"""

from __future__ import annotations

import argparse
import time

import httpx
import pandas as pd

from nids.config import load_config, resolve
from nids.data.clean import LABEL_COL


def sample_flows(n: int, attack_frac: float, seed: int) -> pd.DataFrame:
    cfg = load_config()
    df = pd.read_parquet(resolve(cfg["data"]["processed_path"]))
    attacks = df[df[LABEL_COL] != "Benign"]
    benign = df[df[LABEL_COL] == "Benign"]
    n_att = min(int(n * attack_frac), len(attacks))
    out = pd.concat([attacks.sample(n_att, random_state=seed),
                     benign.sample(min(n - n_att, len(benign)), random_state=seed)])
    return out.sample(frac=1, random_state=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--delay", type=float, default=0.02, help="seconds between flows")
    ap.add_argument("--attack-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    flows = sample_flows(args.n, args.attack_frac, args.seed)
    correct = 0
    with httpx.Client(base_url=args.api, timeout=30) as client:
        for i, (_, row) in enumerate(flows.iterrows(), 1):
            truth = row[LABEL_COL]
            features = row.drop(LABEL_COL).astype(float).to_dict()
            r = client.post("/predict", json={"features": features})
            r.raise_for_status()
            pred = r.json()
            correct += pred["label"] == truth
            if pred["is_attack"]:
                top = pred["explanation"][0]["feature"] if pred["explanation"] else "-"
                print(f"[{i:>4}] ALERT {pred['label']:<10} conf={pred['confidence']:.2f} "
                      f"truth={truth:<10} top-feature={top}")
            time.sleep(args.delay)
    print(f"\nReplayed {len(flows)} flows, accuracy vs ground truth: {correct / len(flows):.3f}")


if __name__ == "__main__":
    main()
