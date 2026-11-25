"""Generate a small synthetic CICIDS-style CSV so the whole pipeline runs without the
real dataset. The numbers are made up — use it only for smoke tests and demos of the
plumbing, never for results in your report.

    python scripts/make_sample_data.py [--rows 30000] [--out data/raw/sample_synthetic.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Rough behavioural profile per raw CICIDS label (means; noise is added on top)
PROFILES = {
    #               share  ports            dur_us   fwd  bwd  size  syn  rst  ack  rate
    "BENIGN":      (0.70, [80, 443, 53],    8e5,     12,  10,  550, 0.1, 0.05, 0.9, 50),
    "DDoS":        (0.08, [80],             2e4,     4,   0.5, 60,  0.9, 0.0, 0.2, 40000),
    "DoS Hulk":    (0.06, [80],             6e4,     6,   4,   300, 0.6, 0.1, 0.6, 8000),
    "PortScan":    (0.08, None,             60,      1,   1,   2,   1.0, 0.9, 0.0, 30000),
    "SSH-Patator": (0.035, [22],            3e6,     20,  25,  90,  0.3, 0.1, 0.9, 15),
    "FTP-Patator": (0.035, [21],            2e6,     10,  12,  25,  0.3, 0.2, 0.9, 10),
    "Bot":         (0.02, [8080],           4e5,     5,   4,   200, 0.2, 0.0, 0.8, 30),
}


def generate(n: int, seed: int = 42, shift: float = 1.0) -> pd.DataFrame:
    """shift != 1 scales rates/durations to imitate a different network (drift demo)."""
    rng = np.random.default_rng(seed)
    parts = []
    for label, (share, ports, dur, fwd, bwd, size, syn, rst, ack, rate) in PROFILES.items():
        m = int(n * share)
        ln = lambda mean, sigma=0.6, m=m: rng.lognormal(np.log(max(mean, 1e-3)), sigma, m)
        fwd_p = np.maximum(1, np.round(ln(fwd))).astype(int)
        bwd_p = np.round(ln(bwd + 0.01)).astype(int)
        pkt = ln(size, 0.5)
        duration = ln(dur * shift, 0.9)
        port = (rng.integers(1, 65535, m) if ports is None else rng.choice(ports, m))
        df = pd.DataFrame({
            "Flow ID": [f"flow-{label}-{i}" for i in range(m)],
            " Source IP": rng.choice(["10.0.0.5", "172.16.0.1", "192.168.10.50"], m),
            " Source Port": rng.integers(1024, 65535, m),
            " Destination IP": "192.168.10.50",
            " Destination Port": port,
            " Timestamp": "2017-07-07 15:00:00",
            " Flow Duration": duration,
            " Total Fwd Packets": fwd_p,
            " Total Backward Packets": bwd_p,
            "Total Length of Fwd Packets": fwd_p * pkt,
            " Fwd Packet Length Mean": pkt,
            " Bwd Packet Length Mean": ln(size * 1.5, 0.7) * (bwd_p > 0),
            "Flow Bytes/s": (fwd_p * pkt) / (duration / 1e6),
            " Flow Packets/s": ln(rate / shift, 0.5),
            " Flow IAT Mean": duration / np.maximum(fwd_p + bwd_p - 1, 1),
            " SYN Flag Count": rng.binomial(1, syn, m),
            " RST Flag Count": rng.binomial(1, rst, m),
            " ACK Flag Count": rng.binomial(1, ack, m),
            " PSH Flag Count": rng.binomial(1, 0.4, m),
            " Average Packet Size": pkt * rng.uniform(0.9, 1.1, m),
            "Init_Win_bytes_forward": rng.choice([-1, 229, 8192, 29200, 65535], m),
            " Constant Column": 0,
            " Label": label,
        })
        parts.append(df)
    df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)
    # Reproduce real-data mess: inf rates from zero duration, and duplicate rows
    idx = rng.choice(len(df), size=max(1, n // 500), replace=False)
    df.iloc[idx, df.columns.get_loc("Flow Bytes/s")] = np.inf
    df = pd.concat([df, df.sample(n // 50, random_state=seed)], ignore_index=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=30000)
    ap.add_argument("--out", default=str(ROOT / "data" / "raw" / "sample_synthetic.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shift", type=float, default=1.0,
                    help="e.g. 3 = a different-looking network, for drift/cross-dataset demos")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.rows, args.seed, args.shift)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} synthetic rows to {out}")
    print("Remember to delete it (and data/processed/) before training on the real dataset.")


if __name__ == "__main__":
    main()
