"""Get the CICIDS2017 CSVs into data/raw/ and record their checksums.

The official download (cicresearch.ca, University of New Brunswick) needs a short
registration form, so download MachineLearningCSV.zip in your browser first, then:

    python scripts/fetch_dataset.py --zip "%USERPROFILE%\\Downloads\\MachineLearningCSV.zip"

or, if you have a direct link:

    python scripts/fetch_dataset.py --url <direct link to MachineLearningCSV.zip>

The zip is only read as data: CSV members are extracted, anything else is ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DOWNLOADS = ROOT / "data" / "downloads"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str) -> Path:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    out = DOWNLOADS / "MachineLearningCSV.zip"
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        if "html" in r.headers.get("content-type", ""):
            sys.exit("The link returned a web page, not the zip (probably the registration "
                     "form). Download it in your browser and use --zip instead.")
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(out, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="")
    print()
    return out


def extract_csvs(zip_path: Path) -> list[Path]:
    if not zipfile.is_zipfile(zip_path):
        sys.exit(f"{zip_path} is not a valid zip file.")
    RAW.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            name = Path(member.filename).name           # flatten folders, drop any path
            if member.is_dir() or not name.lower().endswith(".csv"):
                continue
            target = RAW / name
            with z.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            written.append(target)
            print(f"  extracted {name} ({target.stat().st_size / 1e6:,.0f} MB)")
    if not written:
        sys.exit("No CSV files found in the zip.")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", type=Path, help="MachineLearningCSV.zip you downloaded")
    src.add_argument("--url", help="direct download link")
    args = ap.parse_args()

    zip_path = download(args.url) if args.url else args.zip
    print(f"Archive: {zip_path}  sha256={sha256(zip_path)}")
    files = extract_csvs(zip_path)

    sample = RAW / "sample_synthetic.csv"
    if sample.exists():
        sample.unlink()
        print("  removed the synthetic sample CSV (it must never be mixed with real data)")
    stale = ROOT / "data" / "processed" / "flows.parquet"
    if stale.exists():
        stale.unlink()
        print("  removed cached processed data (will be rebuilt from the real CSVs)")

    lines = [f"{sha256(p)}  {p.name}" for p in sorted(files)]
    (RAW / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {len(files)} CSVs and data/raw/CHECKSUMS.sha256")
    print("Next: python scripts/audit_dataset.py  then  python -m nids.models.train --rebuild")


if __name__ == "__main__":
    main()
