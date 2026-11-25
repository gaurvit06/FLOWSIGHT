"""Paths and config loading shared by every module."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "train.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p: str | Path) -> Path:
    """Resolve a path from the config relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def models_dir() -> Path:
    """Where trained artifacts live. Override with NIDS_MODELS_DIR (used by tests)."""
    return Path(os.environ.get("NIDS_MODELS_DIR", ROOT / "models"))


def db_path() -> Path:
    """SQLite file for predictions/alerts. Override with NIDS_DB_PATH."""
    return Path(os.environ.get("NIDS_DB_PATH", ROOT / "data" / "nids.db"))
