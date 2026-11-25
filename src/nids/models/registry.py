"""Save/load trained models together with everything needed to reproduce and serve them."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from nids.config import models_dir


def save_model(name: str, pipe, metadata: dict, root: Path | None = None) -> Path:
    out = (root or models_dir()) / name
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.joblib")
    metadata = {**metadata, "name": name, "trained_at": datetime.now(timezone.utc).isoformat()}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def mark_best(name: str, root: Path | None = None) -> None:
    root = root or models_dir()
    (root / "best.json").write_text(json.dumps({"name": name}), encoding="utf-8")


def load_metadata(name: str, root: Path | None = None) -> dict:
    return json.loads(((root or models_dir()) / name / "metadata.json").read_text("utf-8"))


def best_name(root: Path | None = None) -> str:
    path = (root or models_dir()) / "best.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run training first (make train).")
    return json.loads(path.read_text("utf-8"))["name"]


def load_model(name: str | None = None, root: Path | None = None):
    """Return (pipeline, metadata). Defaults to the best model."""
    root = root or models_dir()
    name = name or best_name(root)
    return joblib.load(root / name / "model.joblib"), load_metadata(name, root)


def list_models(root: Path | None = None) -> list[dict]:
    root = root or models_dir()
    if not root.exists():
        return []
    return [load_metadata(p.name, root) for p in sorted(root.iterdir())
            if (p / "metadata.json").exists()]
