"""Quality gate: fails if the trained model in models/ is below the thresholds in the config.

Skipped until you've run `make train`. In CI, run it after training so a worse model
breaks the build instead of silently shipping.
"""

import pytest

from nids.config import load_config, models_dir
from nids.models.registry import load_model


@pytest.fixture(scope="module")
def best():
    if not (models_dir() / "best.json").exists():
        pytest.skip("No trained model in models/ — run `make train` first")
    return load_model()[1]


def test_macro_f1(best):
    t = load_config()["thresholds"]["macro_f1"]
    assert best["metrics"]["macro_f1"] >= t, f"macro-F1 {best['metrics']['macro_f1']:.3f} < {t}"


def test_every_attack_class_recall(best):
    t = load_config()["thresholds"]["attack_recall"]
    low = {c: m["recall"] for c, m in best["metrics"]["per_class"].items()
           if c != "Benign" and m["recall"] < t}
    assert not low, f"Attack classes below recall {t}: {low}"
