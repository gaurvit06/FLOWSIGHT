import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def raw_df():
    from make_sample_data import generate
    return generate(6000, seed=1)


@pytest.fixture(scope="session")
def clean_df(raw_df):
    from nids.data.clean import clean
    return clean(raw_df)


@pytest.fixture(scope="session")
def tiny_cfg():
    return {
        "seed": 0,
        "data": {"test_size": 0.25},
        "features": {"k": 8, "method": "f_classif"},
        "models": ["rf", "xgb"],
        "svm_max_rows": 2000,
    }


@pytest.fixture(scope="session")
def trained_dir(clean_df, tiny_cfg, tmp_path_factory):
    """Train small models once into a temp dir, shared by API/pipeline tests."""
    from nids.models.train import train_all
    out = tmp_path_factory.mktemp("models")
    train_all(clean_df, tiny_cfg, out_root=out)
    return out
