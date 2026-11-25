"""Tests for the validation split, dataset fetching and generated documents."""

import zipfile

from nids.models.registry import best_name, list_models, load_metadata


def test_split_is_train_val_test(trained_dir):
    meta = load_metadata("rf", trained_dir)
    s = meta["split_rows"]
    assert s["train"] > s["validation"] > 0 and s["test"] > 0
    assert "val_macro_f1" in meta["metrics"]


def test_best_model_chosen_on_validation(trained_dir):
    models = list_models(trained_dir)
    best = max(models, key=lambda m: m["metrics"]["val_macro_f1"])["name"]
    assert best_name(trained_dir) == best


def test_tree_selector_keeps_informative_features():
    import numpy as np

    from nids.features.pipeline import TreeTopK

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 6))
    y = (X[:, 2] + 0.5 * X[:, 4] > 0).astype(int)          # only features 2 and 4 matter
    sel = TreeTopK(k=2, random_state=0).fit(X, y)
    assert set(np.flatnonzero(sel.get_support())) == {2, 4}
    assert sel.transform(X).shape == (2000, 2)
    assert not hasattr(sel, "estimator_")                    # forest is not kept


def test_class_sample_weights_power():
    import numpy as np

    from nids.models.factory import class_sample_weights

    y = np.array([0] * 90 + [1] * 10)
    full, soft = class_sample_weights(y, 1.0), class_sample_weights(y, 0.5)
    assert full[-1] / full[0] == 9.0
    assert soft[-1] / soft[0] == 3.0


def test_fetch_extracts_only_csvs_and_flattens_paths(tmp_path, monkeypatch):
    import fetch_dataset as fd

    monkeypatch.setattr(fd, "RAW", tmp_path / "raw")
    z = tmp_path / "MachineLearningCSV.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("MachineLearningCVE/Monday.pcap_ISCX.csv", "a,Label\n1,BENIGN\n")
        zf.writestr("MachineLearningCVE/../../evil.csv", "a,Label\n1,BENIGN\n")
        zf.writestr("README.txt", "not data")
        zf.writestr("run.bat", "echo hi")
    written = fd.extract_csvs(z)
    names = sorted(p.name for p in written)
    assert names == ["Monday.pcap_ISCX.csv", "evil.csv"]
    assert all(p.parent == tmp_path / "raw" for p in written)       # no path traversal
    assert not (tmp_path / "raw" / "run.bat").exists()


def test_model_card_is_generated(trained_dir, monkeypatch, tmp_path):
    import _common
    import model_card

    monkeypatch.setenv("NIDS_MODELS_DIR", str(trained_dir))
    monkeypatch.setattr(_common, "REPORTS", tmp_path)
    model_card.main()
    text = (tmp_path / "MODEL_CARD.md").read_text("utf-8")
    assert "Model card" in text and "chosen on validation" in text
    assert "Limitations" in text
