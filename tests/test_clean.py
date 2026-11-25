import numpy as np

from nids.data.clean import LABEL_COL, LEAKY_COLUMNS, map_label


def test_column_names_stripped(clean_df):
    assert all(c == c.strip() for c in clean_df.columns)


def test_leaky_columns_dropped(clean_df):
    assert not (set(clean_df.columns) & LEAKY_COLUMNS)


def test_no_inf_or_nan(clean_df):
    feats = clean_df.drop(columns=[LABEL_COL])
    assert np.isfinite(feats.to_numpy()).all()


def test_no_duplicates(clean_df):
    assert not clean_df.duplicated().any()


def test_constant_columns_dropped(clean_df):
    assert "Constant Column" not in clean_df.columns


def test_labels_mapped_and_other_dropped(clean_df):
    assert set(clean_df[LABEL_COL]) <= {"Benign", "DDoS", "DoS", "PortScan", "BruteForce"}


def test_map_label_cases():
    assert map_label(" BENIGN ") == "Benign"
    assert map_label("DoS slowloris") == "DoS"
    assert map_label("SSH-Patator") == "BruteForce"
    assert map_label("Web Attack \x96 XSS") == "Other"
