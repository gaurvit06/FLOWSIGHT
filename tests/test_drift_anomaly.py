import numpy as np
import pandas as pd

from nids.models.anomaly import AnomalyDetector
from nids.models.profiles import drift_reference, psi
from nids.monitoring import drift

rng = np.random.default_rng(0)
TRAIN = pd.DataFrame({"a": rng.lognormal(3, 1, 5000), "b": rng.normal(100, 10, 5000)})


def test_psi_same_distribution_is_small():
    ref = drift_reference(TRAIN)
    assert psi(ref["a"], rng.lognormal(3, 1, 2000)) < 0.05


def test_psi_shifted_distribution_is_large():
    ref = drift_reference(TRAIN)
    assert psi(ref["a"], rng.lognormal(5, 1, 2000)) > 0.2


def test_drift_check_flags_shifted_feature():
    ref = drift_reference(TRAIN)
    flows = [{"a": a, "b": b} for a, b in
             zip(rng.lognormal(5, 1, 300), rng.normal(100, 10, 300))]
    res = drift.check(flows, ref, ["a", "b"])
    assert res["status"] == "drift" and res["drifted"] == ["a"]


def test_drift_needs_enough_flows():
    assert drift.check([{"a": 1.0}] * 10, drift_reference(TRAIN), ["a"])["status"] \
        == "insufficient_data"


def test_anomaly_detector_flags_extremes_not_normal():
    det = AnomalyDetector(quantile=0.005, seed=0).fit(TRAIN)
    normal = TRAIN.sample(1000, random_state=1)
    assert det.is_anomaly(normal).mean() < 0.03
    extreme = pd.DataFrame({"a": [1e7], "b": [100.0]})       # one wildly extreme feature
    assert det.is_anomaly(extreme)[0]
