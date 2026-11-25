from nids.scoring.severity import load_severity_config, max_band, score

CFG = load_severity_config()


def test_ddos_high_confidence_is_critical():
    s, band = score("DDoS", 0.99, 1, CFG)
    assert band == "Critical" and 0 < s <= 1


def test_low_confidence_portscan_is_low():
    assert score("PortScan", 0.5, 1, CFG)[1] == "Low"


def test_burst_raises_severity():
    single = score("PortScan", 0.95, 1, CFG)
    burst = score("PortScan", 0.95, 500, CFG)
    assert burst[0] > single[0]
    assert max_band(single[1], burst[1]) == burst[1]


def test_unknown_label_uses_default():
    s, _ = score("SomethingNew", 1.0, 1, CFG)
    assert s == CFG["base"]["default"]


def test_max_band():
    assert max_band(None, "Low") == "Low"
    assert max_band("High", "Medium") == "High"
    assert max_band("Medium", "Critical") == "Critical"
