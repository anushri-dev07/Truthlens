def test_load_config_defaults():
    from truthlens.config import load_config

    cfg = load_config()
    assert "weights" in cfg
    assert abs(sum(cfg["weights"].values()) - 1.0) < 1e-6
    assert cfg["thresholds"]["ai"] > cfg["thresholds"]["real"]
    assert cfg["sample_frames"] >= 1


def test_config_override(tmp_path):
    from truthlens.config import load_config

    p = tmp_path / "c.yaml"
    p.write_text(
        "weights:\n  s1_frame: 0.9\nthresholds:\n  ai: 60.0\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg["weights"]["s1_frame"] == 0.9
    assert cfg["weights"]["s2_temporal"] == 0.25
    assert cfg["thresholds"]["ai"] == 60.0
