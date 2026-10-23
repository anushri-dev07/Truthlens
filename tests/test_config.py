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


def test_default_config_is_independent_of_working_directory(tmp_path, monkeypatch):
    from truthlens.config import load_config

    monkeypatch.chdir(tmp_path)
    cfg = load_config()

    assert cfg["sample_frames"] == 2
    assert cfg["thresholds"]["ai"] == 58.0
    assert cfg["thresholds"]["real"] == 42.0


def test_malformed_config_fails_visibly(tmp_path):
    import pytest

    from truthlens.config import load_config

    path = tmp_path / "invalid.yaml"
    path.write_text("weights: [unterminated", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not load TruthLens configuration"):
        load_config(path)


def test_relative_model_paths_are_project_root_relative():
    from truthlens.config import PROJECT_ROOT, _project_path

    assert _project_path("artifacts/example.onnx") == PROJECT_ROOT / "artifacts/example.onnx"
