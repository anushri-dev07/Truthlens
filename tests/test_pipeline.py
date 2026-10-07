def test_pipeline_no_file():
    from truthlens.inference.pipeline import run_pipeline
    res = run_pipeline("no_such_file.mp4")
    assert "ai_probability" in res
    assert "signals" in res
    assert len(res["signals"]) == 6
