def test_pipeline_no_file():
    from truthlens.inference.pipeline import run_pipeline

    res = run_pipeline("no_such_file.mp4")
    assert "ai_probability" in res
    assert "signals" in res
    assert len(res["signals"]) == 6
    assert res["verdict"] == "inconclusive"
    assert res["signals_analysed"] == 0
    assert all(
        signal["status"] != "completed"
        for signal in res["signals"]
        if signal["key"] in {"s2_temporal", "s3_freq"}
    )


def test_pipeline_reuses_one_shared_visual_sample(monkeypatch):
    from truthlens.inference import pipeline
    from truthlens.signals.base import SignalScore

    sampled = {"fps": 24.0, "duration": 1.0, "total_frames": 24, "frames": {}}
    calls = {"decode": 0, "shared": 0}

    def decode(_path, target_frames):
        calls["decode"] += 1
        assert target_frames == 3
        return sampled

    def visual_score(name):
        def score(_path, sampled=None):
            assert sampled is not None
            calls["shared"] += 1
            return SignalScore(name, 0.5, evidence=[], status="not_available")

        return score

    monkeypatch.setattr(pipeline, "read_sampled_frames", decode)
    monkeypatch.setattr(pipeline, "configured_sample_frames", lambda: 2)
    monkeypatch.setattr(pipeline.s1_frame, "score", visual_score("s1_frame"))
    monkeypatch.setattr(pipeline.s2_temporal, "score", visual_score("s2_temporal"))
    monkeypatch.setattr(pipeline.s3_freq, "score", visual_score("s3_freq"))
    monkeypatch.setattr(
        pipeline.s4_audio,
        "score",
        lambda _path: SignalScore("s4_audio", 0.5, evidence=[], status="not_available"),
    )
    monkeypatch.setattr(
        pipeline.s5_meta,
        "score",
        lambda _path: SignalScore("s5_meta", 0.5, evidence=[], status="not_available"),
    )
    monkeypatch.setattr(
        pipeline.s6_semantic,
        "score",
        lambda _path: SignalScore("s6_semantic", 0.5, evidence=[], status="not_available"),
    )

    result = pipeline.run_pipeline("shared.mp4")

    assert result["signals_total"] == 6
    assert calls == {"decode": 1, "shared": 3}
