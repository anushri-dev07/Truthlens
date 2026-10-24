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


def test_pipeline_logs_stage_timings(monkeypatch, caplog):
    from truthlens.inference import pipeline
    from truthlens.signals.base import SignalScore

    sampled = {"fps": 24.0, "duration": 1.0, "total_frames": 24, "frames": {}}
    signal_names = (
        "s1_frame",
        "s2_temporal",
        "s3_freq",
        "s4_audio",
        "s5_meta",
        "s6_semantic",
    )
    monkeypatch.setattr(pipeline, "read_sampled_frames", lambda *_args: sampled)
    monkeypatch.setattr(pipeline, "configured_sample_frames", lambda: 2)
    for name in signal_names[:3]:
        monkeypatch.setattr(
            getattr(pipeline, name.split("_")[0] + "_" + name.split("_")[1]),
            "score",
            lambda *_args, _name=name, **_kwargs: SignalScore(
                _name, 0.5, evidence=[], status="not_available"
            ),
        )
    for name in signal_names[3:]:
        module = getattr(pipeline, name.split("_")[0] + "_" + name.split("_")[1])
        monkeypatch.setattr(
            module,
            "score",
            lambda *_args, _name=name: SignalScore(
                _name, 0.5, evidence=[], status="not_available"
            ),
        )

    with caplog.at_level("INFO", logger="truthlens.inference.pipeline"):
        pipeline.run_pipeline("timed.mp4")

    messages = "\n".join(record.message for record in caplog.records)
    assert "analysis_stage name=shared_decode" in messages
    assert "analysis_stage name=s1_frame" in messages
    assert "analysis_stage name=pipeline_total" in messages
