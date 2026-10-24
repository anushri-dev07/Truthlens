def _sig(name, p, status="completed"):
    return {
        "name": name,
        "prob_ai": p,
        "evidence": ["e"],
        "status": status,
        "details": {},
    }


def _scores():
    return [
        _sig("s1_frame", 0.6),
        _sig("s2_temporal", 0.4),
        _sig("s3_freq", 0.5),
        _sig("s4_audio", 0.5, "not_available"),
        _sig("s5_meta", 0.5, "not_available"),
        _sig("s6_semantic", 0.5, "not_available"),
    ]


def test_build_result_inconclusive():
    from truthlens.inference.report import VERDICT_INCONCLUSIVE, build_result

    res = build_result(_scores(), 55.0, file_meta={"name": "x.mp4"})
    assert res["verdict"] == VERDICT_INCONCLUSIVE
    assert res["overall_score"] is None
    assert res["signals_analysed"] == 3


def test_build_result_verdicts():
    from truthlens.inference.report import VERDICT_AI, VERDICT_REAL, build_result

    assert build_result(_scores(), 85.0)["verdict"] == VERDICT_AI
    assert build_result(_scores(), 10.0)["verdict"] == VERDICT_REAL


def test_result_labels_separation_not_confidence():
    from truthlens.inference.report import build_result

    result = build_result(_scores(), 85.0)

    assert result["score_separation"] == "high"
    assert "confidence" not in result
    assert "not the probability" in result["score_note"]


def test_disagreement_detected():
    from truthlens.inference.report import build_result

    scores = [
        _sig("s1_frame", 0.9),
        _sig("s2_temporal", 0.1),
        _sig("s3_freq", 0.5),
    ]
    res = build_result(scores, 45.0)
    assert res["disagreement"] is True
    assert "disagree" in res["disagreement_note"].lower()
    assert res["detector_version"]


def test_store_roundtrip():
    from truthlens.inference.report import build_result
    from truthlens.inference.store import get_analysis, save_analysis

    res = build_result(_scores(), 75.0, file_meta={"name": "y.mp4"})
    rid = save_analysis(res)
    assert rid
    loaded = get_analysis(rid)
    assert loaded is not None
    assert loaded["ai_probability"] == 75.0


def test_render_pdf():
    from truthlens.inference.pdf_report import render_pdf
    from truthlens.inference.report import build_result

    pdf = render_pdf(build_result(_scores(), 80.0, file_meta={"name": "z.mp4"}))
    assert pdf[:4] == b"%PDF"
