from fastapi.testclient import TestClient

from truthlens.api import main
from truthlens.api.main import app

client = TestClient(app)

def test_index_ok():
    r = client.get("/")
    assert r.status_code == 200
    assert "uncalibrated detection score" in r.text
    assert "not the probability" in r.text
    assert "No AV synchronization check" in r.text
    assert "Not implemented" in r.text
    assert "AI probability" not in r.text


def test_report_renders_mixed_and_unavailable_evidence(monkeypatch):
    from truthlens.inference.report import build_result

    scores = [
        {
            "name": "s1_frame",
            "prob_ai": 0.9,
            "status": "completed",
            "evidence": ["s1_method=cnn_probe", "s1_frames=1", "s1_faces=1"],
            "details": {
                "method": "cnn_probe",
                "timeline": [
                    {
                        "t": 0.0,
                        "score": 0.9,
                        "processed": True,
                        "frame": 0,
                        "preview": "data:image/jpeg;base64,dGVzdA==",
                    }
                ],
                "video": {"duration_sec": 8.0, "fps": 24, "total_frames": 192},
                "sampling": {"scored_frames": 1, "sampled_times": [0.0]},
            },
        },
        {
            "name": "s2_temporal",
            "prob_ai": 0.1,
            "status": "completed",
            "evidence": ["s2_flow_mean=1.2", "s2_diffvar_mean=4.5"],
            "details": {},
        },
        {
            "name": "s3_freq",
            "prob_ai": 0.5,
            "status": "completed",
            "evidence": ["s3_fftvar=2.4", "s3_blockfftvar=3.1"],
            "details": {},
        },
        {
            "name": "s4_audio",
            "prob_ai": 0.5,
            "status": "not_available",
            "evidence": ["no-audio-stream"],
            "details": {},
        },
        {
            "name": "s5_meta",
            "prob_ai": 0.5,
            "status": "completed",
            "evidence": ["tags_present=False"],
            "details": {},
        },
        {
            "name": "s6_semantic",
            "prob_ai": 0.5,
            "status": "not_available",
            "evidence": ["semantic-placeholder"],
            "details": {},
        },
    ]
    result = build_result(scores, 85.0, file_meta={"name": "mixed-demo.mp4"})
    result["id"] = "demo-report"
    monkeypatch.setattr(main, "get_analysis", lambda _record_id: result)

    response = client.get("/report/demo-report")

    assert response.status_code == 200
    assert "mixed-demo.mp4" in response.text
    assert "Mixed signals" in response.text
    assert "Incomplete evidence" in response.text
    assert "not available" in response.text
    assert "Uncalibrated detection score" in response.text
    assert "data:image/jpeg;base64,dGVzdA==" in response.text
    assert 'href="/report/demo-report/pdf"' in response.text
    assert 'href="/history"' in response.text
    assert all(
        title in response.text
        for title in (
            "1. Evidence timeline and sampled frames",
            "2. Frame analysis",
            "3. Temporal consistency",
            "4. Frequency analysis",
            "5. Audio analysis",
            "6. Metadata inspection",
            "7. Semantic analysis status",
            "8. Digital evidence and SHA-256 verification",
            "9. Full limitations and raw technical measurements",
        )
    )
    assert response.text.count("<details class=\"disclosure\">") == 9


def test_history_route_still_lists_and_links_saved_reports(monkeypatch):
    monkeypatch.setattr(
        main,
        "list_analyses",
        lambda: [
            {
                "id": "history-demo",
                "filename": "saved.mp4",
                "verdict": "inconclusive",
                "ai_probability": 55.0,
                "created_at": "2026-10-22T12:00:00+00:00",
                "sha256": "abc123",
            }
        ],
    )

    response = client.get("/history")

    assert response.status_code == 200
    assert "saved.mp4" in response.text
    assert 'href="/report/history-demo"' in response.text


def test_pdf_download_route_still_returns_pdf(monkeypatch):
    monkeypatch.setattr(main, "get_analysis", lambda _record_id: {"id": "pdf-demo"})
    monkeypatch.setattr(main, "render_pdf", lambda _result: b"%PDF-demo")

    response = client.get("/report/pdf-demo/pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")


def test_analyze_without_files_shows_selection_message():
    response = client.post("/analyze")

    assert response.status_code == 400
    assert "Select at least one video" in response.text


def test_analyze_rejects_unsupported_extension():
    response = client.post(
        "/analyze",
        files={"video": ("clip.avi", b"not a video", "video/x-msvideo")},
    )

    assert response.status_code == 400
    assert "not a supported video type" in response.text
    assert all(extension in response.text for extension in ("mp4", "mov", "webm"))


def test_analyze_rejects_file_over_size_limit(monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 4)
    pipeline_called = False

    def unexpected_pipeline(*_args, **_kwargs):
        nonlocal pipeline_called
        pipeline_called = True

    monkeypatch.setattr(main, "run_pipeline", unexpected_pipeline)
    response = client.post(
        "/analyze",
        files={"video": ("clip.mp4", b"12345", "video/mp4")},
    )

    assert response.status_code == 413
    assert "100 MiB per-video limit" in response.text
    assert not pipeline_called


def test_analyze_rejects_too_many_batch_files():
    response = client.post(
        "/analyze",
        files=[("video", (f"clip-{i}.mp4", b"x", "video/mp4")) for i in range(6)],
    )

    assert response.status_code == 400
    assert "no more than 5 videos" in response.text


def test_analyze_rejects_combined_batch_over_size_limit(monkeypatch):
    monkeypatch.setattr(main, "MAX_BATCH_BYTES", 3)
    pipeline_called = False

    def unexpected_pipeline(*_args, **_kwargs):
        nonlocal pipeline_called
        pipeline_called = True

    monkeypatch.setattr(main, "run_pipeline", unexpected_pipeline)
    response = client.post(
        "/analyze",
        files=[
            ("video", ("first.mp4", b"12", "video/mp4")),
            ("video", ("second.mp4", b"34", "video/mp4")),
        ],
    )

    assert response.status_code == 413
    assert "combined upload exceeds the 250 MiB batch limit" in response.text
    assert not pipeline_called


def test_analyze_reports_empty_upload():
    response = client.post(
        "/analyze",
        files={"video": ("empty.mp4", b"", "video/mp4")},
    )

    assert response.status_code == 400
    assert "empty.mp4 is empty" in response.text


def test_analyze_shows_actionable_error_when_pipeline_raises(monkeypatch):
    def fail_pipeline(*_args, **_kwargs):
        raise RuntimeError("private internal detail")

    monkeypatch.setattr(main, "run_pipeline", fail_pipeline)
    response = client.post(
        "/analyze",
        files={"video": ("clip.mp4", b"video bytes", "video/mp4")},
    )

    assert response.status_code == 500
    assert "Analysis failed unexpectedly" in response.text
    assert "private internal detail" not in response.text


def test_batch_displays_per_file_analysis_failure(monkeypatch):
    def sometimes_fail(path, filename=None):
        if filename == "bad.mp4":
            raise RuntimeError("decoder broke")
        return {
            "file": {"name": filename},
            "verdict": "inconclusive",
            "verdict_text": "Inconclusive",
            "overall_score": None,
            "score_separation": None,
            "signals_analysed": 0,
            "signals_total": 6,
        }

    monkeypatch.setattr(main, "run_pipeline", sometimes_fail)
    monkeypatch.setattr(main, "save_analysis", lambda _result: "")
    response = client.post(
        "/analyze",
        files=[
            ("video", ("bad.mp4", b"bad video", "video/mp4")),
            ("video", ("ok.mp4", b"valid enough for mocked pipeline", "video/mp4")),
        ],
    )

    assert response.status_code == 200
    assert "bad.mp4" in response.text
    assert "Analysis failed unexpectedly" in response.text
    assert "ok.mp4" in response.text
