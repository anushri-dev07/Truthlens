import re

from fastapi.testclient import TestClient

from truthlens.api import main
from truthlens.api.main import app

client = TestClient(app)

def test_index_ok():
    r = client.get("/")
    assert r.status_code == 200
    assert "Understand the signals behind a video's authenticity." in r.text
    assert "MP4, WebM, MOV" in r.text
    assert "Up to 100 MiB" in r.text
    assert "Analyze Video" in r.text
    assert "Experimental AI-video screening. Results are not proof of authenticity." in r.text


def test_readiness_measures_evidence_coverage_not_ai_score():
    from truthlens.inference.report import calculate_readiness

    signals = [
        {"key": key, "status": "completed", "score": 99.0}
        for key in ("s1_frame", "s2_temporal", "s3_freq", "s5_meta")
    ]

    readiness = calculate_readiness(
        signals,
        {"requested_frames": 8, "scored_frames": 6, "duration_sec": 8},
    )

    assert readiness["score"] == 88
    assert readiness["label"] == "Good analysis readiness"
    assert readiness["factors"] == [
        "6 of 8 target frames scored.",
        "4 of 4 core checks completed.",
    ]


def test_readiness_is_unavailable_when_measurements_are_missing():
    from truthlens.inference.report import calculate_readiness

    readiness = calculate_readiness([], {})

    assert readiness["score"] is None
    assert readiness["label"] == "Unavailable"


def test_readiness_labels_limited_and_insufficient_evidence():
    from truthlens.inference.report import calculate_readiness

    signals = [
        {"key": "s1_frame", "status": "completed"},
        {"key": "s2_temporal", "status": "completed"},
        {"key": "s3_freq", "status": "failed"},
        {"key": "s5_meta", "status": "not_available"},
    ]
    limited = calculate_readiness(
        signals,
        {"requested_frames": 8, "scored_frames": 4, "duration_sec": 8},
    )
    insufficient = calculate_readiness(
        [{"key": signal["key"], "status": "failed"} for signal in signals],
        {"requested_frames": 8, "scored_frames": 0, "duration_sec": 8},
    )

    assert limited["score"] == 50
    assert limited["label"] == "Limited evidence"
    assert insufficient["score"] == 0
    assert insufficient["label"] == "Insufficient evidence"


def test_registration_login_logout_and_owned_history(monkeypatch, tmp_path):
    from truthlens.inference import store
    from truthlens.inference.report import build_result

    monkeypatch.setattr(store, "DB_PATH", tmp_path / "auth-test.db")
    monkeypatch.setattr(
        main,
        "run_pipeline",
        lambda _path, filename=None: build_result([], 50.0, file_meta={"name": filename}),
    )
    monkeypatch.setattr(main, "render_pdf", lambda _result: b"%PDF-owned")
    auth_client = TestClient(app, follow_redirects=False)

    registration_page = auth_client.get("/register")
    csrf = re.search(r'name="csrf_token" value="([^"]+)"', registration_page.text).group(1)
    registered = auth_client.post(
        "/register",
        data={
            "email": "Person@Example.com",
            "password": "very-secure-test-password",
            "csrf_token": csrf,
        },
    )
    assert registered.status_code == 303
    assert registered.headers["location"] == "/history"
    signed_in_page = auth_client.get("/")
    csrf = re.search(r'name="csrf_token" value="([^"]+)"', signed_in_page.text).group(1)

    owner_id = store.get_user_by_email("person@example.com")["id"]
    other_id = store.create_user("other@example.com", "not-a-real-hash")
    analysis_response = auth_client.post(
        "/analyze",
        data={"csrf_token": csrf},
        files={"video": ("owned.mp4", b"video bytes", "video/mp4")},
    )
    assert analysis_response.status_code == 200
    assert "Readiness Score" in analysis_response.text
    record_id = store.list_analyses(owner_id=owner_id)[0]["id"]
    foreign_id = store.save_analysis(
        build_result([], 50.0, file_meta={"name": "private.mp4"}),
        owner_id=other_id,
    )

    history_response = auth_client.get("/history")
    assert "owned.mp4" in history_response.text
    assert "private.mp4" not in history_response.text
    csrf = re.search(r'name="csrf_token" value="([^"]+)"', history_response.text).group(1)
    owned_report = auth_client.get(f"/report/{record_id}")
    assert owned_report.status_code == 200
    assert 'href="/report/' + record_id + '/pdf"' in owned_report.text
    assert auth_client.get(f"/report/{foreign_id}").status_code == 404
    assert auth_client.get(f"/report/{foreign_id}").status_code == 404
    assert auth_client.get(f"/report/{foreign_id}/pdf").status_code == 404

    delete_foreign = auth_client.post(
        f"/history/{foreign_id}/delete",
        data={"csrf_token": csrf},
    )
    assert delete_foreign.status_code == 303
    assert store.get_analysis(foreign_id, owner_id=other_id) is not None

    delete_owned = auth_client.post(
        f"/history/{record_id}/delete",
        data={"csrf_token": csrf},
    )
    assert delete_owned.status_code == 303
    assert store.get_analysis(record_id, owner_id=owner_id) is None

    auth_client.post("/logout", data={"csrf_token": csrf})
    login_page = auth_client.get("/login")
    login_csrf = re.search(r'name="csrf_token" value="([^"]+)"', login_page.text).group(1)
    login_response = auth_client.post(
        "/login",
        data={
            "email": "person@example.com",
            "password": "very-secure-test-password",
            "csrf_token": login_csrf,
        },
    )
    assert login_response.status_code == 303
    assert auth_client.get("/history").status_code == 200


def test_auth_requires_csrf_and_valid_password(monkeypatch, tmp_path):
    from truthlens.inference import store

    monkeypatch.setattr(store, "DB_PATH", tmp_path / "csrf-test.db")
    auth_client = TestClient(app, follow_redirects=False)
    response = auth_client.post(
        "/register",
        data={
            "email": "valid@example.com",
            "password": "very-secure-test-password",
            "csrf_token": "bad",
        },
    )
    assert response.status_code == 400
    assert store.get_user_by_email("valid@example.com") is None

    page = auth_client.get("/register")
    csrf = re.search(r'name="csrf_token" value="([^"]+)"', page.text).group(1)
    weak_password = auth_client.post(
        "/register",
        data={"email": "valid@example.com", "password": "short", "csrf_token": csrf},
    )
    assert weak_password.status_code == 400
    assert store.get_user_by_email("valid@example.com") is None


def test_login_rate_limit(monkeypatch, tmp_path):
    from truthlens.inference import store

    monkeypatch.setattr(store, "DB_PATH", tmp_path / "login-limit.db")
    auth_client = TestClient(app, follow_redirects=False)
    page = auth_client.get("/login")
    csrf = re.search(r'name="csrf_token" value="([^"]+)"', page.text).group(1)
    payload = {
        "email": "rate-limit@example.com",
        "password": "incorrect-password",
        "csrf_token": csrf,
    }

    responses = [auth_client.post("/login", data=payload) for _ in range(8)]

    assert all(response.status_code == 401 for response in responses)
    assert auth_client.post("/login", data=payload).status_code == 429


def test_database_migration_preserves_guest_history(monkeypatch, tmp_path):
    import sqlite3

    from truthlens.inference import store
    from truthlens.inference.report import build_result

    db_path = tmp_path / "legacy.db"
    monkeypatch.setattr(store, "DB_PATH", db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE analyses (id TEXT PRIMARY KEY, filename TEXT, size_bytes INTEGER, "
            "sha256 TEXT, created_at TEXT, verdict TEXT, ai_probability REAL, status TEXT, "
            "result_json TEXT)"
        )
        conn.execute(
            "INSERT INTO analyses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy-id",
                "legacy.mp4",
                12,
                "hash",
                "2026-01-01T00:00:00+00:00",
                "inconclusive",
                50.0,
                "completed",
                '{"id":"legacy-id"}',
            ),
        )

    migrated = store.list_analyses()
    assert len(migrated) == 1
    assert migrated[0]["id"] == "legacy-id"
    assert store.get_analysis("legacy-id")["id"] == "legacy-id"

    owner_id = store.create_user("migration@example.com", "scrypt-test")
    owned_id = store.save_analysis(
        build_result([], 50.0, file_meta={"name": "new.mp4"}), owner_id=owner_id
    )
    assert [record["id"] for record in store.list_analyses(owner_id=owner_id)] == [owned_id]
    assert store.get_analysis(owned_id) is None


def test_analysis_remains_visible_when_history_save_fails(monkeypatch):
    from truthlens.inference.report import build_result

    monkeypatch.setattr(
        main,
        "run_pipeline",
        lambda _path, filename=None: build_result([], 50.0, file_meta={"name": filename}),
    )
    monkeypatch.setattr(main, "save_analysis", lambda *_args, **_kwargs: "")

    response = client.post(
        "/analyze",
        files={"video": ("not-stored.mp4", b"video bytes", "video/mp4")},
    )

    assert response.status_code == 200
    assert "not-stored.mp4" in response.text
    assert "could not be saved to history" in response.text
    assert 'href="/report/' not in response.text


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
                    "sampling": {
                        "requested_frames": 1,
                        "scored_frames": 1,
                        "sampled_times": [0.0],
                    },
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
    assert "checks unavailable or failed" in response.text
    assert "View technical details" in response.text
    assert "Readiness Score" in response.text
    assert "Evidence completeness for analysis, not likelihood of AI generation." in response.text
    assert "Key findings" in response.text
    assert result["readiness"]["score"] == 56
    assert result["readiness"]["label"] == "Limited evidence"
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
    monkeypatch.setattr(main, "get_analysis", lambda _record_id, owner_id=None: {"id": "pdf-demo"})
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
