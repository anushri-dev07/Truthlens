import asyncio
import hmac
import logging
import os
import re
import secrets
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from truthlens.api.auth import (
    clear_login_failures,
    hash_password,
    login_is_limited,
    record_login_failure,
    verify_password,
)
from truthlens.inference.pdf_report import render_pdf
from truthlens.inference.pipeline import run_pipeline
from truthlens.inference.store import (
    create_user,
    delete_analysis,
    get_analysis,
    get_user,
    get_user_by_email,
    list_analyses,
    save_analysis,
)

app = FastAPI(title="TruthLens")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
logger = logging.getLogger(__name__)
_SESSION_SECRET = os.environ.get("TRUTHLENS_SESSION_SECRET")
if not _SESSION_SECRET:
    _SESSION_SECRET = secrets.token_urlsafe(32)
    logger.warning(
        "TRUTHLENS_SESSION_SECRET is unset; auth sessions will not survive a restart."
    )
app.add_middleware(
    SessionMiddleware,
    secret_key=_SESSION_SECRET,
    session_cookie="truthlens_session",
    max_age=60 * 60 * 24 * 7,
    same_site="lax",
    https_only=os.environ.get("TRUTHLENS_HTTPS_ONLY", "0") == "1",
)

_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_BATCH_BYTES = 250 * 1024 * 1024
MAX_BATCH_FILES = 5
UPLOAD_CHUNK_BYTES = 1024 * 1024


class UploadValidationError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _upload_error(request: Request, message: str, status_code: int) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"error": message, "user": _current_user(request), "csrf_token": _csrf_token(request)},
        status_code=status_code,
    )


def _current_user(request: Request) -> dict | None:
    user_id = request.session.get("user_id")
    if not isinstance(user_id, str):
        return None
    user = get_user(user_id)
    if user is None:
        request.session.clear()
    return user


def _csrf_token(request: Request) -> str:
    token = request.session.get("csrf_token")
    if not isinstance(token, str):
        token = secrets.token_urlsafe(32)
        request.session["csrf_token"] = token
    return token


def _valid_csrf(request: Request, submitted: str | None) -> bool:
    expected = request.session.get("csrf_token")
    return (
        isinstance(expected, str)
        and isinstance(submitted, str)
        and hmac.compare_digest(expected, submitted)
    )


def _signed_in(request: Request) -> bool:
    return _current_user(request) is not None


async def _stage_upload(upload: UploadFile, suffix: str) -> tuple[str, int]:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    size = 0
    try:
        with tmp:
            while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise UploadValidationError(
                        f"{upload.filename} exceeds the 100 MiB per-video limit.",
                        status_code=413,
                    )
                tmp.write(chunk)
        if size == 0:
            raise UploadValidationError(f"{upload.filename} is empty.")
        return tmp.name, size
    except Exception:
        Path(tmp.name).unlink(missing_ok=True)
        raise


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"user": _current_user(request), "csrf_token": _csrf_token(request)},
    )


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    if _current_user(request):
        return RedirectResponse("/history", status_code=303)
    return templates.TemplateResponse(
        request, "register.html", {"csrf_token": _csrf_token(request)}
    )


@app.post("/register", response_class=HTMLResponse)
async def register(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    csrf_token: str = Form(...),
):
    normalized_email = email.strip().lower()
    error = None
    if not _valid_csrf(request, csrf_token):
        error = "Your form expired. Reload the page and try again."
    elif len(normalized_email) > 254 or not _EMAIL_RE.fullmatch(normalized_email):
        error = "Enter a valid email address."
    elif len(password) < 12 or len(password) > 128:
        error = "Use a password between 12 and 128 characters."
    else:
        user_id = create_user(normalized_email, hash_password(password))
        if user_id is None:
            error = "An account with that email already exists."
        else:
            request.session.clear()
            request.session["user_id"] = user_id
            _csrf_token(request)
            return RedirectResponse("/history", status_code=303)
    return templates.TemplateResponse(
        request,
        "register.html",
        {"error": error, "csrf_token": _csrf_token(request)},
        status_code=400,
    )


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if _current_user(request):
        return RedirectResponse("/history", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"csrf_token": _csrf_token(request)})


@app.post("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    csrf_token: str = Form(...),
):
    normalized_email = email.strip().lower()
    client_host = request.client.host if request.client else "unknown"
    login_identity = f"{client_host}:{normalized_email}"
    if not _valid_csrf(request, csrf_token):
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "error": "Your form expired. Reload the page and try again.",
                "csrf_token": _csrf_token(request),
            },
            status_code=400,
        )
    if login_is_limited(login_identity):
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "error": "Too many attempts. Try again in 15 minutes.",
                "csrf_token": _csrf_token(request),
            },
            status_code=429,
        )
    if len(password) > 128:
        record_login_failure(login_identity)
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Email or password is incorrect.", "csrf_token": _csrf_token(request)},
            status_code=401,
        )
    user = get_user_by_email(normalized_email)
    if not user or not verify_password(password, user["password_hash"]):
        record_login_failure(login_identity)
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Email or password is incorrect.", "csrf_token": _csrf_token(request)},
            status_code=401,
        )
    clear_login_failures(login_identity)
    request.session.clear()
    request.session["user_id"] = user["id"]
    _csrf_token(request)
    return RedirectResponse("/history", status_code=303)


@app.post("/logout")
async def logout(request: Request, csrf_token: str = Form(...)):
    if not _valid_csrf(request, csrf_token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    request.session.clear()
    return RedirectResponse("/", status_code=303)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    video: list[UploadFile] = File(default=[]),
    csrf_token: str | None = Form(default=None),
):
    user = _current_user(request)
    request_started = time.perf_counter()
    if user and not _valid_csrf(request, csrf_token):
        return _upload_error(request, "The request could not be verified. Reload and retry.", 403)
    uploads = [u for u in (video or []) if u and u.filename]
    logger.info("analysis request received file_count=%d", len(uploads))
    if not uploads:
        return _upload_error(request, "Select at least one video to analyze.", 400)
    if len(uploads) > MAX_BATCH_FILES:
        for upload in uploads:
            await upload.close()
        return _upload_error(
            request,
            f"Choose no more than {MAX_BATCH_FILES} videos per batch.",
            400,
        )

    for upload in uploads:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
            for item in uploads:
                await item.close()
            allowed = ", ".join(sorted(ext.lstrip(".") for ext in SUPPORTED_VIDEO_EXTENSIONS))
            return _upload_error(
                request,
                f"{upload.filename} is not a supported video type. Use {allowed}.",
                400,
            )

    results = []
    staged: list[tuple[UploadFile, str]] = []
    total_size = 0
    try:
        for upload in uploads:
            path, size = await _stage_upload(upload, Path(upload.filename).suffix.lower())
            staged.append((upload, path))
            total_size += size
            if total_size > MAX_BATCH_BYTES:
                raise UploadValidationError(
                    "The combined upload exceeds the 250 MiB batch limit.",
                    status_code=413,
                )
    except UploadValidationError as exc:
        for _, path in staged:
            Path(path).unlink(missing_ok=True)
        return _upload_error(request, str(exc), exc.status_code)
    except Exception:
        logger.exception("Could not stage uploaded video files")
        for _, path in staged:
            Path(path).unlink(missing_ok=True)
        return _upload_error(
            request,
            "The upload could not be read. Please try again with a supported video file.",
            400,
        )
    finally:
        for upload in uploads:
            await upload.close()
    logger.info("upload staged file_count=%d total_bytes=%d", len(staged), total_size)

    for upload, path in staged:
        filename = Path(upload.filename).name
        pipeline_started = time.perf_counter()
        try:
            res = await asyncio.to_thread(run_pipeline, path, filename=filename)
            logger.info(
                "analysis pipeline completed suffix=%s elapsed_seconds=%.2f signals=%s",
                Path(filename).suffix.lower(),
                time.perf_counter() - pipeline_started,
                res.get("signals_analysed"),
            )
            owner_id = user["id"] if user else None
            record_id = save_analysis(res, owner_id=owner_id) if owner_id else save_analysis(res)
            if not record_id:
                res["history_warning"] = (
                    "Analysis completed, but this report could not be saved to history."
                )
            res["user"] = user
            res["csrf_token"] = _csrf_token(request)
            results.append(res)
        except Exception:
            logger.exception(
                "video analysis failed suffix=%s elapsed_seconds=%.2f",
                Path(filename).suffix.lower(),
                time.perf_counter() - pipeline_started,
            )
            results.append(
                {
                    "error": (
                        "Analysis failed unexpectedly. Try another supported video; "
                        "no report was saved for this file."
                    ),
                    "file": {"name": filename},
                }
            )
        finally:
            Path(path).unlink(missing_ok=True)

    logger.info(
        "analysis request completed elapsed_seconds=%.2f result_count=%d",
        time.perf_counter() - request_started,
        len(results),
    )

    if len(results) == 1 and "error" not in results[0]:
        return templates.TemplateResponse(request, "result.html", results[0])
    if len(results) == 1:
        return _upload_error(request, results[0]["error"], 500)
    return templates.TemplateResponse(
        request, "batch.html", {"results": results, "count": len(results)}
    )


@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    user = _current_user(request)
    return templates.TemplateResponse(
        request,
        "history.html",
        {
            "records": (
                list_analyses(owner_id=user["id"]) if user else list_analyses()
            ),
            "user": user,
            "csrf_token": _csrf_token(request),
        },
    )


@app.get("/report/{record_id}", response_class=HTMLResponse)
async def report(request: Request, record_id: str):
    user = _current_user(request)
    res = get_analysis(record_id, owner_id=user["id"]) if user else get_analysis(record_id)
    if res is None:
        raise HTTPException(status_code=404, detail="Report not found")
    res["user"] = user
    res["csrf_token"] = _csrf_token(request)
    return templates.TemplateResponse(request, "result.html", res)


@app.get("/report/{record_id}/pdf")
async def report_pdf(request: Request, record_id: str):
    user = _current_user(request)
    res = get_analysis(record_id, owner_id=user["id"]) if user else get_analysis(record_id)
    if res is None:
        raise HTTPException(status_code=404, detail="Report not found")
    pdf_bytes = render_pdf(res)
    headers = {"Content-Disposition": f'attachment; filename="truthlens-{record_id}.pdf"'}
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)


@app.post("/history/{record_id}/delete")
async def delete_history_record(
    request: Request,
    record_id: str,
    csrf_token: str = Form(...),
):
    user = _current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Log in to manage saved analyses")
    if not _valid_csrf(request, csrf_token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    delete_analysis(record_id, user["id"])
    return RedirectResponse("/history", status_code=303)
