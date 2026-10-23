import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from truthlens.inference.pdf_report import render_pdf
from truthlens.inference.pipeline import run_pipeline
from truthlens.inference.store import get_analysis, list_analyses, save_analysis

app = FastAPI(title="TruthLens")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
logger = logging.getLogger(__name__)

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
        {"error": message},
        status_code=status_code,
    )


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
    return templates.TemplateResponse(request, "index.html")


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, video: list[UploadFile] = File(default=[])):
    uploads = [u for u in (video or []) if u and u.filename]
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

    for upload, path in staged:
        filename = Path(upload.filename).name
        try:
            res = run_pipeline(path, filename=filename)
            save_analysis(res)
            results.append(res)
        except Exception:
            logger.exception("Video analysis failed for %s", filename)
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

    if len(results) == 1 and "error" not in results[0]:
        return templates.TemplateResponse(request, "result.html", results[0])
    if len(results) == 1:
        return _upload_error(request, results[0]["error"], 500)
    return templates.TemplateResponse(
        request, "batch.html", {"results": results, "count": len(results)}
    )


@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    return templates.TemplateResponse(
        request, "history.html", {"records": list_analyses()}
    )


@app.get("/report/{record_id}", response_class=HTMLResponse)
async def report(request: Request, record_id: str):
    res = get_analysis(record_id)
    if res is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return templates.TemplateResponse(request, "result.html", res)


@app.get("/report/{record_id}/pdf")
async def report_pdf(record_id: str):
    res = get_analysis(record_id)
    if res is None:
        raise HTTPException(status_code=404, detail="Report not found")
    pdf_bytes = render_pdf(res)
    headers = {"Content-Disposition": f'attachment; filename="truthlens-{record_id}.pdf"'}
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
