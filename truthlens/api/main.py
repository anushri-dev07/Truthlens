import shutil
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


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, video: UploadFile = File(...)):
    if not video or not video.filename:
        return templates.TemplateResponse(request, "index.html")
    suffix = Path(video.filename).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(video.file, tmp)
        tmp.close()
        res = run_pipeline(tmp.name, filename=video.filename)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    save_analysis(res)
    return templates.TemplateResponse(request, "result.html", res)


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
