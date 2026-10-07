import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from truthlens.inference.pipeline import run_pipeline

app = FastAPI(title="TruthLens")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, video: UploadFile = File(...)):
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(video.file, tmp)
        tmp.close()
        res = run_pipeline(tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    return templates.TemplateResponse(request, "result.html", res)
