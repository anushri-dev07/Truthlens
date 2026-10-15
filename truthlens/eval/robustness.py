from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

STABLE_DELTA = 0.15


def _variant(src: str | Path, dst: str, scale: float = 1.0) -> bool:
    import cv2

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        cap.release()
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int((cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) * scale)
    h = int((cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) * scale)
    w = max(2, w - (w % 2))
    h = max(2, h - (h % 2))
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    ok = False
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
        ok = True
    cap.release()
    writer.release()
    return ok


def _pipeline_predictor(video_path: str) -> float:
    from truthlens.inference.pipeline import run_pipeline

    return float(run_pipeline(video_path)["ai_probability"]) / 100.0


def run_robustness(
    video_path: str | Path,
    predictor: Callable[[str], float] = _pipeline_predictor,
) -> dict:
    baseline = predictor(str(video_path))
    variants: dict[str, float] = {}
    transforms = {
        "resize_50pct": 0.5,
        "resize_75pct": 0.75,
        "reencode_same": 1.0,
    }
    for name, scale in transforms.items():
        fd = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        fd.close()
        try:
            if not _variant(video_path, fd.name, scale):
                continue
            variants[name] = round(predictor(fd.name), 4)
        except Exception:
            continue
        finally:
            Path(fd.name).unlink(missing_ok=True)

    deltas = [abs(v - baseline) for v in variants.values()]
    max_delta = round(max(deltas), 4) if deltas else 0.0
    return {
        "baseline": round(baseline, 4),
        "variants": variants,
        "max_delta": max_delta,
        "stable": max_delta <= STABLE_DELTA,
    }
