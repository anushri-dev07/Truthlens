from __future__ import annotations

from pathlib import Path

import cv2

from truthlens.data.faces import extract_faces
from truthlens.data.video import uniform_sample_indices
from truthlens.signals.base import SignalScore


def _get_fps_duration(cap: cv2.VideoCapture) -> tuple[float, float]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = total_frames / fps if fps > 0 else 0.0
    return float(fps), float(duration)


def score(video_path: str | Path, sample_frames: int = 2) -> SignalScore:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return SignalScore("s1_frame", prob_ai=0.5, evidence=["cannot-open-video"])
    fps, duration = _get_fps_duration(cap)
    idxs = uniform_sample_indices(duration, fps, sample_frames)
    crops_count = 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        faces = extract_faces(frame)
        crops_count += len(faces)
    cap.release()
    # simple heuristic: more faces detected -> slightly less "suspicious" for some cases? weak
    p = 0.5
    if crops_count == 0:
        p = 0.55  # no faces found; could be synthetic scene
    else:
        p = max(0.3, 0.5 - min(0.2, crops_count * 0.01))
    return SignalScore(
        "s1_frame",
        prob_ai=p,
        evidence=[f"s1_faces={crops_count}", "s1_faceheuristic"],
    )
