from __future__ import annotations

from pathlib import Path

import cv2

from truthlens.data.faces import extract_faces
from truthlens.data.video import uniform_sample_indices
from truthlens.signals.base import SignalScore

FACE_FOUND_P = 0.4
NO_FACE_P = 0.6


def _fps_duration(cap: cv2.VideoCapture) -> tuple[float, float]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = total_frames / fps if fps > 0 else 0.0
    return float(fps), float(duration)


def score(video_path: str | Path, sample_frames: int = 8) -> SignalScore:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return SignalScore(
            "s1_frame",
            prob_ai=0.5,
            evidence=["cannot-open-video"],
            status="failed",
        )
    fps, duration = _fps_duration(cap)
    idxs = uniform_sample_indices(duration, fps, sample_frames)
    timeline: list[dict] = []
    crops_count = 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        t = round(i / fps, 3) if fps > 0 else 0.0
        if not ret or frame is None:
            timeline.append({"t": t, "score": 0.5, "processed": False})
            continue
        faces = extract_faces(frame)
        crops_count += len(faces)
        p = FACE_FOUND_P if faces else NO_FACE_P
        timeline.append({"t": t, "score": p, "processed": True})
    cap.release()
    if not timeline:
        return SignalScore(
            "s1_frame",
            prob_ai=0.5,
            evidence=["no-frames-sampled"],
            status="failed",
        )
    processed = [f for f in timeline if f["processed"]]
    if not processed:
        return SignalScore(
            "s1_frame",
            prob_ai=0.5,
            evidence=["frames-unreadable"],
            status="failed",
            details={"timeline": timeline},
        )
    p = float(sum(f["score"] for f in processed) / len(processed))
    return SignalScore(
        "s1_frame",
        prob_ai=p,
        evidence=[f"s1_faces={crops_count}", f"s1_frames={len(processed)}"],
        status="completed",
        details={"timeline": timeline, "faces": crops_count},
    )
