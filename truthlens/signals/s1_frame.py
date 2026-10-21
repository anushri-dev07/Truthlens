from __future__ import annotations

from pathlib import Path

import cv2

from truthlens.config import sample_frames as _cfg_sample_frames
from truthlens.data.faces import extract_faces
from truthlens.data.video import uniform_sample_indices
from truthlens.models.frame_clf import FrameClassifier
from truthlens.signals.base import SignalScore

FACE_FOUND_P = 0.4
NO_FACE_P = 0.6

_clf: FrameClassifier | None = None


def _classifier() -> FrameClassifier:
    global _clf
    if _clf is None:
        _clf = FrameClassifier()
    return _clf


def _fps_duration(cap: cv2.VideoCapture) -> tuple[float, float]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = total_frames / fps if fps > 0 else 0.0
    return float(fps), float(duration)


def score(video_path: str | Path, sample_frames: int | None = None) -> SignalScore:
    if sample_frames is None:
        sample_frames = _cfg_sample_frames()
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
    clf = _classifier()
    use_model = clf.available
    timeline: list[dict] = []
    frame_probs: list[float] = []
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
        if use_model:
            if faces:
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                crop = frame[y : y + h, x : x + w]
            else:
                crop = frame
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            p = clf.predict(rgb)
        else:
            p = FACE_FOUND_P if faces else NO_FACE_P
        frame_probs.append(p)
        timeline.append({"t": t, "score": round(p, 4), "processed": True})

    cap.release()
    if not frame_probs:
        return SignalScore(
            "s1_frame",
            prob_ai=0.5,
            evidence=["frames-unreadable"],
            status="failed",
            details={"timeline": timeline},
        )

    p = float(sum(frame_probs) / len(frame_probs))
    method = "cnn_vit" if use_model else "face-heuristic"
    evidence = [
        f"s1_method={method}",
        f"s1_frames={len(frame_probs)}",
        f"s1_faces={crops_count}",
    ]
    return SignalScore(
        "s1_frame",
        prob_ai=p,
        evidence=evidence,
        status="completed",
        details={"timeline": timeline, "faces": crops_count, "method": method},
    )
