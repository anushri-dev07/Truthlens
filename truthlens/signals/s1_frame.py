from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from truthlens.data.video import uniform_sample_indices
from truthlens.models.frame_clf import FrameClassifier
from truthlens.signals.base import SignalScore


def _get_fps_duration(cap: cv2.VideoCapture) -> tuple[float, float]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = total_frames / fps if fps > 0 else 0.0
    return float(fps), float(duration)


def score(video_path: str | Path, sample_frames: int = 4) -> SignalScore:
    clf = FrameClassifier()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return SignalScore("s1_frame", prob_ai=0.5, evidence=["cannot-open-video"])
    fps, duration = _get_fps_duration(cap)
    idxs = uniform_sample_indices(duration, fps, sample_frames)
    probs = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        p = clf.predict(frame)
        probs.append(p)
    cap.release()
    if not probs:
        return SignalScore("s1_frame", prob_ai=0.5, evidence=["no-frames-read"])
    p_mean = float(np.mean(probs))
    return SignalScore("s1_frame", prob_ai=p_mean, evidence=[f"s1_sampled={len(probs)}"])
