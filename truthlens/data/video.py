from __future__ import annotations

import math
from pathlib import Path

import cv2


def uniform_sample_indices(duration_sec: float, fps: float, target_frames: int) -> list[int]:
    if (
        not math.isfinite(duration_sec)
        or not math.isfinite(fps)
        or duration_sec <= 0
        or fps <= 0
        or target_frames <= 0
    ):
        return []
    total = math.floor(duration_sec * fps)
    if total <= 0:
        return []
    if target_frames >= total:
        return list(range(total))
    if target_frames == 1:
        return [0]
    return [round(i * (total - 1) / (target_frames - 1)) for i in range(target_frames)]


def read_sampled_frames(video_path: str | Path, target_frames: int) -> dict | None:
    """Decode one reusable sample set for frame, temporal, and frequency signals."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0.0
    indices = uniform_sample_indices(duration, fps, target_frames)
    frames: dict[int, object | None] = {}
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        frames[index] = frame if ok and frame is not None else None
    cap.release()
    return {
        "fps": fps,
        "duration": duration,
        "total_frames": total_frames,
        "indices": indices,
        "frames": frames,
    }
