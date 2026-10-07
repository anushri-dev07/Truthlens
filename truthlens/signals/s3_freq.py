from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from truthlens.data.video import uniform_sample_indices
from truthlens.signals.base import SignalScore


def _fft_var(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    return float(mag.var())


def score(video_path: str | Path, sample_frames: int = 10) -> SignalScore:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return SignalScore("s3_freq", prob_ai=0.5, evidence=["cannot-open-video"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    dur = total / fps if fps > 0 else 0.0
    idxs = uniform_sample_indices(dur, fps, sample_frames)
    vars_ = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        vars_.append(_fft_var(frame))
    cap.release()
    if not vars_:
        return SignalScore("s3_freq", prob_ai=0.5, evidence=["no-frames"])
    v = float(np.mean(vars_))
    p = float(np.clip(1.0 / (1.0 + np.exp(-(v * 0.01 - 1.5))), 0.2, 0.8))
    return SignalScore("s3_freq", prob_ai=p, evidence=[f"s3_fftvar_mean={v:.2f}"])
