from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from truthlens.data.video import uniform_sample_indices
from truthlens.signals.base import SignalScore


def _fft_var(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (480, 270), interpolation=cv2.INTER_AREA)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    return float(mag.var())


def _block_var(gray: np.ndarray, block: int = 8) -> float:
    h, w = gray.shape
    h2, w2 = (h // block) * block, (w // block) * block
    g = gray[:h2, :w2].astype(np.float32)
    blocks = g.reshape(h2 // block, block, w2 // block, block).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(-1, block * block)
    dct = np.abs(np.fft.fft(blocks, axis=1))
    return float(dct.var())


def score(video_path: str | Path, sample_frames: int = 2) -> SignalScore:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return SignalScore("s3_freq", prob_ai=0.5, evidence=["cannot-open-video"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    dur = total / fps if fps > 0 else 0.0
    idxs = uniform_sample_indices(dur, fps, sample_frames)
    vars_fft = []
    vars_dct = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        vars_fft.append(_fft_var(frame))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vars_dct.append(_block_var(gray))
    cap.release()
    if not vars_fft:
        return SignalScore("s3_freq", prob_ai=0.5, evidence=["no-frames"])
    v_fft = float(np.mean(vars_fft))
    v_dct = float(np.mean(vars_dct)) if vars_dct else 0.0
    p = float(np.clip(1.0 / (1.0 + np.exp(-(v_fft * 0.008 + v_dct * 0.0001 - 1.2))), 0.2, 0.85))
    return SignalScore(
        "s3_freq",
        prob_ai=p,
        evidence=[f"s3_fftvar={v_fft:.2f}", f"s3_dct1dvar={v_dct:.2f}"],
    )
