from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from truthlens.data.video import uniform_sample_indices
from truthlens.signals.base import SignalScore


def _fps_dur(cap: cv2.VideoCapture) -> tuple[float, float]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    dur = total / fps if fps > 0 else 0.0
    return float(fps), float(dur)


def score(video_path: str | Path, sample_frames: int = 3) -> SignalScore:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return SignalScore(
            "s2_temporal",
            prob_ai=0.5,
            evidence=["cannot-open-video"],
            status="failed",
        )
    fps, dur = _fps_dur(cap)
    idxs = uniform_sample_indices(dur, fps, sample_frames)
    if not idxs:
        cap.release()
        return SignalScore(
            "s2_temporal",
            prob_ai=0.5,
            evidence=["no-sample-indices"],
            status="not_available",
        )
    flow_mags: list[float] = []
    diff_vars: list[float] = []
    prev_gray = None
    decoded_frames = 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        decoded_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_gray is not None:
            diff = np.abs(gray - prev_gray)
            diff_vars.append(float(diff.var()))
            try:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 1, 5, 1, 2, 1.1, 0)
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                flow_mags.append(float(mag.mean()))
            except Exception:
                flow_mags.append(0.0)
        prev_gray = gray
    cap.release()
    if not flow_mags and not diff_vars:
        evidence = ["no-readable-frames"] if decoded_frames == 0 else ["no-temporal-pairs"]
        status = "failed" if decoded_frames == 0 else "not_available"
        return SignalScore(
            "s2_temporal", prob_ai=0.5, evidence=evidence, status=status
        )
    v_flow = float(np.mean(flow_mags)) if flow_mags else 0.0
    v_diff = float(np.mean(diff_vars)) if diff_vars else 0.0
    p = 1.0 / (1.0 + np.exp(-(v_flow * 0.02 + v_diff * 0.0005 - 1.0)))
    p = float(np.clip(p, 0.1, 0.9))
    evidence = [f"s2_flow_mean={v_flow:.2f}", f"s2_diffvar_mean={v_diff:.2f}"]
    return SignalScore("s2_temporal", prob_ai=p, evidence=evidence)
