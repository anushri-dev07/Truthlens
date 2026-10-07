from __future__ import annotations

from pathlib import Path

from .base import SignalScore


def score_frame(video_path: str | Path, sample_frames: int = 16) -> SignalScore:
    return SignalScore("s1_frame", prob_ai=0.5, evidence=["frame-signal-placeholder"])
