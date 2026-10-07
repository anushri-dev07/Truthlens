from __future__ import annotations

from pathlib import Path

from .base import SignalScore


def score(video_path: str | Path) -> SignalScore:
    return SignalScore("s6_semantic", prob_ai=0.5, evidence=["semantic-placeholder"])
