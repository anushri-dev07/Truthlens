from __future__ import annotations

from pathlib import Path

from truthlens.signals.base import SignalScore


def score(video_path: str | Path) -> SignalScore:
    # CLIP-like semantics placeholder (CPU-only). Future: open_clip + frames
    return SignalScore(
        "s6_semantic",
        prob_ai=0.5,
        evidence=["semantic-placeholder"],
        status="not_available",
    )
