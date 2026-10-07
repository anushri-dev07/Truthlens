from __future__ import annotations

from pathlib import Path

from truthlens.data.io import probe_video
from truthlens.signals.base import SignalScore


def score(video_path: str | Path) -> SignalScore:
    info = probe_video(video_path)
    evidence = []
    # simple heuristics: check for major AI-like red flags via metadata? placeholder
    fmt = info.get("format", {}) if isinstance(info, dict) else {}
    evidence.append(f"format_size={fmt.get('size')}")
    return SignalScore("s5_meta", prob_ai=0.5, evidence=evidence)
