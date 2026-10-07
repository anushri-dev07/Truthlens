from __future__ import annotations

from pathlib import Path

from truthlens.data.io import probe_video
from truthlens.signals.base import SignalScore


def score(video_path: str | Path) -> SignalScore:
    info = probe_video(video_path)
    streams = info.get("streams", []) if isinstance(info, dict) else []
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    if not has_audio:
        return SignalScore("s4_audio", prob_ai=0.5, evidence=["no-audio-stream"])
    # Librosa optional? install if needed later; lightweight placeholder
    return SignalScore("s4_audio", prob_ai=0.5, evidence=["audio-present-placeholder"])
