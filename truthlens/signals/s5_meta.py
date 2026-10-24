from __future__ import annotations

from pathlib import Path

from truthlens.data.io import probe_video
from truthlens.signals.base import SignalScore


def _has_tag(fmt: dict, key: str) -> bool:
    tags = fmt.get("tags") or {}
    return key in tags or any(k.lower() == key.lower() for k in tags)


def score(video_path: str | Path) -> SignalScore:
    info = probe_video(video_path)
    fmt = info.get("format", {}) if isinstance(info, dict) else {}
    if not fmt:
        return SignalScore(
            "s5_meta",
            prob_ai=0.5,
            evidence=["metadata-unavailable"],
            status="not_available",
        )
    evidence = []
    if fmt:
        evidence.append(f"nb_streams={fmt.get('nb_streams')}")
    tags_present = bool(fmt and fmt.get("tags"))
    # Weak heuristic: missing camera metadata common in many synthetic renders
    p = 0.5
    if not tags_present:
        p = min(0.65, p + 0.15)
    else:
        p = max(0.4, p - 0.1)
    evidence.append(f"tags_present={tags_present}")
    return SignalScore("s5_meta", prob_ai=p, evidence=evidence)
