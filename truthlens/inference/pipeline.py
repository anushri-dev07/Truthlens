from __future__ import annotations

from pathlib import Path

import numpy as np

from truthlens.signals import (
    s1_frame,
    s2_temporal,
    s3_freq,
    s4_audio,
    s5_meta,
    s6_semantic,
)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))


def fuse_scores(scores: list[dict], weights: dict[str, float] | None = None) -> float:
    weights = weights or {
        "s1_frame": 0.35,
        "s2_temporal": 0.25,
        "s3_freq": 0.15,
        "s4_audio": 0.10,
        "s5_meta": 0.10,
        "s6_semantic": 0.05,
    }
    logit = 0.0
    for s in scores:
        name = s.get("name")
        p = float(s.get("prob_ai", 0.5))
        w = float(weights.get(name or "", 0.0))
        # map prob to logit approx
        eps = 1e-6
        p_cl = max(eps, min(1 - eps, p))
        logit += w * np.log(p_cl / (1 - p_cl))
    return float(_sigmoid(logit)) * 100.0


def run_pipeline(video_path: str | Path) -> dict:
    scores = [
        s1_frame.score(video_path).to_dict(),
        s2_temporal.score(video_path).to_dict(),
        s3_freq.score(video_path).to_dict(),
        s4_audio.score(video_path).to_dict(),
        s5_meta.score(video_path).to_dict(),
        s6_semantic.score(video_path).to_dict(),
    ]
    ai_pct = fuse_scores(scores)
    return {
        "ai_probability": round(ai_pct, 2),
        "signals": scores,
    }
