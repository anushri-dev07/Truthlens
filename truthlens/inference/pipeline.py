from __future__ import annotations

from pathlib import Path

from truthlens.signals import s1_frame, s2_temporal, s3_freq, s4_audio, s5_meta, s6_semantic


def _conf_from_p(p: float) -> str:
    if p < 0.4:
        return "low"
    if p < 0.7:
        return "medium"
    return "high"


def run_pipeline(video_path: str | Path) -> dict:
    s1 = s1_frame.score(video_path).to_dict()
    s2 = s2_temporal.score(video_path).to_dict()
    s3 = s3_freq.score(video_path).to_dict()
    s4 = s4_audio.score(video_path).to_dict()
    s5 = s5_meta.score(video_path).to_dict()
    s6 = s6_semantic.score(video_path).to_dict()
    scores = [s1, s2, s3, s4, s5, s6]
    # fusion
    from truthlens.inference.pipeline import fuse_scores

    ai_pct = fuse_scores(scores)
    # add confidence
    p_ai = ai_pct / 100.0
    confidence = _conf_from_p(p_ai)
    return {
        "ai_probability": round(ai_pct, 2),
        "signals": scores,
        "confidence": confidence,
    }
