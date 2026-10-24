from __future__ import annotations

from pathlib import Path

from truthlens.config import sample_frames as configured_sample_frames
from truthlens.data.video import read_sampled_frames
from truthlens.inference.report import build_result, file_info
from truthlens.models.fusion import Fusion
from truthlens.signals import s1_frame, s2_temporal, s3_freq, s4_audio, s5_meta, s6_semantic

_fusion = Fusion()


def run_pipeline(video_path: str | Path, filename: str | None = None) -> dict:
    # S1/S2/S3 use the same sparse decode set; each signal still applies its
    # existing sample count and calculations to preserve detector behavior.
    shared_sample_count = max(configured_sample_frames(), 3)
    sampled = read_sampled_frames(video_path, shared_sample_count)
    s1 = s1_frame.score(video_path, sampled=sampled).to_dict()
    s2 = s2_temporal.score(video_path, sampled=sampled).to_dict()
    s3 = s3_freq.score(video_path, sampled=sampled).to_dict()
    s4 = s4_audio.score(video_path).to_dict()
    s5 = s5_meta.score(video_path).to_dict()
    s6 = s6_semantic.score(video_path).to_dict()
    scores = [s1, s2, s3, s4, s5, s6]
    ai_pct = _fusion.predict_proba_pct(scores)
    res = build_result(scores, ai_pct, file_meta=file_info(video_path, filename))
    res["explanation"] = (
        "Verdict based on fused forensic evidence "
        f"({res['signals_analysed']} of {res['signals_total']} checks ran). "
        "The headline number is a detection score, not a calibrated "
        "probability. See per-check findings and limitations below."
    )
    return res
