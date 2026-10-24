from __future__ import annotations

import logging
import time
from pathlib import Path

from truthlens.config import sample_frames as configured_sample_frames
from truthlens.data.video import read_sampled_frames
from truthlens.inference.report import build_result, file_info
from truthlens.models.fusion import Fusion
from truthlens.signals import s1_frame, s2_temporal, s3_freq, s4_audio, s5_meta, s6_semantic

_fusion = Fusion()
logger = logging.getLogger(__name__)


def _run_signal(label: str, scorer) -> dict:
    started = time.perf_counter()
    result = scorer().to_dict()
    logger.info(
        "analysis_stage name=%s elapsed_seconds=%.3f status=%s",
        label,
        time.perf_counter() - started,
        result.get("status"),
    )
    return result


def run_pipeline(video_path: str | Path, filename: str | None = None) -> dict:
    started = time.perf_counter()
    # S1/S2/S3 use the same sparse decode set; each signal still applies its
    # existing sample count and calculations to preserve detector behavior.
    shared_sample_count = max(configured_sample_frames(), 3)
    decode_started = time.perf_counter()
    sampled = read_sampled_frames(video_path, shared_sample_count)
    logger.info(
        "analysis_stage name=shared_decode elapsed_seconds=%.3f sampled_frames=%d "
        "decoded_frames=%d video_duration_seconds=%.3f",
        time.perf_counter() - decode_started,
        len(sampled.get("indices", [])) if sampled else 0,
        sum(frame is not None for frame in sampled.get("frames", {}).values()) if sampled else 0,
        sampled.get("duration", 0.0) if sampled else 0.0,
    )
    s1 = _run_signal(
        "s1_frame", lambda: s1_frame.score(video_path, sampled=sampled)
    )
    s2 = _run_signal(
        "s2_temporal", lambda: s2_temporal.score(video_path, sampled=sampled)
    )
    s3 = _run_signal("s3_freq", lambda: s3_freq.score(video_path, sampled=sampled))
    s4 = _run_signal("s4_audio", lambda: s4_audio.score(video_path))
    s5 = _run_signal("s5_meta", lambda: s5_meta.score(video_path))
    s6 = _run_signal("s6_semantic", lambda: s6_semantic.score(video_path))
    scores = [s1, s2, s3, s4, s5, s6]
    fusion_started = time.perf_counter()
    ai_pct = _fusion.predict_proba_pct(scores)
    metadata = file_info(video_path, filename)
    res = build_result(scores, ai_pct, file_meta=metadata)
    logger.info(
        "analysis_stage name=aggregation elapsed_seconds=%.3f file_size_bytes=%s "
        "detection_score=%.1f",
        time.perf_counter() - fusion_started,
        metadata.get("size_bytes"),
        ai_pct,
    )
    res["explanation"] = (
        "Verdict based on fused forensic evidence "
        f"({res['signals_analysed']} of {res['signals_total']} checks ran). "
        "The headline number is a detection score, not a calibrated "
        "probability. See per-check findings and limitations below."
    )
    logger.info(
        "analysis_stage name=pipeline_total elapsed_seconds=%.3f signals_completed=%d",
        time.perf_counter() - started,
        res.get("signals_analysed", 0),
    )
    return res
