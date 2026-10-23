from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from truthlens.config import thresholds
from truthlens.inference.findings import SIGNAL_LABELS, build_all_findings

DETECTOR_VERSION = "0.2.0"

VERDICT_AI = "likely_ai"
VERDICT_REAL = "likely_real"
VERDICT_INCONCLUSIVE = "inconclusive"

VERDICT_TEXT = {
    VERDICT_AI: "Likely AI-generated",
    VERDICT_REAL: "No strong AI-generation evidence detected",
    VERDICT_INCONCLUSIVE: "Inconclusive — more evidence needed",
}

VERDICT_MEANING = {
    VERDICT_AI: (
        "The completed checks leaned towards AI-generated content. This is a "
        "signal that further review is warranted, not proof of synthesis."
    ),
    VERDICT_REAL: (
        "The completed checks did not produce evidence that pointed towards "
        "AI-generation. This is not a confirmation that the video is "
        "authentic - checks that did not run cannot support that conclusion."
    ),
    VERDICT_INCONCLUSIVE: (
        "The available evidence was insufficient or contradictory, so no "
        "verdict could be reached."
    ),
}

# What the headline number actually is. The fused score is a weighted-logit
# combination of signal outputs; it is NOT a calibrated probability.
SCORE_LABEL = "AI-generation detection score"
SCORE_NOTE = (
    "This is an uncalibrated detection score, not the probability that the "
    "video is AI-generated. It combines completed checks using fixed weights "
    "unless an optional learned fusion model is present. The heuristic checks "
    "and overall score have not been validated on a general population of videos."
)

LIMITATIONS = [
    "Automated results are not proof of authenticity; edited, compressed or "
    "unfamiliar videos can cause errors.",
    "The headline score is a heuristic fusion of signal outputs, not a "
    "calibrated probability.",
    "A 'no strong evidence' result means no check produced strong evidence "
    "against the AI hypothesis - it is not a confirmation that the video is "
    "authentic.",
]


def sha256_file(path: str | Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_info(path: str | Path, filename: str | None = None) -> dict:
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError:
        size = 0
    return {
        "name": filename or p.name,
        "size_bytes": size,
        "size_mb": round(size / (1024 * 1024), 3),
        "sha256": sha256_file(path) if p.exists() else None,
    }


def _score_separation(pct: float, verdict: str) -> str | None:
    """Describe distance from the neutral score; this is not statistical confidence."""
    if verdict == VERDICT_INCONCLUSIVE:
        return None
    dist = abs(pct - 50.0)
    if dist >= 30:
        return "high"
    if dist >= 15:
        return "medium"
    return "low"


def _join_labels(labels: list[str]) -> str:
    """Join signal labels with correct plural agreement for the verb."""
    if not labels:
        return "No checks"
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def build_result(
    scores: list[dict],
    ai_pct: float,
    file_meta: dict | None = None,
) -> dict:
    completed = [s for s in scores if s.get("status", "completed") == "completed"]
    signals = []
    for s in scores:
        signals.append(
            {
                "key": s["name"],
                "label": SIGNAL_LABELS.get(s["name"], s["name"]),
                "status": s.get("status", "completed"),
                "score": round(float(s.get("prob_ai", 0.5)) * 100, 1),
                "evidence": s.get("evidence", []),
            }
        )

    th = thresholds()
    ai_threshold = float(th.get("ai", 70.0))
    real_threshold = float(th.get("real", 30.0))
    min_signals = int(th.get("min_signals", 2))
    strong_ai_thr = float(th.get("strong_ai", 65.0))
    strong_real_thr = float(th.get("strong_real", 35.0))

    if len(completed) < min_signals:
        verdict = VERDICT_INCONCLUSIVE
    elif ai_pct >= ai_threshold:
        verdict = VERDICT_AI
    elif ai_pct <= real_threshold:
        verdict = VERDICT_REAL
    else:
        verdict = VERDICT_INCONCLUSIVE

    supporting = [s["label"] for s in signals if s["status"] == "completed" and s["score"] >= 60]
    contradicting = [s["label"] for s in signals if s["status"] == "completed" and s["score"] <= 40]
    missing = [s["label"] for s in signals if s["status"] != "completed"]

    strong_ai = [
        s["label"] for s in signals if s["status"] == "completed" and s["score"] >= strong_ai_thr
    ]
    strong_real = [
        s["label"]
        for s in signals
        if s["status"] == "completed" and s["score"] <= strong_real_thr
    ]
    disagreement = bool(strong_ai and strong_real)
    disagreement_note = ""
    if disagreement:
        ai_side = _join_labels(strong_ai)
        real_side = _join_labels(strong_real)
        disagreement_note = (
            f"Signals disagree: {ai_side} point towards AI, while {real_side} "
            "point towards real. Treat the verdict with caution — the evidence "
            "is mixed."
        )

    timeline = []
    timeline_meta: dict = {}
    for s in scores:
        if s["name"] == "s1_frame":
            details = s.get("details", {}) or {}
            timeline = details.get("timeline", [])
            video_meta = details.get("video", {}) or {}
            sampling = details.get("sampling", {}) or {}
            duration = float(video_meta.get("duration_sec") or 0.0)
            timeline_meta = {
                "duration_sec": round(duration, 3),
                "fps": video_meta.get("fps"),
                "total_frames": video_meta.get("total_frames"),
                "width": video_meta.get("width"),
                "height": video_meta.get("height"),
                "requested_frames": sampling.get("requested_frames"),
                "scored_frames": sampling.get("scored_frames"),
                "interval_sec": sampling.get("interval_sec"),
                "sampled_times": sampling.get("sampled_times", []),
                "source": "s1_frame",
            }
            break

    findings = build_all_findings(scores)

    limitations = list(LIMITATIONS)
    for s in signals:
        if s["status"] == "not_available":
            limitations.append(f"{s['label']} was not available and was excluded from the verdict.")
        elif s["status"] == "failed":
            limitations.append(f"{s['label']} failed to run and was excluded from the verdict.")

    return {
        "status": "completed",
        "verdict": verdict,
        "verdict_text": VERDICT_TEXT[verdict],
        "verdict_meaning": VERDICT_MEANING[verdict],
        "overall_score": round(ai_pct, 1) if verdict != VERDICT_INCONCLUSIVE else None,
        "ai_probability": round(ai_pct, 1),
        "score_label": SCORE_LABEL,
        "score_note": SCORE_NOTE,
        "score_is_calibrated": False,
        "score_separation": _score_separation(ai_pct, verdict),
        "signals_analysed": len(completed),
        "signals_total": len(signals),
        "signals": signals,
        "findings": findings,
        "timeline": timeline,
        "timeline_meta": timeline_meta,
        "supporting": supporting,
        "contradicting": contradicting,
        "missing": missing,
        "disagreement": disagreement,
        "disagreement_note": disagreement_note,
        "limitations": limitations,
        "detector_version": DETECTOR_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "file": file_meta or {},
    }
