from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

DETECTOR_VERSION = "0.1.0"

SIGNAL_LABELS = {
    "s1_frame": "Frame analysis",
    "s2_temporal": "Temporal consistency",
    "s3_freq": "Frequency analysis",
    "s4_audio": "Audio & AV",
    "s5_meta": "Metadata inspection",
    "s6_semantic": "Semantic cues",
}

VERDICT_AI = "likely_ai"
VERDICT_REAL = "likely_real"
VERDICT_INCONCLUSIVE = "inconclusive"

VERDICT_TEXT = {
    VERDICT_AI: "Likely AI-generated",
    VERDICT_REAL: "No strong AI-generation evidence detected",
    VERDICT_INCONCLUSIVE: "Inconclusive — more evidence needed",
}

AI_THRESHOLD = 70.0
REAL_THRESHOLD = 30.0
MIN_SIGNALS_FOR_VERDICT = 2
STRONG_AI = 65.0
STRONG_REAL = 35.0

LIMITATIONS = [
    "Automated results are not proof of authenticity; edited, compressed or "
    "unfamiliar videos can cause errors.",
    "Scores are heuristic unless a validated model is loaded; do not treat "
    "them as calibrated probabilities.",
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


def _confidence(pct: float, verdict: str) -> str | None:
    if verdict == VERDICT_INCONCLUSIVE:
        return None
    dist = abs(pct - 50.0)
    if dist >= 30:
        return "high"
    if dist >= 15:
        return "medium"
    return "low"


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

    if len(completed) < MIN_SIGNALS_FOR_VERDICT:
        verdict = VERDICT_INCONCLUSIVE
    elif ai_pct >= AI_THRESHOLD:
        verdict = VERDICT_AI
    elif ai_pct <= REAL_THRESHOLD:
        verdict = VERDICT_REAL
    else:
        verdict = VERDICT_INCONCLUSIVE

    supporting = [s["label"] for s in signals if s["status"] == "completed" and s["score"] >= 60]
    contradicting = [s["label"] for s in signals if s["status"] == "completed" and s["score"] <= 40]
    missing = [s["label"] for s in signals if s["status"] != "completed"]

    strong_ai = [
        s["label"] for s in signals if s["status"] == "completed" and s["score"] >= STRONG_AI
    ]
    strong_real = [
        s["label"] for s in signals if s["status"] == "completed" and s["score"] <= STRONG_REAL
    ]
    disagreement = bool(strong_ai and strong_real)
    disagreement_note = ""
    if disagreement:
        disagreement_note = (
            "Signals disagree: "
            f"{', '.join(strong_ai)} point to AI, while {', '.join(strong_real)} point to real. "
            "Treat the verdict with caution — the evidence is mixed."
        )

    timeline = []
    for s in scores:
        if s["name"] == "s1_frame":
            timeline = s.get("details", {}).get("timeline", [])
            break

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
        "overall_score": round(ai_pct, 1) if verdict != VERDICT_INCONCLUSIVE else None,
        "ai_probability": round(ai_pct, 1),
        "confidence": _confidence(ai_pct, verdict),
        "signals_analysed": len(completed),
        "signals_total": len(signals),
        "signals": signals,
        "timeline": timeline,
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
