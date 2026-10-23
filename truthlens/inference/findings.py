"""Human-readable forensic findings.

This module turns the raw measurements produced by the individual signals into
structured, explainable findings. It never invents a meaning: every
interpretation below is derived from what the signal code actually computes.

Rules followed here:
- A measurement is only given a directional meaning when the signal code
  applies a documented, fixed rule to it.
- Where a measurement only feeds an arbitrary hand-tuned logistic with no
  validated threshold, it is reported as an internal measurement and the
  interpretation is explicitly declared unavailable.
- Unavailable and failed checks are reported as such. A missing check is
  never treated as evidence of authenticity.
"""

from __future__ import annotations

# Direction labels used by the UI and the PDF.
SUPPORTS = "supports"
CONTRADICTS = "contradicts"
NEUTRAL = "no_effect"
INCONCLUSIVE = "inconclusive"

STATUS_COMPLETED = "completed"
STATUS_NOT_AVAILABLE = "not_available"
STATUS_FAILED = "failed"


def _finding(
    key: str,
    title: str,
    measured: str,
    interpretation: str,
    direction: str,
    explanation: str,
    limitation: str | None = None,
    unit: str | None = None,
    threshold: str | None = None,
    references: list | None = None,
) -> dict:
    return {
        "key": key,
        "title": title,
        "measured": measured,
        "unit": unit,
        "threshold": threshold,
        "interpretation": interpretation,
        "direction": direction,
        "explanation": explanation,
        "limitation": limitation,
        "references": references or [],
    }


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _parse_evidence(evidence: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in evidence or []:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Per-signal finding builders
# --------------------------------------------------------------------------


def _s1_frame(sig: dict) -> list[dict]:
    """Frame analysis: a learned classifier applied to sampled frames.

    The score is the mean classifier output over the sampled frames. When the
    linear probe is active the value is the probe's decision probability for
    the frame; when the raw ONNX model is active it is that model's softmax
    probability; the face heuristic is a fixed constant lookup.
    """
    details = sig.get("details") or {}
    raw = _parse_evidence(sig.get("evidence"))
    method = details.get("method") or raw.get("s1_method") or "unknown"
    n_frames = _to_float(raw.get("s1_frames"))
    n_faces = _to_float(raw.get("s1_faces"))
    timeline = details.get("timeline") or []
    processed = [t for t in timeline if t.get("processed")]

    method_text = {
        "cnn_probe": (
            "A linear probe on frozen ViT embeddings (Community Forensics ViT) "
            "was applied to the face crop of each sampled frame."
        ),
        "cnn_vit": (
            "The pretrained deepfake ViT (Community Forensics) was applied to "
            "the face crop of each sampled frame."
        ),
        "face-heuristic": (
            "No image model was available, so a fixed face-detection heuristic "
            "was used instead."
        ),
    }.get(method, "The frame classifier was applied to each sampled frame.")

    limitations = []
    if method == "face-heuristic":
        limitations.append(
            "This fallback uses only whether a face was detected; it carries no "
            "learned evidence about manipulation."
        )
    if timeline and processed:
        span = processed[-1].get("t", 0) - processed[0].get("t", 0)
        limitations.append(
            f"Only {len(processed)} frame(s) were sampled across a {span:.2f}s "
            "span of the timeline; the video was not analysed frame by frame."
        )
    if not processed:
        limitations.append("No sampled frame could be scored.")

    findings = [
        _finding(
            key="s1_frames",
            title="Frames analysed",
            measured=(
                f"{int(n_frames)} frame(s) scored"
                if n_frames is not None
                else "Frame count not reported"
            ),
            interpretation="The classifier score below is the mean over these frames.",
            direction=NEUTRAL,
            explanation=method_text,
            limitation="; ".join(limitations) or None,
            references=[{"t": t.get("t"), "score": t.get("score")} for t in processed],
        )
    ]

    if n_faces is not None:
        findings.append(
            _finding(
                key="s1_faces",
                title="Faces detected in sampled frames",
                measured=f"{int(n_faces)} face detection(s) across the sampled frames",
                unit="count",
                interpretation=(
                    "Face detections are the regions the classifier was applied to; "
                    "a count above the number of frames means multiple faces were "
                    "present in some frames."
                ),
                direction=NEUTRAL,
                explanation=(
                    "The classifier is run on the largest detected face in each "
                    "frame, or on the whole frame if no face was found."
                ),
                limitation=(
                    "Face detection uses the OpenCV Haar cascade, which is a "
                    "reliable detector but not a manipulation detector."
                ),
            )
        )

    score = float(sig.get("prob_ai", 0.5))
    direction = SUPPORTS if score >= 0.6 else (CONTRADICTS if score <= 0.4 else NEUTRAL)
    findings.append(
        _finding(
            key="s1_score",
            title="Mean frame classifier score",
            measured=f"{score * 100:.1f}%",
            unit="0-100 scale",
            threshold="Model decision boundary (not an equal-evidence threshold)",
            interpretation=(
                "The classifier leans towards AI-generated for these frames."
                if direction == SUPPORTS
                else (
                    "The classifier leans towards not-AI for these frames."
                    if direction == CONTRADICTS
                    else "The classifier did not lean strongly in either direction."
                )
            ),
            direction=direction,
            explanation=method_text,
            limitation=(
                "This value is a model score on a 0-100 scale. It is not a "
                "calibrated probability that the video is AI-generated."
            ),
        )
    )
    return findings


def _s2_temporal(sig: dict) -> list[dict]:
    """Temporal consistency: optical flow and frame-difference statistics.

    Both measurements only feed a hand-tuned logistic in the signal code. There
    is no validated threshold for either value, so no directional meaning is
    claimed here.
    """
    raw = _parse_evidence(sig.get("evidence"))
    flow = _to_float(raw.get("s2_flow_mean"))
    diff = _to_float(raw.get("s2_diffvar_mean"))
    limitations = (
        "The motion and difference statistics are combined by a hand-tuned "
        "logistic function in the detector; it was not validated against a "
        "reference dataset, so a high or low reading cannot be read as "
        "evidence of manipulation on its own."
    )
    findings = [
        _finding(
            key="s2_flow",
            title="Mean optical-flow magnitude",
            measured=f"{_fmt(flow)}" if flow is not None else "not measured",
            unit="pixels per frame (average over sampled frame pairs)",
            interpretation=(
                "Interpretation unavailable: no validated threshold exists for "
                "this measurement."
            ),
            direction=NEUTRAL,
            explanation=(
                "Measures how much pixel content moved between consecutive "
                "sampled frames using Farneback optical flow. Genuine camera "
                "motion and generated motion both produce movement."
            ),
            limitation=limitations,
        ),
        _finding(
            key="s2_diffvar",
            title="Mean frame-difference variance",
            measured=f"{_fmt(diff)}" if diff is not None else "not measured",
            unit="variance of absolute pixel difference (8-bit scale)",
            interpretation=(
                "Interpretation unavailable: no validated threshold exists for "
                "this measurement."
            ),
            direction=NEUTRAL,
            explanation=(
                "Measures how much the brightness pattern changed between "
                "consecutive sampled frames. Cuts and scene changes also raise "
                "this value."
            ),
            limitation=limitations,
        ),
    ]
    return findings


def _s3_freq(sig: dict) -> list[dict]:
    """Frequency analysis: spectrum and block statistics.

    Same situation as the temporal signal: the two measurements feed a
    hand-tuned logistic without a validated threshold.
    """
    raw = _parse_evidence(sig.get("evidence"))
    fft_var = _to_float(raw.get("s3_fftvar"))
    block_fft_var = _to_float(raw.get("s3_blockfftvar"))
    limitations = (
        "These spectrum statistics are combined by a hand-tuned logistic "
        "function in the detector and were not validated against a reference "
        "dataset. Compression, resizing and camera noise all change them, so a "
        "reading cannot be read as evidence of manipulation on its own."
    )
    return [
        _finding(
            key="s3_fftvar",
            title="Spectral magnitude variance",
            measured=f"{_fmt(fft_var)}" if fft_var is not None else "not measured",
            unit="variance of log(1+|FFT|) over the centred spectrum",
            interpretation=(
                "Interpretation unavailable: no validated threshold exists for "
                "this measurement."
            ),
            direction=NEUTRAL,
            explanation=(
                "Summarises how the frame's brightness is distributed across "
                "spatial frequencies. Generated images and heavily compressed "
                "video both alter this distribution."
            ),
            limitation=limitations,
        ),
        _finding(
            key="s3_dctvar",
            title="Per-block 1D FFT variance",
            measured=(
                f"{_fmt(block_fft_var)}" if block_fft_var is not None else "not measured"
            ),
            unit="variance of |1D FFT| across 8x8 pixel blocks",
            interpretation=(
                "Interpretation unavailable: no validated threshold exists for "
                "this measurement."
            ),
            direction=NEUTRAL,
            explanation=(
                "Applies a one-dimensional fast Fourier transform to flattened "
                "8x8 pixel blocks and summarizes coefficient variation. This is "
                "not a discrete cosine transform, and no validated manipulation "
                "threshold is available."
            ),
            limitation=limitations,
        ),
    ]


def _s4_audio(sig: dict) -> list[dict]:
    raw = _parse_evidence(sig.get("evidence"))
    mfcc = _to_float(raw.get("s4_mfccstd"))
    zcr = _to_float(raw.get("s4_zcr"))
    flat = _to_float(raw.get("s4_flat"))
    limitations = (
        "The audio measurements are combined by a hand-tuned formula in the "
        "detector and were not validated against a reference dataset."
    )
    return [
        _finding(
            key="s4_mfccstd",
            title="MFCC standard deviation",
            measured=f"{_fmt(mfcc)}" if mfcc is not None else "not measured",
            unit="mean standard deviation across 13 MFCC coefficients",
            interpretation=(
                "Interpretation unavailable: no validated threshold exists for "
                "this measurement."
            ),
            direction=NEUTRAL,
            explanation=(
                "Describes how much the spectral envelope of the audio varies "
                "over time. Speech and music both produce large values."
            ),
            limitation=limitations,
        ),
        _finding(
            key="s4_zcr",
            title="Zero-crossing rate",
            measured=f"{_fmt(zcr, 4)}" if zcr is not None else "not measured",
            unit="crossings per sample (0-1 range)",
            interpretation=(
                "Interpretation unavailable: no validated threshold exists for "
                "this measurement."
            ),
            direction=NEUTRAL,
            explanation=(
                "Counts how often the waveform crosses zero per sample. "
                "High-pitched or noisy content crosses more often."
            ),
            limitation=limitations,
        ),
        _finding(
            key="s4_flat",
            title="Spectral flatness",
            measured=f"{_fmt(flat, 4)}" if flat is not None else "not measured",
            unit="0 (tonal) to 1 (noise-like)",
            interpretation=(
                "Interpretation unavailable: no validated threshold exists for "
                "this measurement."
            ),
            direction=NEUTRAL,
            explanation=(
                "Measures how noise-like the audio spectrum is. Silence and "
                "noise sit near 1; tonal music sits near 0."
            ),
            limitation=limitations,
        ),
    ]


def _s5_meta(sig: dict) -> list[dict]:
    raw = _parse_evidence(sig.get("evidence"))
    tags_present = raw.get("tags_present")
    nb_streams = raw.get("nb_streams")
    tags_present_bool = None
    if isinstance(tags_present, str):
        tags_present_bool = tags_present.strip().lower() in {"true", "1", "yes"}
    limitations = (
        "Metadata presence is a very weak indicator: re-encoding, uploading "
        "and platform processing routinely strip camera metadata, and "
        "synthetic content can carry camera metadata."
    )
    findings = [
        _finding(
            key="s5_tags",
            title="Container metadata tags",
            measured=(
                "Present" if tags_present_bool else ("Absent" if tags_present_bool is False else "not measured")
            ),
            interpretation=(
                "The detector applies a small fixed adjustment to its score when "
                "camera-style tags are missing."
            )
            if tags_present_bool is not None
            else "Not measured.",
            direction=NEUTRAL,
            explanation=(
                "Checks whether the file's container carries author, encoder or "
                "camera tags. A synthetic file often has none, but so does any "
                "re-encoded or uploaded video."
            ),
            limitation=limitations,
        )
    ]
    if nb_streams is not None:
        findings.append(
            _finding(
                key="s5_streams",
                title="Number of media streams",
                measured=str(nb_streams),
                unit="count (video + audio + other streams)",
                interpretation="Descriptive only; carries no detection weight.",
                direction=NEUTRAL,
                explanation=(
                    "Reports how many separate streams the container holds, for "
                    "example one video stream plus one audio stream."
                ),
                limitation="Structural information only.",
            )
        )
    return findings


def _s6_semantic(sig: dict) -> list[dict]:
    return [
        _finding(
            key="s6_placeholder",
            title="Semantic cue check",
            measured="Not implemented",
            interpretation="No interpretation available.",
            direction=INCONCLUSIVE,
            explanation=(
                "A CLIP-style semantic check was planned but is not implemented; "
                "this check contributes nothing to the verdict."
            ),
            limitation="This check is a placeholder and never runs.",
        )
    ]


_BUILDERS = {
    "s1_frame": _s1_frame,
    "s2_temporal": _s2_temporal,
    "s3_freq": _s3_freq,
    "s4_audio": _s4_audio,
    "s5_meta": _s5_meta,
    "s6_semantic": _s6_semantic,
}

SIGNAL_LABELS = {
    "s1_frame": "Frame analysis",
    "s2_temporal": "Temporal consistency",
    "s3_freq": "Frequency analysis",
    "s4_audio": "Audio analysis",
    "s5_meta": "Metadata inspection",
    "s6_semantic": "Semantic check (not implemented)",
}

SIGNAL_MEASURES = {
    "s1_frame": "Learned image classifier applied to sampled frames",
    "s2_temporal": "Motion consistency between sampled frames",
    "s3_freq": "Spatial frequency distribution of sampled frames",
    "s4_audio": "Audio spectrum statistics of the soundtrack (no AV synchronization check)",
    "s5_meta": "Container/encoder metadata of the file",
    "s6_semantic": "Semantic analysis (not implemented)",
}


def build_findings(sig: dict) -> dict:
    """Build the structured, human-readable finding set for one signal."""
    name = sig.get("name") or sig.get("key") or ""
    status = sig.get("status", STATUS_COMPLETED)
    label = sig.get("label") or SIGNAL_LABELS.get(name, name)
    measures = SIGNAL_MEASURES.get(name, "Forensic measurement")

    base = {
        "signal": name,
        "label": label,
        "status": status,
        "measures": measures,
        "findings": [],
        "raw_evidence": list(sig.get("evidence") or []),
        "summary": "",
    }

    if status == STATUS_NOT_AVAILABLE:
        base["summary"] = (
            "This check did not run on this file. It was excluded from the "
            "verdict, and its absence is not evidence that the video is "
            "authentic."
        )
        return base

    if status == STATUS_FAILED:
        base["summary"] = (
            "This check started but could not complete. It was excluded from "
            "the verdict, and its absence is not evidence that the video is "
            "authentic."
        )
        return base

    builder = _BUILDERS.get(name)
    if builder is None:
        base["findings"] = [
            _finding(
                key="raw",
                title="Raw measurements",
                measured=", ".join(sig.get("evidence") or []) or "none",
                interpretation="Interpretation unavailable for this signal.",
                direction=INCONCLUSIVE,
                explanation="No structured interpretation is defined for this signal.",
                limitation="Raw output is shown unchanged.",
            )
        ]
        base["summary"] = "This check completed and produced raw measurements."
        return base

    base["findings"] = builder(sig)
    base["summary"] = _summarise(base["findings"], float(sig.get("prob_ai", 0.5)))
    return base


def _summarise(findings: list[dict], score: float) -> str:
    supports = [f for f in findings if f["direction"] == SUPPORTS]
    contradicts = [f for f in findings if f["direction"] == CONTRADICTS]
    neutral = [f for f in findings if f["direction"] == NEUTRAL]
    if supports and not contradicts:
        return (
            "This check completed and its result leans towards AI-generated "
            "content."
        )
    if contradicts and not supports:
        return (
            "This check completed and its result leans towards content that is "
            "not AI-generated."
        )
    if supports and contradicts:
        return "This check completed and produced mixed evidence."
    return (
        "This check completed. Its measurements did not move the assessment in "
        "either direction."
    )


def build_all_findings(signals: list[dict]) -> list[dict]:
    return [build_findings(s) for s in signals]
