from __future__ import annotations

from datetime import datetime


def _safe(text) -> str:
    s = str(text)
    return s.encode("latin-1", "replace").decode("latin-1")


def render_pdf(result: dict) -> bytes:
    from fpdf import FPDF

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "TruthLens - Forensic Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    version = result.get("detector_version", "n/a")
    generated = result.get("generated_at", datetime.now().isoformat(timespec="seconds"))
    pdf.cell(0, 6, _safe(f"Detector v{version} · Generated: {generated}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    overall = result.get("overall_score")
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, _safe(f"Verdict: {result.get('verdict_text', 'n/a')}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    score_txt = f"{overall}%" if overall is not None else "not produced (inconclusive)"
    score_label = result.get("score_label", "AI-generation detection score")
    pdf.cell(0, 6, _safe(f"{score_label}: {score_txt}"), new_x="LMARGIN", new_y="NEXT")
    separation = result.get("score_separation") or "n/a"
    pdf.cell(
        0,
        6,
        _safe(f"Distance from neutral score: {separation}"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(
        0, 6,
        _safe(f"Checks completed: {result.get('signals_analysed')}"
              f" / {result.get('signals_total')}"),
        new_x="LMARGIN", new_y="NEXT",
    )
    meaning = result.get("verdict_meaning")
    if meaning:
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(110, 110, 110)
        pdf.multi_cell(0, 5, _safe(meaning), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
    note = result.get("score_note")
    if note:
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(110, 110, 110)
        pdf.multi_cell(0, 5, _safe(note), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 11)
    pdf.ln(4)

    file_meta = result.get("file", {}) or {}
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "File verification", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _safe(f"Filename: {file_meta.get('name')}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, _safe(f"Size: {file_meta.get('size_mb')} MB"), new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(
        0, 6,
        _safe(f"SHA-256: {file_meta.get('sha256')}"),
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Forensic findings", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for f in result.get("findings", []):
        pdf.set_font("Helvetica", "B", 10)
        pdf.multi_cell(
            0, 5,
            _safe(f"{f.get('label', '')} [{f.get('status', '')}]"),
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.set_font("Helvetica", "", 9)
        measures = f.get("measures")
        if measures:
            pdf.multi_cell(0, 5, _safe(f"Measures: {measures}"), new_x="LMARGIN", new_y="NEXT")
        summary = f.get("summary")
        if summary:
            pdf.multi_cell(0, 5, _safe(summary), new_x="LMARGIN", new_y="NEXT")
        for fd in f.get("findings", []):
            direction = fd.get("direction", "")
            direction_txt = "" if direction == "no_effect" else f" [{direction.replace('_', ' ')}]"
            pdf.set_font("Helvetica", "B", 9)
            pdf.multi_cell(
                0, 5,
                _safe(f"{fd.get('title', '')}: {fd.get('measured', '')}{direction_txt}"),
                new_x="LMARGIN", new_y="NEXT",
            )
            pdf.set_font("Helvetica", "", 9)
            if fd.get("unit"):
                pdf.multi_cell(0, 4, _safe(f"Unit: {fd['unit']}"), new_x="LMARGIN", new_y="NEXT")
            if fd.get("interpretation"):
                pdf.multi_cell(
                    0, 4, _safe(f"Interpretation: {fd['interpretation']}"),
                    new_x="LMARGIN", new_y="NEXT",
                )
            if fd.get("limitation"):
                pdf.set_text_color(110, 110, 110)
                pdf.multi_cell(
                    0, 4, _safe(f"Limitation: {fd['limitation']}"),
                    new_x="LMARGIN", new_y="NEXT",
                )
                pdf.set_text_color(0, 0, 0)
        raw = f.get("raw_evidence") or []
        if raw:
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(130, 130, 130)
            pdf.multi_cell(
                0, 4, _safe("Raw measurements: " + ", ".join(raw)),
                new_x="LMARGIN", new_y="NEXT",
            )
            pdf.set_text_color(0, 0, 0)
        pdf.ln(2)
    pdf.ln(2)

    timeline = result.get("timeline") or []
    if timeline:
        meta = result.get("timeline_meta") or {}
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Evidence timeline", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        duration = meta.get("duration_sec")
        interval = meta.get("interval_sec")
        header = (
            f"{meta.get('scored_frames', len(timeline))} frame(s) sampled"
        )
        if duration:
            header += f" from a {duration}s video"
        if interval is not None:
            header += f", sampling interval about {interval}s"
        pdf.multi_cell(0, 5, _safe(header), new_x="LMARGIN", new_y="NEXT")
        for f in timeline:
            score = f.get("score")
            processed = f.get("processed")
            score_txt = f"{(score or 0) * 100:.1f}%" if processed else "not scored"
            frame_no = f.get("frame")
            frame_txt = f", frame {frame_no}" if frame_no is not None else ""
            pdf.multi_cell(
                0, 5,
                _safe(f"- t={f.get('t')}s{frame_txt}: classifier score {score_txt}"),
                new_x="LMARGIN", new_y="NEXT",
            )
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(130, 130, 130)
        pdf.multi_cell(
            0, 4,
            _safe("Frame previews are available in the interactive web report."),
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    if result.get("supporting") or result.get("contradicting"):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Evidence summary", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        supporting = ", ".join(result.get("supporting") or ["None"])
        contradicting = ", ".join(result.get("contradicting") or ["None"])
        pdf.multi_cell(0, 6, _safe(f"Supporting AI: {supporting}"),
                       new_x="LMARGIN", new_y="NEXT")
        pdf.multi_cell(0, 6, _safe(f"Contradicting: {contradicting}"),
                       new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    if result.get("disagreement"):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Cross-signal disagreement", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _safe(result.get("disagreement_note", "")),
                       new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Limitations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for item in result.get("limitations", []):
        pdf.multi_cell(0, 6, _safe(f"- {item}"), new_x="LMARGIN", new_y="NEXT")

    out = pdf.output()
    return bytes(out)
