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
    pdf.cell(0, 6, _safe(f"Generated: {datetime.now().isoformat(timespec='seconds')}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    overall = result.get("overall_score")
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, _safe(f"Verdict: {result.get('verdict_text', 'n/a')}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    score_txt = f"{overall}%" if overall is not None else "not produced (inconclusive)"
    pdf.cell(0, 6, _safe(f"AI probability: {score_txt}"), new_x="LMARGIN", new_y="NEXT")
    conf = result.get("confidence") or "n/a"
    pdf.cell(0, 6, _safe(f"Confidence: {conf}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(
        0, 6,
        _safe(f"Signals analysed: {result.get('signals_analysed')}"
              f" / {result.get('signals_total')}"),
        new_x="LMARGIN", new_y="NEXT",
    )
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
    pdf.cell(0, 8, "Signal findings", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for s in result.get("signals", []):
        score = f"{s['score']}%" if s.get("status") == "completed" else "-"
        findings = ", ".join(s.get("evidence") or [])
        pdf.multi_cell(
            0, 6,
            _safe(f"[{s.get('status')}] {s.get('label')}: {score} | {findings}"),
            new_x="LMARGIN", new_y="NEXT",
        )
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

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Limitations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for item in result.get("limitations", []):
        pdf.multi_cell(0, 6, _safe(f"- {item}"), new_x="LMARGIN", new_y="NEXT")

    out = pdf.output()
    return bytes(out)
