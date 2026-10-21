"""PDF analysis report (ReportLab - pure Python, no system dependencies).

`build_pdf_report(analysis)` turns an `AnalysisResult` (see utils/results.py) into PDF bytes
containing: the compared statements, source references, NLI predictions, probabilities,
similarity, confidence, notes and a limitations statement. Nothing here calls a model.
"""
from __future__ import annotations

import io
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from utils.results import STATUS_STYLE, AnalysisResult, uncertainty_message

NAVY = colors.HexColor("#0B1020")
PURPLE = colors.HexColor("#7C3AED")
BLUE = colors.HexColor("#3B82F6")
MUTED = colors.HexColor("#64748B")
LINE = colors.HexColor("#E2E8F0")
MAX_PDF_PAIRS = 200   # keeps the file size and build time sane for very large result sets


def _t(text) -> str:
    """Make arbitrary text safe for ReportLab's built-in Helvetica (WinAnsi) font + XML markup."""
    s = str(text).encode("cp1252", "replace").decode("cp1252")
    return escape(s)


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold", fontSize=24,
                                textColor=NAVY, alignment=TA_LEFT, spaceAfter=2),
        "tag": ParagraphStyle("tag", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=10,
                              textColor=PURPLE, spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13,
                             textColor=NAVY, spaceBefore=12, spaceAfter=6),
        "body": ParagraphStyle("b", parent=base["Normal"], fontName="Helvetica", fontSize=9.5, leading=13),
        "small": ParagraphStyle("s", parent=base["Normal"], fontName="Helvetica", fontSize=8, leading=10.5,
                                textColor=MUTED),
        "label": ParagraphStyle("l", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=8,
                                textColor=MUTED, leading=10),
        "quote": ParagraphStyle("q", parent=base["Normal"], fontName="Helvetica", fontSize=9.5, leading=13,
                                leftIndent=0),
    }


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "CONTRADIX - Beyond Words. Detect the Contradiction.")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _kv_table(pairs: list[tuple[str, str]], st: dict, col_widths=(48 * mm, 120 * mm)) -> Table:
    data = [[Paragraph(_t(k), st["label"]), Paragraph(_t(v), st["body"])] for k, v in pairs]
    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return tbl


def _pair_block(row: dict, st: dict, show_no: bool) -> KeepTogether:
    kind, hex_colour = STATUS_STYLE[row["Status"]]
    colour = colors.HexColor(hex_colour)
    title = f"#{row['No']}  " if show_no else ""
    header = Table([[Paragraph(f"<b>{_t(title + row['Status'])}</b>", ParagraphStyle(
        "hd", fontName="Helvetica-Bold", fontSize=10, textColor=colors.white)),
        Paragraph(f"NLI label: {_t(row['NLI label'])} | Model confidence: {_t(row['Confidence'])}",
                  ParagraphStyle("hr", fontName="Helvetica", fontSize=8.5, textColor=colors.white,
                                 alignment=2))]],
        colWidths=[84 * mm, 84 * mm])
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colour), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))

    statements = Table([
        [Paragraph("PREMISE", st["label"]), Paragraph("HYPOTHESIS", st["label"])],
        [Paragraph(_t(row["Premise"]), st["quote"]), Paragraph(_t(row["Hypothesis"]), st["quote"])],
        [Paragraph("Source: " + _t(row["Premise source"]), st["small"]),
         Paragraph("Source: " + _t(row["Hypothesis source"]), st["small"])],
    ], colWidths=[84 * mm, 84 * mm])
    statements.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE), ("LINEAFTER", (0, 0), (0, -1), 0.5, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))

    score_head = ["Similarity", "Entailment", "Contradiction", "Neutral"]
    score_vals = [f"{row['Similarity']:.4f}", f"{row['Entailment']:.4f}",
                  f"{row['Contradiction']:.4f}", f"{row['Neutral']:.4f}"]
    scores = Table([[Paragraph(h, st["label"]) for h in score_head],
                    [Paragraph(f"<b>{v}</b>", st["body"]) for v in score_vals]],
                   colWidths=[42 * mm] * 4)
    scores.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 3),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))

    parts = [header, statements, scores]
    if row["Reverse NLI label"]:
        parts.append(Paragraph(
            f"Reverse direction (hypothesis -> premise): {_t(row['Reverse NLI label'])} "
            f"(entailment {row['Reverse entailment']:.3f}, contradiction {row['Reverse contradiction']:.3f}, "
            f"neutral {row['Reverse neutral']:.3f})", st["small"]))
    if row["Notes"]:
        parts.append(Paragraph("Note: " + _t(row["Notes"]), st["small"]))
    parts.append(Spacer(1, 9))
    return KeepTogether(parts)


def build_pdf_report(analysis: AnalysisResult) -> bytes:
    """Render the analysis as a formatted A4 PDF and return its bytes."""
    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm,
                            title=f"CONTRADIX analysis report - {analysis.mode}", author="CONTRADIX")
    story = [
        Paragraph("CONTRADIX", st["title"]),
        Paragraph("Beyond Words. Detect the Contradiction.", st["tag"]),
        HRFlowable(width="100%", thickness=1.2, color=PURPLE, spaceAfter=8),
        Paragraph("Analysis report", st["h2"]),
    ]

    meta = [("Comparison mode", analysis.mode),
            ("Generated", f"{analysis.generated_at:%Y-%m-%d %H:%M}"),
            ("NLI model", "cross-encoder/nli-deberta-v3-small"),
            ("Similarity model", "sentence-transformers/all-MiniLM-L6-v2")]
    for src in analysis.sources:
        meta.append(("Source", src))
    for k, v in analysis.settings.items():
        meta.append((k, v))
    story.append(_kv_table(meta, st))

    # ---- summary
    story.append(Paragraph("Summary", st["h2"]))
    if analysis.stats:
        s = analysis.stats
        best = "-" if s["best_similarity"] is None else f"{s['best_similarity']:.2f}"
        story.append(_kv_table([
            ("Sentences extracted (premise side)", str(s["premise_count"])),
            ("Sentences extracted (hypothesis side)", str(s["hypothesis_count"])),
            ("Possible sentence combinations", str(s["total_possible_pairs"])),
            ("Relevant pairs analysed with NLI", str(s["candidate_pairs"])),
            ("Highest similarity found", best),
        ], st))
        story.append(Spacer(1, 6))
    summary_rows = [[Paragraph(f"<b>{_t(name)}</b>", st["body"]), Paragraph(str(count), st["body"])]
                    for name, count in analysis.summary.items()]
    if summary_rows:
        tbl = Table(summary_rows, colWidths=[110 * mm, 58 * mm])
        style = [("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("TOPPADDING", (0, 0), (-1, -1), 3),
                 ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        for i, name in enumerate(analysis.summary):
            style.append(("LINEBEFORE", (0, i), (0, i), 4, colors.HexColor(STATUS_STYLE[name][1])))
        tbl.setStyle(TableStyle(style))
        story.append(tbl)

    for msg in analysis.messages:
        story.append(Spacer(1, 4))
        story.append(Paragraph("Notice: " + _t(msg), st["small"]))

    # ---- results
    story.append(Paragraph("Detailed results", st["h2"]))
    if not analysis.rows:
        story.append(Paragraph("No sentence pairs were analysed for this comparison.", st["body"]))
    else:
        rows = analysis.rows[:MAX_PDF_PAIRS]
        show_no = len(analysis.rows) > 1
        if analysis.is_pair:
            r0 = rows[0]
            story.append(Paragraph(_t(uncertainty_message(r0["Confidence"], r0["NLI label"])), st["small"]))
            story.append(Spacer(1, 6))
        for row in rows:
            story.append(_pair_block(row, st, show_no))
        if len(analysis.rows) > MAX_PDF_PAIRS:
            story.append(Paragraph(
                f"Only the first {MAX_PDF_PAIRS} of {len(analysis.rows)} pairs are shown here. "
                "The CSV export contains all of them.", st["small"]))

    story.append(Paragraph("Interpretation and limitations", st["h2"]))
    story.append(Paragraph(
        "NLI is directional: it asks whether the hypothesis follows from the premise, so swapping the two can change "
        "the result. Semantic similarity only measures topical closeness, not agreement. Probabilities come from a "
        "small pretrained model and can be wrong; different numbers or dates are surfaced for human review and are "
        "not automatically logical contradictions. CONTRADIX reports <i>potential</i> contradictions - please verify "
        "them against the original documents.", st["body"]))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
