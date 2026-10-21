"""PDF report tests: generate real PDFs and read them back with PyMuPDF."""
from datetime import datetime
from pathlib import Path

import pymupdf
import pytest

from nlp import ComparisonConfig, compare_documents, compare_statements, compare_text_to_document, extract_document_from_path
from utils.report import MAX_PDF_PAIRS, build_pdf_report
from utils.results import AnalysisResult, build_analysis

SAMPLES = Path(__file__).resolve().parent.parent / "sample_data"
CFG = ComparisonConfig(similarity_threshold=0.30, max_pairs=40, top_k=3)
NOW = datetime(2026, 1, 2, 3, 4, 5)


def pdf_text(data: bytes) -> str:
    assert data.startswith(b"%PDF")
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        return "\n".join(page.get_text() for page in pdf)


def test_text_vs_text_report_contains_statements_scores_and_label(fake_backend):
    res = compare_statements("The library is open on Sundays.", "The library is closed every Sunday.")
    text = pdf_text(build_pdf_report(build_analysis(res, NOW)))
    for needle in ("CONTRADIX", "Beyond Words. Detect the Contradiction.", "Text vs Text",
                   "The library is open on Sundays.", "The library is closed every Sunday.",
                   "Statement A", "Statement B", "Potential contradiction", "Contradiction",
                   f"{res['contradiction_score']:.4f}", f"{res['similarity']:.4f}",
                   "2026-01-02 03:04", "Interpretation and limitations", "not a verified fact"):
        assert needle in text.replace("\n", " "), needle


def test_document_report_contains_sources_and_summary(fake_backend):
    a = extract_document_from_path(str(SAMPLES / "circular_a_original_policy.pdf"))
    b = extract_document_from_path(str(SAMPLES / "circular_b_revised_notice.docx"))
    report = compare_documents(a, b, CFG)
    data = build_pdf_report(build_analysis(report, NOW))
    flat = pdf_text(data).replace("\n", " ")
    assert "circular_a_original_policy.pdf" in flat and "circular_b_revised_notice.docx" in flat
    assert "Relevant pairs analysed with NLI" in flat and str(report.candidate_pairs) in flat
    assert "page 1" in flat and "paragraph" in flat                    # source references
    assert "Similarity threshold" in flat and "0.30" in flat           # settings used
    first = report.pairs[0]
    assert first.premise.text[:40].replace("\n", " ") in flat


def test_text_vs_document_report(fake_backend):
    doc = extract_document_from_path(str(SAMPLES / "circular_a_original_policy.txt"))
    report = compare_text_to_document("Students with 60% attendance are eligible for examinations.", doc,
                                      ComparisonConfig(similarity_threshold=0.30))
    flat = pdf_text(build_pdf_report(build_analysis(report, NOW))).replace("\n", " ")
    assert "Text vs Document" in flat and "Entered statement" in flat and "Evidence from" in flat


def test_report_with_no_pairs_still_builds_and_shows_notice(fake_backend):
    doc = extract_document_from_path(str(SAMPLES / "circular_a_original_policy.txt"))
    report = compare_text_to_document("Bananas grow in tropical climates.", doc,
                                      ComparisonConfig(similarity_threshold=0.95))
    flat = pdf_text(build_pdf_report(build_analysis(report, NOW))).replace("\n", " ")
    assert "No sentence pairs were analysed" in flat and "No verdict was produced" in flat


def test_report_survives_special_characters_and_markup(fake_backend):
    res = compare_statements("<b>Tom & Jerry</b> library is open → 中文 café", "Library closed Sundays <i>x")
    data = build_pdf_report(build_analysis(res, NOW))
    flat = pdf_text(data)
    assert "Tom & Jerry" in flat and "café" in flat          # escaped, not parsed as markup


def test_report_caps_very_large_result_sets(fake_backend):
    res = compare_statements("The library is open on Sundays.", "The library is closed every Sunday.")
    base = build_analysis(res, NOW)
    big = AnalysisResult(mode=base.mode, generated_at=NOW, summary=base.summary, sources=base.sources,
                         rows=[dict(base.rows[0], No=i) for i in range(1, MAX_PDF_PAIRS + 6)])
    flat = pdf_text(build_pdf_report(big)).replace("\n", " ")
    assert f"Only the first {MAX_PDF_PAIRS} of {MAX_PDF_PAIRS + 5} pairs" in flat


def test_reverse_direction_is_reported(fake_backend):
    a = extract_document_from_path(str(SAMPLES / "circular_a_original_policy.txt"))
    b = extract_document_from_path(str(SAMPLES / "circular_b_revised_notice.txt"))
    report = compare_documents(a, b, ComparisonConfig(similarity_threshold=0.30, bidirectional=True))
    flat = pdf_text(build_pdf_report(build_analysis(report, NOW))).replace("\n", " ")
    assert "Reverse direction" in flat and "Reverse NLI direction on" in flat.replace("  ", " ")
