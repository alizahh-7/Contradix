"""Tests for result formatting + CSV export (utils/results.py). No real models: conftest fakes only."""
import csv
import io
from datetime import datetime
from pathlib import Path

import pytest

from nlp import ComparisonConfig, compare_documents, compare_statements, compare_text_to_document, extract_document_from_path
from nlp.comparison import STATUS_CONSISTENT, STATUS_CONTRADICTION, STATUS_NEUTRAL, STATUS_REVIEW
from utils import components as ui
from utils.results import (
    CSV_COLUMNS,
    MODE_DOC,
    MODE_TEXT,
    MODE_TEXT_DOC,
    build_analysis,
    confidence_level,
    export_filename,
    to_csv_bytes,
    uncertainty_message,
)

SAMPLES = Path(__file__).resolve().parent.parent / "sample_data"
CFG = ComparisonConfig(similarity_threshold=0.30, max_pairs=40, top_k=3)
NOW = datetime(2026, 1, 2, 3, 4, 5)


def parse_csv(data: bytes) -> list[dict]:
    text = data.decode("utf-8")
    assert text.startswith("﻿")          # BOM so Excel reads UTF-8
    return list(csv.DictReader(io.StringIO(text.lstrip("﻿"))))


# --------------------------------------------------------------- confidence
@pytest.mark.parametrize("scores,level", [
    ((0.05, 0.90, 0.05), "High"),
    ((0.10, 0.65, 0.25), "Moderate"),
    ((0.40, 0.35, 0.25), "Low"),
    ((0.33, 0.34, 0.33), "Low"),
])
def test_confidence_levels(scores, level):
    assert confidence_level(*scores)["level"] == level


def test_uncertainty_message_mentions_uncertainty_for_low_confidence():
    assert "not confident" in uncertainty_message("Low", "Neutral")
    assert "not a verified fact" in uncertainty_message("High", "Contradiction")


# ------------------------------------------------------------------ Text vs Text
def test_text_analysis_reflects_backend_values_exactly(fake_backend):
    res = compare_statements("The library is open on Sundays.", "The library is closed every Sunday.")
    analysis = build_analysis(res, NOW)
    assert analysis.mode == MODE_TEXT and analysis.is_pair and len(analysis.rows) == 1
    row = analysis.rows[0]
    assert row["Status"] == STATUS_CONTRADICTION
    assert row["NLI label"] == res["relationship"] == "Contradiction"
    assert row["Contradiction"] == pytest.approx(res["contradiction_score"], abs=1e-4)
    assert row["Entailment"] == pytest.approx(res["entailment_score"], abs=1e-4)
    assert row["Neutral"] == pytest.approx(res["neutral_score"], abs=1e-4)
    assert row["Similarity"] == pytest.approx(res["similarity"], abs=1e-4)
    # premise / hypothesis order is preserved
    assert row["Premise"] == res["statement_a"] and row["Hypothesis"] == res["statement_b"]
    assert analysis.summary[STATUS_CONTRADICTION] == 1 and sum(analysis.summary.values()) == 1


def test_text_entailment_and_neutral_status(fake_backend):
    same = build_analysis(compare_statements("Employees carry ID cards.", "Employees carry ID cards."), NOW)
    assert same.rows[0]["Status"] == STATUS_CONSISTENT
    other = build_analysis(compare_statements("The exam starts at ten.", "Bananas grow in warm places."), NOW)
    assert other.rows[0]["Status"] == STATUS_NEUTRAL


def test_text_with_differing_percentages_is_flagged_for_review_not_contradiction(fake_backend):
    res = compare_statements("Students require 75% attendance to qualify for examinations.",
                             "Students with 60% attendance qualify for examinations.")
    assert res["relationship"] == "Neutral"                       # fake NLI says neutral
    row = build_analysis(res, NOW).rows[0]
    assert row["Status"] == STATUS_REVIEW                          # surfaced for a human, not called a contradiction
    assert "differing_percentage" in row["Flags"] and "75%" in row["Notes"] and "60%" in row["Notes"]


# ------------------------------------------------------------------ documents
def test_document_report_rows_keep_sources_and_scores(fake_backend):
    a = extract_document_from_path(str(SAMPLES / "circular_a_original_policy.pdf"))
    b = extract_document_from_path(str(SAMPLES / "circular_b_revised_notice.docx"))
    report = compare_documents(a, b, CFG)
    analysis = build_analysis(report, NOW)
    assert analysis.mode == MODE_DOC and len(analysis.rows) == len(report.pairs) > 0
    for row, pair in zip(analysis.rows, report.pairs):
        assert row["Premise"] == pair.premise.text and row["Hypothesis"] == pair.hypothesis.text
        assert row["Premise source"] == pair.premise.location and "page" in row["Premise source"]
        assert "paragraph" in row["Hypothesis source"]
        assert row["Status"] == pair.status and row["Contradiction"] == pytest.approx(pair.contradiction_score, abs=1e-4)
    assert analysis.summary == report.summary
    assert analysis.stats["candidate_pairs"] == report.candidate_pairs
    assert analysis.settings["Similarity threshold"] == "0.30"


def test_reverse_direction_columns_only_filled_when_bidirectional(fake_backend):
    a = extract_document_from_path(str(SAMPLES / "circular_a_original_policy.txt"))
    b = extract_document_from_path(str(SAMPLES / "circular_b_revised_notice.txt"))
    one = build_analysis(compare_documents(a, b, CFG), NOW)
    assert all(r["Reverse NLI label"] == "" for r in one.rows)
    both = build_analysis(compare_documents(a, b, ComparisonConfig(similarity_threshold=0.30, bidirectional=True)), NOW)
    assert all(r["Reverse NLI label"] in ("Entailment", "Contradiction", "Neutral") for r in both.rows)
    assert both.settings["Reverse NLI direction"] == "on"


def test_text_vs_document_analysis(fake_backend):
    doc = extract_document_from_path(str(SAMPLES / "circular_a_original_policy.txt"))
    report = compare_text_to_document("Students with 60% attendance are eligible for examinations.", doc,
                                      ComparisonConfig(similarity_threshold=0.30))
    analysis = build_analysis(report, NOW)
    assert analysis.mode == MODE_TEXT_DOC and analysis.rows
    assert all(r["Hypothesis source"] == "Entered statement" for r in analysis.rows)


def test_no_pairs_gives_empty_rows_and_message(fake_backend):
    doc = extract_document_from_path(str(SAMPLES / "circular_a_original_policy.txt"))
    report = compare_text_to_document("Bananas grow in tropical climates.", doc,
                                      ComparisonConfig(similarity_threshold=0.95))
    analysis = build_analysis(report, NOW)
    assert analysis.rows == [] and analysis.messages
    assert parse_csv(to_csv_bytes(analysis)) == []                 # header-only CSV still valid


def test_build_analysis_rejects_unknown_payload():
    with pytest.raises(TypeError):
        build_analysis(object())


# ----------------------------------------------------------------------- CSV
def test_csv_has_all_columns_and_matches_rows(fake_backend):
    res = compare_statements("The library is open on Sundays.", "The library is closed every Sunday.")
    analysis = build_analysis(res, NOW)
    data = to_csv_bytes(analysis)
    rows = parse_csv(data)
    assert list(rows[0].keys()) == CSV_COLUMNS
    assert rows[0]["Status"] == STATUS_CONTRADICTION and rows[0]["Premise"] == res["statement_a"]
    assert float(rows[0]["Contradiction"]) == pytest.approx(res["contradiction_score"], abs=1e-4)


def test_csv_neutralises_formula_injection(fake_backend):
    res = compare_statements("=HYPERLINK(\"http://x\") library is open", "+cmd library is closed on Sundays")
    analysis = build_analysis(res, NOW)
    rows = parse_csv(to_csv_bytes(analysis))
    assert rows[0]["Premise"].startswith("'=") and rows[0]["Hypothesis"].startswith("'+")


def test_csv_handles_commas_quotes_and_unicode(fake_backend):
    res = compare_statements('The "library", open daily - café', "The library is closed, naïve test")
    rows = parse_csv(to_csv_bytes(build_analysis(res, NOW)))
    assert rows[0]["Premise"] == res["statement_a"] and "café" in rows[0]["Premise"]


def test_export_filename_is_safe_and_dated(fake_backend):
    analysis = build_analysis(compare_statements("A library is open.", "A library is closed."), NOW)
    assert export_filename(analysis, "csv") == "contradix_text_vs_text_20260102_030405.csv"


# ------------------------------------------------------- HTML components
def test_components_escape_user_text():
    html = ui.statement_compare("<script>alert(1)</script>", "x<b>", "a & b", "y")
    assert "<script>" not in html and "&lt;script&gt;" in html and "a &amp; b" in html


def test_probability_bars_use_real_values_and_clamp():
    html = ui.prob_bars(0.1, 0.85, 0.05)
    assert "85.0%" in html and "width:85.0%" in html and "10.0%" in html
    assert "width:100.0%" in ui.prob_bar("x", 1.7, "danger") and "width:0.0%" in ui.prob_bar("x", -3, "danger")


@pytest.mark.parametrize("status,kind", [(STATUS_CONTRADICTION, "danger"), (STATUS_CONSISTENT, "success"),
                                         (STATUS_NEUTRAL, "neutral"), (STATUS_REVIEW, "warning")])
def test_classification_card_colour_follows_status(status, kind):
    html = ui.classification_card(status, "Neutral", {"entailment": 0.2, "contradiction": 0.2, "neutral": 0.6})
    assert f"cx-{kind}" in html and status in html


def test_low_confidence_card_shows_uncertainty():
    html = ui.classification_card(STATUS_NEUTRAL, "Neutral", {"entailment": 0.34, "contradiction": 0.33, "neutral": 0.33})
    assert "not confident" in html and "cx-conf-low" in html


def test_results_table_escapes_and_truncates():
    row = {"No": 1, "Status": STATUS_CONTRADICTION, "Confidence": "High", "Premise": "<i>" + "x" * 300,
           "Hypothesis": "h", "Similarity": 0.5, "Contradiction": 0.9, "Entailment": 0.05, "Neutral": 0.05}
    html = ui.results_table([row])
    assert "<i>" not in html and "&lt;i&gt;" in html and "…" in html and "0.90" in html


def test_highlight_in_context_marks_sentence_and_escapes():
    html = ui.highlight_in_context("Intro. Students need 75% & more. End.", "Students need 75% & more.")
    assert "<mark>Students need 75% &amp; more.</mark>" in html
