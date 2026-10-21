"""Tests for the shared comparison pipeline (retrieval + NLI + verdict rules).

Heavy models are replaced by the fakes in conftest.py (fake_backend fixture); the real
extraction, segmentation, retrieval and verdict logic runs. Tests marked `model` use the
real pretrained models instead.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from nlp import (
    ComparisonConfig,
    compare_documents,
    compare_statements,
    compare_text_to_document,
    extract_document,
    extract_document_from_path,
    inference,
)
from nlp.comparison import (
    STATUS_CONSISTENT,
    STATUS_CONTRADICTION,
    STATUS_NEUTRAL,
    STATUS_REVIEW,
    extract_values,
    find_value_discrepancies,
    retrieve_candidate_pairs,
)

SAMPLES = Path(__file__).resolve().parent.parent / "sample_data"
CFG = ComparisonConfig(similarity_threshold=0.30, max_pairs=40, top_k=3)
skip_models = pytest.mark.skipif(os.environ.get("CONTRADIX_SKIP_MODEL_TESTS") == "1",
                                 reason="CONTRADIX_SKIP_MODEL_TESTS=1")


@pytest.fixture
def circular_a():
    return extract_document_from_path(str(SAMPLES / "circular_a_original_policy.pdf"))


@pytest.fixture
def circular_b():
    return extract_document_from_path(str(SAMPLES / "circular_b_revised_notice.docx"))


def find(report, needle_premise=None, needle_hyp=None):
    for p in report.pairs:
        if (needle_premise is None or needle_premise in p.premise.text) and \
           (needle_hyp is None or needle_hyp in p.hypothesis.text):
            return p
    raise AssertionError(f"pair not found: {needle_premise!r} / {needle_hyp!r}")


# ------------------------------------------------- value / context detection
def test_extract_values_normalises_dates_percentages_durations():
    v = extract_values("Submit by November 15, within three days, with 75% attendance.")
    assert v["date"] == {"Nov 15"} and v["percentage"] == {"75%"} and v["duration"] == {"3 days"}
    assert extract_values("on 20th of November")["date"] == {"Nov 20"}
    assert extract_values("a 75 percent rule")["percentage"] == {"75%"}


def test_differing_values_need_same_subject():
    same_subject = find_value_discrepancies("Minimum attendance is 75% for students.",
                                            "Minimum attendance is 65% for students.")
    assert same_subject and same_subject[0]["kind"] == "percentage"
    # numbers differ but the sentences are about different things -> NOT flagged
    assert find_value_discrepancies("The hall opens 30 minutes early.",
                                    "Attendance must be 75% overall.") == []


def test_equal_values_are_not_discrepancies():
    assert find_value_discrepancies("Late work is allowed within three days.",
                                    "Late work is accepted within 3 days.") == []


# ---------------------------------------------------------------- retrieval
def test_retrieval_threshold_topk_and_cap():
    a = np.array([[1, 0], [0, 1]], dtype=float)
    b = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.6, 0.8]], dtype=float)
    pairs = retrieve_candidate_pairs(a, b, threshold=0.95, top_k=1, max_pairs=10)
    assert (0, 0, pytest.approx(1.0)) in [(i, j, s) for i, j, s in pairs]
    assert all(s >= 0.95 for _, _, s in pairs)
    assert len(retrieve_candidate_pairs(a, b, threshold=0.0, top_k=4, max_pairs=3)) == 3
    sims = [s for _, _, s in retrieve_candidate_pairs(a, b, 0.0, 4, 8)]
    assert sims == sorted(sims, reverse=True)


def test_retrieval_single_direction_for_text_vs_document():
    stmt = np.array([[1.0, 0.0]])
    docs = np.array([[1, 0], [0.99, 0.1], [0, 1]], dtype=float)
    pairs = retrieve_candidate_pairs(stmt, docs, threshold=0.9, top_k=5, max_pairs=5, both_directions=False)
    assert {j for _, j, _ in pairs} == {0, 1}


def test_config_validation():
    for bad in (ComparisonConfig(max_pairs=0), ComparisonConfig(top_k=0),
                ComparisonConfig(contradiction_threshold=1.5), ComparisonConfig(similarity_threshold=2)):
        with pytest.raises(ValueError):
            bad.validate()


# ------------------------------------------------------- Document vs Document
def test_doc_vs_doc_finds_contradictions_review_items_and_agreement(fake_backend, circular_a, circular_b):
    report = compare_documents(circular_a, circular_b, CFG)
    assert report.mode == "Document vs Document"
    assert report.premise_count == circular_a.num_sentences
    assert report.hypothesis_count == circular_b.num_sentences

    # retrieval: far fewer NLI calls than all n*m combinations
    assert 0 < report.candidate_pairs < report.total_possible_pairs
    assert len(fake_backend.nli_calls) == report.candidate_pairs

    late = find(report, "No late submissions", "Late submissions will be permitted")
    assert late.status == STATUS_CONTRADICTION

    attendance = find(report, "75%", "65%")
    assert attendance.status == STATUS_REVIEW            # numbers alone are NOT a contradiction
    assert "differing_percentage" in attendance.flags
    assert attendance.relationship == "Neutral"

    deadline = find(report, "November 15", "November 20")
    assert deadline.status == STATUS_REVIEW
    assert "differing_date" in deadline.flags
    assert any("cannot tell which document is valid" in n for n in deadline.notes)

    library = find(report, "library is open", "library is open")
    assert library.status == STATUS_CONSISTENT

    s = report.summary
    assert s[STATUS_CONTRADICTION] >= 1 and s[STATUS_REVIEW] >= 2 and s[STATUS_CONSISTENT] >= 2


def test_doc_vs_doc_keeps_source_locations(fake_backend, circular_a, circular_b):
    report = compare_documents(circular_a, circular_b, CFG)
    pair = find(report, "75%", "65%")
    assert pair.premise.source == "circular_a_original_policy.pdf" and pair.premise.page == 1
    assert pair.hypothesis.source == "circular_b_revised_notice.docx" and pair.hypothesis.paragraph is not None
    assert "page 1" in pair.to_dict()["premise_location"]


def test_nli_direction_document_a_is_premise(fake_backend, circular_a, circular_b):
    compare_documents(circular_a, circular_b, CFG)
    a_texts = {s.text for s in circular_a.sentences}
    b_texts = {s.text for s in circular_b.sentences}
    assert fake_backend.nli_calls
    for premise, hypothesis in fake_backend.nli_calls:
        assert premise in a_texts and hypothesis in b_texts


def test_results_sorted_contradictions_first(fake_backend, circular_a, circular_b):
    report = compare_documents(circular_a, circular_b, CFG)
    ranks = {STATUS_CONTRADICTION: 0, STATUS_REVIEW: 1, STATUS_CONSISTENT: 2, STATUS_NEUTRAL: 3}
    order = [ranks[p.status] for p in report.pairs]
    assert order == sorted(order)


def test_max_pairs_and_threshold_are_configurable(fake_backend, circular_a, circular_b):
    small = compare_documents(circular_a, circular_b, ComparisonConfig(similarity_threshold=0.3, max_pairs=3))
    assert small.candidate_pairs == 3 and len(small.pairs) == 3
    strict = compare_documents(circular_a, circular_b, ComparisonConfig(similarity_threshold=0.999))
    assert all(p.similarity >= 0.999 for p in strict.pairs)


def test_doc_vs_doc_nothing_relevant_gives_message(fake_backend):
    a = extract_document("a.txt", b"Bananas grow in tropical climates.")
    b = extract_document("b.txt", b"The stock market closed higher today.")
    report = compare_documents(a, b, ComparisonConfig(similarity_threshold=0.9))
    assert report.pairs == [] and report.messages and fake_backend.nli_calls == []


def test_bidirectional_runs_reverse_nli(fake_backend, circular_a, circular_b):
    report = compare_documents(circular_a, circular_b,
                               ComparisonConfig(similarity_threshold=0.3, bidirectional=True))
    assert len(fake_backend.nli_calls) == 2 * report.candidate_pairs
    assert all(p.reverse is not None for p in report.pairs)


def test_report_to_dict_structure(fake_backend, circular_a, circular_b):
    d = compare_documents(circular_a, circular_b, CFG).to_dict()
    for key in ("mode", "premise_count", "candidate_pairs", "summary", "pairs", "messages", "config"):
        assert key in d
    row = d["pairs"][0]
    for key in ("similarity", "relationship", "entailment_score", "contradiction_score",
                "neutral_score", "status", "premise_location", "hypothesis_location"):
        assert key in row
    assert row["entailment_score"] + row["contradiction_score"] + row["neutral_score"] == pytest.approx(1.0)


# ---------------------------------------------------------- Text vs Document
def test_text_vs_doc_finds_attendance_evidence(fake_backend, circular_a):
    stmt = "Students with 60% attendance are eligible for examinations."
    report = compare_text_to_document(stmt, circular_a, CFG)
    assert report.mode == "Text vs Document" and report.hypothesis_count == 1
    top = report.pairs[0]
    assert "75%" in top.premise.text                       # relevant evidence retrieved
    assert top.hypothesis.text == stmt and top.hypothesis.source == "Entered statement"
    assert top.status == STATUS_REVIEW and "differing_percentage" in top.flags
    assert top.premise.page == 1 and "circular_a" in top.premise.location
    assert top.premise.context                              # paragraph kept for highlighting
    # direction: document evidence = premise, entered statement = hypothesis
    assert all(h == stmt for _, h in fake_backend.nli_calls)


def test_text_vs_doc_no_evidence_message_not_invented_verdict(fake_backend, circular_a):
    report = compare_text_to_document("Bananas grow in tropical climates.", circular_a,
                                      ComparisonConfig(similarity_threshold=0.6))
    assert report.pairs == [] and fake_backend.nli_calls == []
    assert "No relevant evidence" in report.messages[0]
    assert "No verdict" in report.messages[0]


def test_text_vs_doc_rejects_empty_statement(fake_backend, circular_a):
    for bad in ("", "   ", None):
        with pytest.raises(ValueError):
            compare_text_to_document(bad, circular_a)


# ------------------------------------------------- Phase 1 stays intact
def test_phase1_text_vs_text_still_works_and_shares_nli(fake_backend):
    r = compare_statements("The library is open on Sundays.", "The library is closed every Sunday.")
    assert set(r) == {"statement_a", "statement_b", "similarity", "relationship",
                      "entailment_score", "contradiction_score", "neutral_score"}
    assert r["relationship"] == "Contradiction"
    assert fake_backend.nli_calls == [("The library is open on Sundays.", "The library is closed every Sunday.")]


def test_predict_nli_is_wrapper_over_batch(monkeypatch):
    canned = {"relationship": "Neutral", "entailment": 0.2, "contradiction": 0.1, "neutral": 0.7}
    seen = {}
    monkeypatch.setattr(inference, "predict_nli_batch", lambda pairs, batch_size=16: seen.setdefault("p", list(pairs)) and [canned])
    assert inference.predict_nli("a", "b") == canned
    assert seen["p"] == [("a", "b")]


# -------------------------------------------- integration (real models)
@skip_models
@pytest.mark.model
def test_real_models_doc_vs_doc(circular_a, circular_b):
    report = compare_documents(circular_a, circular_b, ComparisonConfig())
    print("\nstatus summary:", report.summary)
    assert report.candidate_pairs > 0 and report.pairs
    for p in report.pairs:
        assert p.relationship in {"Entailment", "Contradiction", "Neutral"}
        assert p.entailment_score + p.contradiction_score + p.neutral_score == pytest.approx(1.0, abs=1e-4)


@skip_models
@pytest.mark.model
def test_real_models_text_vs_doc(circular_a):
    report = compare_text_to_document("Students with 60% attendance are eligible for examinations.",
                                      circular_a, ComparisonConfig())
    print("\ntop evidence:", report.pairs[0].premise.text if report.pairs else report.messages)
    assert report.pairs, "expected at least one relevant evidence sentence"
