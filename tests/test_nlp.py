"""Automated tests for CONTRADIX Phase 1.

Two groups:
  * Fast unit tests - no model download (logic, validation, label mapping,
    pipeline structure with the models mocked out).
  * Integration tests (marked `model`) - load the REAL pretrained models.
    First run needs internet. Skip them with:  set CONTRADIX_SKIP_MODEL_TESTS=1

The demo pairs are demonstrations, so integration tests check structure and
sanity (valid label, scores sum to 1) rather than demanding a specific label.
"""
import os

import numpy as np
import pytest

from nlp import compare_statements, inference, similarity
from nlp.preprocessing import analyze_text, clean_text, split_sentences, validate_text
from utils.examples import DEMO_EXAMPLES

VALID_LABELS = {"Entailment", "Contradiction", "Neutral"}
RESULT_KEYS = {
    "statement_a", "statement_b", "similarity", "relationship",
    "entailment_score", "contradiction_score", "neutral_score",
}

skip_models = pytest.mark.skipif(
    os.environ.get("CONTRADIX_SKIP_MODEL_TESTS") == "1",
    reason="CONTRADIX_SKIP_MODEL_TESTS=1",
)


# ---------------------------------------------------------------- preprocessing
def test_clean_text_normalises_whitespace_and_quotes():
    assert clean_text("  Hello \n\t world \u2019s  ") == "Hello world 's"


def test_clean_text_keeps_case_and_negation():
    assert clean_text("The library is NOT open.") == "The library is NOT open."


def test_validate_text_rejects_empty_and_whitespace():
    for bad in ("", "   ", "\n\t", None):
        with pytest.raises(ValueError):
            validate_text(bad, "Statement A")


def test_validate_text_rejects_too_long():
    with pytest.raises(ValueError):
        validate_text("a" * 6000)


def test_split_sentences():
    sents = split_sentences("The exam is at 10 AM. The hall is open. Bring your ID.")
    assert len(sents) == 3


def test_analyze_text_structure_and_negation():
    info = analyze_text("The library is not open on Sundays.")
    assert info["num_sentences"] == 1
    assert info["num_tokens"] >= 7
    assert "not" in info["negations"]


# ------------------------------------------------------------------ similarity
def test_cosine_similarity_basic_cases():
    assert similarity.cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert similarity.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert similarity.cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert similarity.cosine_similarity([0, 0], [1, 1]) == 0.0


# ------------------------------------------------------------------- inference
def test_label_mapping_uses_model_config_names():
    cfg = {0: "contradiction", 1: "entailment", 2: "neutral"}
    assert inference.resolve_label_mapping(cfg) == cfg
    shuffled = {0: "ENTAILMENT", 1: "Neutral", 2: "contradiction"}
    assert inference.resolve_label_mapping(shuffled) == {0: "entailment", 1: "neutral", 2: "contradiction"}


def test_label_mapping_generic_labels_fall_back_with_warning():
    with pytest.warns(UserWarning):
        m = inference.resolve_label_mapping({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
    assert m == {0: "contradiction", 1: "entailment", 2: "neutral"}


def test_label_mapping_rejects_unknown_labels():
    with pytest.raises(ValueError):
        inference.resolve_label_mapping({0: "yes", 1: "no", 2: "maybe"})


def test_scores_from_logits_softmax_and_label():
    mapping = {0: "contradiction", 1: "entailment", 2: "neutral"}
    out = inference.scores_from_logits(np.array([5.0, 0.1, -1.0]), mapping)
    assert out["label"] == "contradiction"
    assert sum(out["scores"].values()) == pytest.approx(1.0)
    assert set(out["scores"]) == {"entailment", "contradiction", "neutral"}


def test_scores_from_logits_is_stable_for_large_values():
    mapping = {0: "contradiction", 1: "entailment", 2: "neutral"}
    out = inference.scores_from_logits(np.array([1000.0, 999.0, 0.0]), mapping)
    assert sum(out["scores"].values()) == pytest.approx(1.0)


# ---------------------------------------------------- pipeline (models mocked)
def test_compare_statements_structure_with_mocked_models(monkeypatch):
    calls = {}

    def fake_nli(premise, hypothesis):
        calls["premise"], calls["hypothesis"] = premise, hypothesis
        return {"relationship": "Contradiction", "entailment": 0.05,
                "contradiction": 0.9, "neutral": 0.05}

    monkeypatch.setattr(similarity, "compute_similarity", lambda a, b: 0.87)
    monkeypatch.setattr(inference, "predict_nli", fake_nli)

    r = compare_statements("  The library is open on Sundays. ", "The library is closed every Sunday.")
    assert set(r) == RESULT_KEYS
    assert r["similarity"] == pytest.approx(0.87)
    assert r["relationship"] == "Contradiction"
    # A is premise, B is hypothesis
    assert calls["premise"] == "The library is open on Sundays."
    assert calls["hypothesis"] == "The library is closed every Sunday."


def test_similarity_and_nli_are_independent(monkeypatch):
    """High similarity must not force an 'Entailment' label."""
    monkeypatch.setattr(similarity, "compute_similarity", lambda a, b: 0.99)
    monkeypatch.setattr(inference, "predict_nli", lambda premise, hypothesis: {
        "relationship": "Contradiction", "entailment": 0.02, "contradiction": 0.97, "neutral": 0.01})
    r = compare_statements("It is open.", "It is closed.")
    assert r["similarity"] > 0.9 and r["relationship"] == "Contradiction"


@pytest.mark.parametrize("a,b", [("", "text"), ("text", ""), ("   ", "text"), ("text", "\n"), (None, "x")])
def test_compare_statements_rejects_empty_input(a, b):
    with pytest.raises(ValueError):
        compare_statements(a, b)


# ------------------------------------------- integration tests (real models)
@skip_models
@pytest.mark.model
def test_real_nli_label_mapping_is_valid():
    mapping = inference.get_label_mapping()
    assert set(mapping.values()) == {"entailment", "contradiction", "neutral"}
    assert len(mapping) == 3


@skip_models
@pytest.mark.model
@pytest.mark.parametrize("ex", DEMO_EXAMPLES, ids=[e["expected"] for e in DEMO_EXAMPLES])
def test_real_models_on_demo_pairs(ex):
    r = compare_statements(ex["text_a"], ex["text_b"])
    print(f"\n{ex['name']}: expected={ex['expected']} predicted={r['relationship']} "
          f"sim={r['similarity']:.3f}")
    assert set(r) == RESULT_KEYS
    assert r["relationship"] in VALID_LABELS
    assert -1.0 <= r["similarity"] <= 1.0
    total = r["entailment_score"] + r["contradiction_score"] + r["neutral_score"]
    assert total == pytest.approx(1.0, abs=1e-4)


@skip_models
@pytest.mark.model
def test_real_similarity_identical_vs_unrelated():
    same = similarity.compute_similarity("The exam is at 10 AM.", "The exam is at 10 AM.")
    other = similarity.compute_similarity("The exam is at 10 AM.", "Bananas are rich in potassium.")
    assert same > 0.99
    assert other < same


@skip_models
@pytest.mark.model
def test_real_direction_swap_returns_valid_result():
    r = compare_statements("The college has a large library.", "The library is closed every Sunday.")
    assert r["relationship"] in VALID_LABELS
