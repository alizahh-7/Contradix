"""Core CONTRADIX function: combines preprocessing, similarity and NLI."""
from __future__ import annotations

from nlp import inference, similarity
from nlp.preprocessing import validate_text


def compare_statements(text_a: str, text_b: str) -> dict:
    """Compare two statements.

    Statement A is the PREMISE and Statement B is the HYPOTHESIS (NLI is directional).
    Semantic similarity and NLI are independent measurements: a high similarity
    does NOT imply entailment.

    Returns a dict with keys:
        statement_a, statement_b          (cleaned input text)
        similarity                        (cosine similarity, -1..1)
        relationship                      ('Entailment' | 'Contradiction' | 'Neutral')
        entailment_score, contradiction_score, neutral_score   (sum to 1)

    Raises ValueError if either input is empty / invalid.
    """
    a = validate_text(text_a, "Statement A")
    b = validate_text(text_b, "Statement B")

    sim = similarity.compute_similarity(a, b)
    nli = inference.predict_nli(premise=a, hypothesis=b)

    return {
        "statement_a": a,
        "statement_b": b,
        "similarity": float(sim),
        "relationship": nli["relationship"],
        "entailment_score": float(nli["entailment"]),
        "contradiction_score": float(nli["contradiction"]),
        "neutral_score": float(nli["neutral"]),
    }
