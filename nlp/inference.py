"""Natural Language Inference (NLI) with a lightweight pretrained cross-encoder.

Model: cross-encoder/nli-deberta-v3-small  (CPU friendly)

DIRECTION MATTERS: NLI asks "given the PREMISE, is the HYPOTHESIS true?"
In CONTRADIX, Statement A = premise and Statement B = hypothesis.
compare(A, B) can differ from compare(B, A).

LABEL MAPPING: we never hard-code the logit order. The mapping is read from the
model's own config (`config.id2label`) at load time. The published order for this
model is {0: contradiction, 1: entailment, 2: neutral}; it is used only as a
fallback if the config contains generic names like LABEL_0.
"""
from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
CANONICAL_LABELS = ("entailment", "contradiction", "neutral")
DOCUMENTED_FALLBACK = {0: "contradiction", 1: "entailment", 2: "neutral"}


def resolve_label_mapping(id2label: dict) -> dict[int, str]:
    """Turn a model's id2label into {index: 'entailment'|'contradiction'|'neutral'}."""
    normalized = {int(i): str(lbl).strip().lower() for i, lbl in id2label.items()}
    if set(normalized.values()) == set(CANONICAL_LABELS) and len(normalized) == 3:
        return normalized
    if len(normalized) == 3 and all(v.startswith("label_") for v in normalized.values()):
        warnings.warn(
            "Model config has generic labels; using documented order "
            "[contradiction, entailment, neutral].",
            stacklevel=2,
        )
        return dict(DOCUMENTED_FALLBACK)
    raise ValueError(f"Unexpected NLI labels in model config: {id2label}")


def scores_from_logits(logits, mapping: dict[int, str]) -> dict:
    """Softmax the logits and return {'scores': {...}, 'label': best_label}."""
    x = np.asarray(logits, dtype=np.float64).ravel()
    x = x - x.max()  # numerical stability
    probs = np.exp(x) / np.exp(x).sum()
    scores = {mapping[i]: float(probs[i]) for i in range(len(probs))}
    label = max(scores, key=scores.get)
    return {"scores": scores, "label": label}


@lru_cache(maxsize=1)
def _load():
    import torch  # lazy: heavy imports
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    mapping = resolve_label_mapping(model.config.id2label)
    return tokenizer, model, mapping, torch


def get_label_mapping() -> dict[int, str]:
    """Return the verified {logit_index: label} mapping of the loaded model."""
    return dict(_load()[2])


def _result_from_logits(logits, mapping) -> dict:
    out = scores_from_logits(logits, mapping)
    return {
        "relationship": out["label"].capitalize(),
        "entailment": out["scores"]["entailment"],
        "contradiction": out["scores"]["contradiction"],
        "neutral": out["scores"]["neutral"],
    }


def predict_nli_batch(pairs, batch_size: int = 16) -> list[dict]:
    """Classify many (premise, hypothesis) pairs with the SAME cached model.

    `pairs` is a list of (premise, hypothesis) tuples - the FIRST item is always
    the premise and the SECOND the hypothesis. Returns one result dict per pair
    (same structure as `predict_nli`), in the same order.
    """
    pairs = list(pairs)
    if not pairs:
        return []
    tokenizer, model, mapping, torch = _load()
    results: list[dict] = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        inputs = tokenizer(
            [p for p, _ in chunk], [h for _, h in chunk],
            return_tensors="pt", padding=True, truncation=True, max_length=512,
        )
        with torch.no_grad():
            logits = model(**inputs).logits.cpu().numpy()
        results.extend(_result_from_logits(row, mapping) for row in logits)
    return results


def predict_nli(premise: str, hypothesis: str) -> dict:
    """Classify (premise -> hypothesis).

    Returns {'relationship': 'Entailment'|'Contradiction'|'Neutral',
             'entailment': float, 'contradiction': float, 'neutral': float}
    The three scores are softmax probabilities and sum to 1.
    Phase 2: this is a thin wrapper over `predict_nli_batch`, so every comparison
    mode runs through exactly one inference implementation.
    """
    return predict_nli_batch([(premise, hypothesis)])[0]
