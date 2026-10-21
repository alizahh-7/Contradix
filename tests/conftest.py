"""Shared test fixtures: lightweight FAKE embedding + NLI backends.

They replace only the two heavy model calls (`similarity.encode_texts` /
`compute_similarity` and `inference.predict_nli_batch` / `predict_nli`). All the real
Contradix logic (extraction, segmentation, retrieval, verdict rules, UI) still runs.
"""
import re

import numpy as np
import pytest

from nlp import inference, similarity

NEGATIONS = {"no", "not", "never", "closed"}
DIM = 512


def _tokens(text):
    return re.findall(r"[a-z0-9%]+", text.lower())


def fake_embed(text):
    vec = np.zeros(DIM)
    for tok in _tokens(text):
        vec[hash(tok) % DIM] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def fake_nli_pair(premise, hypothesis):
    """Deterministic toy NLI: identical -> entailment; negation mismatch -> contradiction."""
    p, h = _tokens(premise), _tokens(hypothesis)
    if p == h:
        return {"relationship": "Entailment", "entailment": 0.9, "contradiction": 0.05, "neutral": 0.05}
    shared = set(p) & set(h)
    if (set(p) & NEGATIONS) != (set(h) & NEGATIONS) and len(shared) >= 2:
        return {"relationship": "Contradiction", "entailment": 0.05, "contradiction": 0.85, "neutral": 0.10}
    return {"relationship": "Neutral", "entailment": 0.1, "contradiction": 0.1, "neutral": 0.8}


class FakeBackend:
    def __init__(self):
        self.nli_calls = []      # list of (premise, hypothesis) in call order

    def predict_batch(self, pairs, batch_size=16):
        pairs = list(pairs)
        self.nli_calls.extend(pairs)
        return [fake_nli_pair(p, h) for p, h in pairs]


@pytest.fixture
def fake_backend(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(similarity, "encode_texts", lambda texts, batch_size=32: np.array([fake_embed(t) for t in texts]).reshape(-1, DIM))
    monkeypatch.setattr(similarity, "compute_similarity",
                        lambda a, b: float(np.dot(fake_embed(a), fake_embed(b))))
    monkeypatch.setattr(inference, "predict_nli_batch", backend.predict_batch)
    monkeypatch.setattr(inference, "predict_nli", lambda premise, hypothesis: backend.predict_batch([(premise, hypothesis)])[0])
    return backend
