"""Semantic similarity using Sentence Transformers (all-MiniLM-L6-v2).

Similarity answers "are these about the same thing?" - NOT "do they agree?".
"The library is open" and "The library is closed" are highly similar yet
contradictory. That is why similarity and NLI are kept as separate measurements.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load (once) the sentence-embedding model on CPU."""
    from sentence_transformers import SentenceTransformer  # lazy: heavy import

    return SentenceTransformer(MODEL_NAME, device="cpu")


def cosine_similarity(vec_a, vec_b) -> float:
    """Cosine similarity of two vectors, clipped to [-1, 1]."""
    a = np.asarray(vec_a, dtype=np.float64).ravel()
    b = np.asarray(vec_b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def compute_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between the sentence embeddings of two texts."""
    model = get_embedding_model()
    emb = model.encode([text_a, text_b], convert_to_numpy=True, show_progress_bar=False)
    return cosine_similarity(emb[0], emb[1])


def encode_texts(texts, batch_size: int = 32) -> np.ndarray:
    """Embed many texts at once. Returns an (n, dim) array of L2-normalised vectors."""
    texts = list(texts)
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_embedding_model()
    return model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )


def similarity_matrix(emb_a, emb_b) -> np.ndarray:
    """Cosine similarity of every row of emb_a against every row of emb_b -> (n, m)."""
    a = np.asarray(emb_a, dtype=np.float64)
    b = np.asarray(emb_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)))
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return np.clip(a @ b.T, -1.0, 1.0)
