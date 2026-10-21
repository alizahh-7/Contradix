"""Text preprocessing and basic linguistic analysis using spaCy.

Design notes
------------
* We do NOT lowercase or strip punctuation: transformer models are case- and
  punctuation-aware, and removing words like "not" would destroy meaning.
* spaCy's `en_core_web_sm` is used when installed. If it is missing we fall
  back to a blank English pipeline with a rule-based sentence splitter so the
  app still works (POS tags / lemmas / entities are then unavailable).
"""
from __future__ import annotations

import re
import unicodedata
import warnings
from functools import lru_cache

import spacy

SPACY_MODEL = "en_core_web_sm"
MAX_CHARS = 5000

# Words/clitics that commonly signal negation (useful hint for contradictions).
NEGATION_WORDS = {
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor",
    "nowhere", "cannot", "n't", "without", "deny", "denied",
}

_QUOTE_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-",
}


@lru_cache(maxsize=1)
def get_nlp() -> "spacy.language.Language":
    """Load (once) and return the spaCy pipeline."""
    try:
        return spacy.load(SPACY_MODEL)
    except OSError:
        warnings.warn(
            f"spaCy model '{SPACY_MODEL}' not found. Using a basic fallback "
            f"pipeline. Install it with: python -m spacy download {SPACY_MODEL}",
            stacklevel=2,
        )
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp


def clean_text(text: str) -> str:
    """Normalise unicode, curly quotes and whitespace. Keeps case and punctuation."""
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _QUOTE_MAP.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def validate_text(text: str, label: str = "Text") -> str:
    """Clean `text` and make sure it is usable. Returns the cleaned text.

    Raises ValueError for empty or over-long input.
    """
    if text is None or not isinstance(text, str):
        raise ValueError(f"{label} must be a non-empty string.")
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError(f"{label} is empty. Please enter some text.")
    if len(cleaned) > MAX_CHARS:
        raise ValueError(
            f"{label} is too long ({len(cleaned)} characters). "
            f"Maximum allowed is {MAX_CHARS}."
        )
    return cleaned


def split_sentences(text: str) -> list[str]:
    """Sentence segmentation with spaCy."""
    doc = get_nlp()(clean_text(text))
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def split_sentences_batch(texts: list[str]) -> list[list[str]]:
    """Sentence-split many paragraphs efficiently (spaCy `nlp.pipe`)."""
    cleaned = [clean_text(t) for t in texts]
    out = []
    for doc in get_nlp().pipe(cleaned):
        out.append([s.text.strip() for s in doc.sents if s.text.strip()])
    return out


def analyze_text(text: str) -> dict:
    """Basic linguistic analysis of a statement."""
    cleaned = clean_text(text)
    nlp = get_nlp()
    doc = nlp(cleaned)
    tokens = [t for t in doc if not t.is_space]
    has_lemmas = nlp.has_pipe("lemmatizer") or nlp.has_pipe("tagger")
    return {
        "text": cleaned,
        "num_characters": len(cleaned),
        "num_tokens": len(tokens),
        "num_sentences": len(list(doc.sents)),
        "sentences": [s.text.strip() for s in doc.sents],
        "keywords": [
            (t.lemma_.lower() if has_lemmas and t.lemma_ else t.lower_)
            for t in tokens
            if t.is_alpha and not t.is_stop
        ],
        "entities": [(e.text, e.label_) for e in doc.ents],
        "negations": [t.text for t in tokens if t.lower_ in NEGATION_WORDS],
    }
