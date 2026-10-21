"""Shared comparison pipeline for Document-vs-Document and Text-vs-Document.

    sentences -> embeddings -> semantic retrieval of relevant pairs
              -> NLI (the SAME cached model as Phase 1) -> classified results

Why retrieval first?  Running NLI on every sentence combination is O(n*m).
Embeddings are cheap, so we only send the most semantically relevant pairs
(above a similarity threshold, capped per sentence and overall) to the NLI model.

NLI DIRECTION (always explicit):
  * Document vs Document : sentence from Document A = PREMISE, from Document B = HYPOTHESIS
  * Text vs Document     : sentence from the document (evidence) = PREMISE,
                           the entered statement = HYPOTHESIS ("does the evidence support the claim?")
  * Optional `bidirectional` also runs hypothesis -> premise and reports it separately.

Similarity and NLI stay separate measurements. A numeric / date difference is NEVER
turned into a contradiction by itself: it is surfaced as "Needs review" (see
`find_value_discrepancies`) and the final decision is left to the human reader.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS

from nlp import inference, similarity
from nlp.document_processor import ExtractedDocument, SentenceRecord
from nlp.preprocessing import validate_text

STATUS_CONTRADICTION = "Potential contradiction"
STATUS_REVIEW = "Needs review (differing values)"
STATUS_CONSISTENT = "Consistent (entailment)"
STATUS_NEUTRAL = "Neutral / no clear relation"
_STATUS_RANK = {STATUS_CONTRADICTION: 0, STATUS_REVIEW: 1, STATUS_CONSISTENT: 2, STATUS_NEUTRAL: 3}


# ------------------------------------------------------------------- config
@dataclass
class ComparisonConfig:
    similarity_threshold: float = 0.45        # min cosine similarity for a pair to be "relevant"
    max_pairs: int = 25                       # hard cap on pairs sent to NLI
    top_k: int = 3                            # best matches kept per sentence (doc vs doc)
    contradiction_threshold: float = 0.50     # min contradiction probability to flag
    review_similarity_threshold: float = 0.0  # extra min similarity to flag differing values (0 = any retrieved pair;
                                              # the shared-subject check already guards against unrelated numbers)
    bidirectional: bool = False               # also run NLI in the reverse direction

    def validate(self) -> "ComparisonConfig":
        if not -1.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1 and 1.")
        if not 0.0 <= self.contradiction_threshold <= 1.0:
            raise ValueError("contradiction_threshold must be between 0 and 1.")
        if self.max_pairs < 1 or self.top_k < 1:
            raise ValueError("max_pairs and top_k must be at least 1.")
        return self


# ------------------------------------------------------------ result classes
@dataclass
class PairResult:
    premise: SentenceRecord
    hypothesis: SentenceRecord
    similarity: float
    relationship: str
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    status: str
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reverse: dict | None = None   # NLI result for hypothesis -> premise (if bidirectional)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["premise_location"] = self.premise.location
        d["hypothesis_location"] = self.hypothesis.location
        return d


@dataclass
class ComparisonReport:
    mode: str
    premise_label: str
    hypothesis_label: str
    premise_count: int
    hypothesis_count: int
    total_possible_pairs: int
    candidate_pairs: int
    pairs: list[PairResult] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    best_similarity: float | None = None
    config: ComparisonConfig = field(default_factory=ComparisonConfig)

    @property
    def summary(self) -> dict:
        counts = {s: 0 for s in _STATUS_RANK}
        for p in self.pairs:
            counts[p.status] += 1
        return counts

    def to_dict(self) -> dict:
        return {
            "mode": self.mode, "premise_label": self.premise_label,
            "hypothesis_label": self.hypothesis_label,
            "premise_count": self.premise_count, "hypothesis_count": self.hypothesis_count,
            "total_possible_pairs": self.total_possible_pairs,
            "candidate_pairs": self.candidate_pairs, "best_similarity": self.best_similarity,
            "summary": self.summary, "messages": list(self.messages),
            "pairs": [p.to_dict() for p in self.pairs], "config": asdict(self.config),
        }


# ------------------------------------------- value / context signal detection
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
          r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_MONTH_NUM = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MONTH_ABBR = {v: k.capitalize() for k, v in _MONTH_NUM.items()}
_DATE_MD = re.compile(rf"\b({_MONTH})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.I)
_DATE_DM = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH})\b", re.I)
_DATE_NUM = re.compile(r"\b(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})\b")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|per\s?cent\b|percent\b)", re.I)
_NUM_WORDS = {w: i for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
     "eleven", "twelve"])}
_DURATION = re.compile(
    r"\b(\d+|" + "|".join(_NUM_WORDS) + r")[\s-]+(minute|hour|day|week|month|year)s?\b", re.I)

_CONDITION_WORDS = {"if", "unless", "except", "only", "provided", "subject", "otherwise",
                    "excluding", "exception", "whereas", "when", "until"}
_GENERIC = {m for m in _MONTH_NUM} | {"january", "february", "march", "april", "june", "july",
            "august", "september", "october", "november", "december", "minute", "minutes",
            "hour", "hours", "day", "days", "week", "weeks", "month", "months", "year", "years",
            "percent", "per", "cent"}


def _month_number(token: str) -> int:
    return _MONTH_NUM[token.lower()[:3]]


def extract_values(text: str) -> dict[str, set[str]]:
    """Find percentages, dates and durations in a sentence (normalised strings)."""
    values: dict[str, set[str]] = {"percentage": set(), "date": set(), "duration": set()}
    for m in _PERCENT.finditer(text):
        num = float(m.group(1))
        values["percentage"].add(f"{int(num) if num.is_integer() else num}%")
    for m in _DATE_MD.finditer(text):
        values["date"].add(f"{_MONTH_ABBR[_month_number(m.group(1))]} {int(m.group(2))}")
    for m in _DATE_DM.finditer(text):
        values["date"].add(f"{_MONTH_ABBR[_month_number(m.group(2))]} {int(m.group(1))}")
    for m in _DATE_NUM.finditer(text):
        values["date"].add(m.group(1))
    for m in _DURATION.finditer(text):
        raw = m.group(1).lower()
        n = int(raw) if raw.isdigit() else _NUM_WORDS[raw]
        unit = m.group(2).lower()
        values["duration"].add(f"{n} {unit}{'' if n == 1 else 's'}")
    return values


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z]{3,}", text.lower())
    return {w for w in words if w not in STOP_WORDS and w not in _GENERIC}


def shared_subject_words(text_a: str, text_b: str) -> set[str]:
    """Content words both sentences share - a cheap check that they talk about the same thing."""
    return _content_words(text_a) & _content_words(text_b)


def find_value_discrepancies(text_a: str, text_b: str, min_shared_words: int = 2) -> list[dict]:
    """Differing percentages / dates / durations between two sentences about the same subject.

    Only reported when the sentences share >= `min_shared_words` content words
    (same subject) and the value sets for a category are completely different.
    This is a *signal for human review*, never a verdict.
    """
    if len(shared_subject_words(text_a, text_b)) < min_shared_words:
        return []
    va, vb = extract_values(text_a), extract_values(text_b)
    found = []
    for kind in ("percentage", "date", "duration"):
        if va[kind] and vb[kind] and va[kind].isdisjoint(vb[kind]):
            found.append({"kind": kind, "a": sorted(va[kind]), "b": sorted(vb[kind])})
    return found


def _has_condition(text: str) -> bool:
    return bool(set(re.findall(r"[a-z]+", text.lower())) & _CONDITION_WORDS)


# ------------------------------------------------------------ retrieval step
def retrieve_candidate_pairs(emb_a, emb_b, threshold: float, top_k: int, max_pairs: int,
                             both_directions: bool = True) -> list[tuple[int, int, float]]:
    """Semantic retrieval: pick the relevant (i, j, similarity) pairs instead of all n*m.

    For every row i the `top_k` most similar columns above `threshold` are kept; with
    `both_directions` the same is done for every column j. Result is sorted by
    similarity (highest first) and capped at `max_pairs`.
    """
    sims = similarity.similarity_matrix(emb_a, emb_b)
    n, m = sims.shape
    chosen: dict[tuple[int, int], float] = {}
    if n == 0 or m == 0:
        return []
    for i in range(n):
        for j in np.argsort(-sims[i])[:top_k]:
            if sims[i, j] >= threshold:
                chosen[(i, int(j))] = float(sims[i, j])
    if both_directions:
        for j in range(m):
            for i in np.argsort(-sims[:, j])[:top_k]:
                if sims[i, j] >= threshold:
                    chosen[(int(i), j)] = float(sims[i, j])
    ranked = sorted(chosen.items(), key=lambda kv: kv[1], reverse=True)[:max_pairs]
    return [(i, j, s) for (i, j), s in ranked]


# ------------------------------------------------------------- NLI + verdict
def _classify(pairs: list[tuple[SentenceRecord, SentenceRecord, float]],
              config: ComparisonConfig) -> list[PairResult]:
    """Run the shared NLI model on the retrieved pairs and build PairResults."""
    if not pairs:
        return []
    forward = inference.predict_nli_batch([(p.text, h.text) for p, h, _ in pairs])
    backward = (inference.predict_nli_batch([(h.text, p.text) for p, h, _ in pairs])
                if config.bidirectional else [None] * len(pairs))

    results = []
    for (prem, hyp, sim), fwd, rev in zip(pairs, forward, backward):
        flags, notes = [], []
        con = fwd["contradiction"]
        rev_con = rev["contradiction"] if rev else 0.0
        contradicted = con >= config.contradiction_threshold
        if not contradicted and rev and rev_con >= config.contradiction_threshold:
            contradicted = True
            notes.append("Contradiction was detected in the reverse direction (hypothesis -> premise).")

        discrepancies = []
        if sim >= config.review_similarity_threshold:
            discrepancies = find_value_discrepancies(prem.text, hyp.text)
        for d in discrepancies:
            flags.append(f"differing_{d['kind']}")
            label = {"percentage": "percentages", "date": "dates", "duration": "time periods"}[d["kind"]]
            notes.append(f"Different {label}: {', '.join(d['a'])} vs {', '.join(d['b'])}.")
            if d["kind"] == "date":
                notes.append("Dates differ. CONTRADIX flags this as a possible inconsistency only; it cannot "
                             "tell which document is valid, current or authoritative.")
        if _has_condition(prem.text) != _has_condition(hyp.text):
            flags.append("condition_or_scope")
            notes.append("Only one sentence contains a condition/scope word (e.g. 'if', 'unless', 'only'); "
                         "the apparent conflict may depend on that condition.")

        if contradicted:
            status = STATUS_CONTRADICTION
        elif fwd["relationship"] == "Entailment":
            status = STATUS_CONSISTENT
        elif discrepancies:
            status = STATUS_REVIEW
            notes.append("The NLI model did not label this a contradiction; the differing values may be an "
                         "update, a different scope, or a real conflict - please review the context.")
        else:
            status = STATUS_NEUTRAL

        results.append(PairResult(
            premise=prem, hypothesis=hyp, similarity=float(sim),
            relationship=fwd["relationship"], entailment_score=fwd["entailment"],
            contradiction_score=fwd["contradiction"], neutral_score=fwd["neutral"],
            status=status, flags=flags, notes=notes, reverse=rev))

    results.sort(key=lambda r: (_STATUS_RANK[r.status], -r.contradiction_score, -r.similarity))
    return results


# ----------------------------------------------------------------- public API
def compare_documents(doc_a: ExtractedDocument, doc_b: ExtractedDocument,
                      config: ComparisonConfig | None = None) -> ComparisonReport:
    """Document vs Document: Document A sentences = premises, Document B sentences = hypotheses."""
    cfg = (config or ComparisonConfig()).validate()
    sents_a, sents_b = doc_a.sentences, doc_b.sentences
    if not sents_a or not sents_b:
        raise ValueError("Both documents must contain at least one sentence.")

    emb_a = similarity.encode_texts([s.text for s in sents_a])
    emb_b = similarity.encode_texts([s.text for s in sents_b])
    sims = similarity.similarity_matrix(emb_a, emb_b)
    cands = retrieve_candidate_pairs(emb_a, emb_b, cfg.similarity_threshold, cfg.top_k, cfg.max_pairs)

    report = ComparisonReport(
        mode="Document vs Document",
        premise_label=f"Document A: {doc_a.name} (premise)",
        hypothesis_label=f"Document B: {doc_b.name} (hypothesis)",
        premise_count=len(sents_a), hypothesis_count=len(sents_b),
        total_possible_pairs=len(sents_a) * len(sents_b), candidate_pairs=len(cands),
        best_similarity=float(sims.max()) if sims.size else None, config=cfg)
    for d in (doc_a, doc_b):
        if d.truncated:
            report.messages.append(f"'{d.name}' is very long; only the first {len(d.sentences)} sentences were used.")
    if not cands:
        report.messages.append(
            "No sentence pairs were similar enough to compare "
            f"(threshold {cfg.similarity_threshold:.2f}). Try lowering the similarity threshold.")
        return report
    report.pairs = _classify([(sents_a[i], sents_b[j], s) for i, j, s in cands], cfg)
    return report


def compare_text_to_document(statement: str, doc: ExtractedDocument,
                             config: ComparisonConfig | None = None) -> ComparisonReport:
    """Text vs Document: relevant document sentences (premise) are tested against the statement (hypothesis)."""
    cfg = (config or ComparisonConfig()).validate()
    statement = validate_text(statement, "Statement")
    if not doc.sentences:
        raise ValueError("The document contains no sentences.")

    emb_s = similarity.encode_texts([statement])
    emb_d = similarity.encode_texts([s.text for s in doc.sentences])
    sims = similarity.similarity_matrix(emb_s, emb_d)
    cands = retrieve_candidate_pairs(emb_s, emb_d, cfg.similarity_threshold,
                                     top_k=cfg.max_pairs, max_pairs=cfg.max_pairs, both_directions=False)

    stmt_record = SentenceRecord(text=statement, source="Entered statement", index=0, context=statement)
    report = ComparisonReport(
        mode="Text vs Document",
        premise_label=f"Evidence from {doc.name} (premise)",
        hypothesis_label="Entered statement (hypothesis)",
        premise_count=len(doc.sentences), hypothesis_count=1,
        total_possible_pairs=len(doc.sentences), candidate_pairs=len(cands),
        best_similarity=float(sims.max()) if sims.size else None, config=cfg)
    if doc.truncated:
        report.messages.append(f"'{doc.name}' is very long; only the first {len(doc.sentences)} sentences were used.")
    if not cands:
        best = f" (highest similarity found: {report.best_similarity:.2f})" if report.best_similarity is not None else ""
        report.messages.append(
            f"No relevant evidence was found in '{doc.name}' for this statement{best}; "
            f"nothing reached the similarity threshold of {cfg.similarity_threshold:.2f}. "
            "No verdict was produced.")
        return report
    report.pairs = _classify([(doc.sentences[j], stmt_record, s) for _, j, s in cands], cfg)
    return report
