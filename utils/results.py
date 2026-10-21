"""Result formatting shared by the Streamlit dashboard, the CSV export and the PDF report.

Nothing in this module runs a model. It only RESHAPES what the NLP backend returned
(`compare_statements` dict, or a `ComparisonReport`) into one flat, uniform structure:

    AnalysisResult
        .rows      -> list of flat dicts (one per compared sentence pair) - used by the table, CSV and PDF
        .summary   -> counts per status
        .messages  -> backend warnings (e.g. "no relevant pairs found")

Every number shown to the user comes straight from the backend result; nothing is invented here.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime

from nlp.comparison import (
    STATUS_CONSISTENT,
    STATUS_CONTRADICTION,
    STATUS_NEUTRAL,
    STATUS_REVIEW,
    ComparisonReport,
    find_value_discrepancies,
)

MODE_TEXT = "Text vs Text"
MODE_DOC = "Document vs Document"
MODE_TEXT_DOC = "Text vs Document"

# status -> (visual kind, hex colour). "kind" is also the CSS class suffix used by the dashboard.
STATUS_STYLE = {
    STATUS_CONTRADICTION: ("danger", "#F43F5E"),
    STATUS_REVIEW: ("warning", "#F59E0B"),
    STATUS_CONSISTENT: ("success", "#10B981"),
    STATUS_NEUTRAL: ("neutral", "#64748B"),
}
RELATIONSHIP_STATUS = {
    "Contradiction": STATUS_CONTRADICTION,
    "Entailment": STATUS_CONSISTENT,
    "Neutral": STATUS_NEUTRAL,
}

# Columns of the exported CSV, in order.
CSV_COLUMNS = [
    "No", "Mode", "Status", "NLI label", "Confidence", "Premise", "Premise source",
    "Hypothesis", "Hypothesis source", "Similarity", "Entailment", "Contradiction", "Neutral",
    "Reverse NLI label", "Reverse entailment", "Reverse contradiction", "Reverse neutral",
    "Flags", "Notes",
]


# ----------------------------------------------------------------- confidence
def confidence_level(entailment: float, contradiction: float, neutral: float) -> dict:
    """How decisive is the NLI probability distribution?

    * High     : top class >= 0.80 and it leads the runner-up by >= 0.30
    * Moderate : top class >= 0.55
    * Low      : otherwise - the model is torn between classes

    This describes the *model's* certainty only. A high value is still not proof that the
    statement pair truly contradicts / entails; small NLI models can be confidently wrong.
    """
    probs = sorted([entailment, contradiction, neutral], reverse=True)
    top, margin = probs[0], probs[0] - probs[1]
    if top >= 0.80 and margin >= 0.30:
        level = "High"
    elif top >= 0.55:
        level = "Moderate"
    else:
        level = "Low"
    return {"level": level, "top": top, "margin": margin}


def uncertainty_message(level: str, relationship: str) -> str:
    """Plain-language caveat shown next to every prediction."""
    if level == "Low":
        return ("The model is not confident here: its probabilities are spread across several classes, "
                "so treat this prediction as unreliable and check the statements yourself.")
    if level == "Moderate":
        return (f"The model leans towards '{relationship}' but not decisively. "
                "Read both statements before relying on it.")
    return ("The model is fairly confident, but this is still a statistical prediction, not a verified fact. "
            "Context, conditions and dates can change the real meaning.")


# --------------------------------------------------------------- data classes
@dataclass
class AnalysisResult:
    mode: str
    generated_at: datetime
    rows: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    # How the result was produced (for the PDF header)
    settings: dict = field(default_factory=dict)
    # Counters shown in the document summary (None for Text vs Text)
    stats: dict = field(default_factory=dict)
    # Source names, e.g. ["Document A: x.pdf", "Document B: y.docx"]
    sources: list[str] = field(default_factory=list)

    @property
    def is_pair(self) -> bool:
        return self.mode == MODE_TEXT


def _fmt(x) -> float:
    return round(float(x), 4)


def _row(no, mode, status, label, scores, premise, premise_src, hypothesis, hyp_src,
         similarity, reverse=None, flags=None, notes=None) -> dict:
    conf = confidence_level(scores["entailment"], scores["contradiction"], scores["neutral"])
    row = {
        "No": no, "Mode": mode, "Status": status, "NLI label": label,
        "Confidence": conf["level"], "Premise": premise, "Premise source": premise_src,
        "Hypothesis": hypothesis, "Hypothesis source": hyp_src,
        "Similarity": _fmt(similarity),
        "Entailment": _fmt(scores["entailment"]), "Contradiction": _fmt(scores["contradiction"]),
        "Neutral": _fmt(scores["neutral"]),
        "Reverse NLI label": "", "Reverse entailment": "", "Reverse contradiction": "", "Reverse neutral": "",
        "Flags": "; ".join(flags or []), "Notes": " ".join(notes or []),
    }
    if reverse:
        row.update({
            "Reverse NLI label": reverse["relationship"],
            "Reverse entailment": _fmt(reverse["entailment"]),
            "Reverse contradiction": _fmt(reverse["contradiction"]),
            "Reverse neutral": _fmt(reverse["neutral"]),
        })
    return row


# ------------------------------------------------------------------ builders
def text_status(result: dict, discrepancies: list[dict]) -> str:
    """Display status for Text vs Text (same vocabulary as the document modes)."""
    status = RELATIONSHIP_STATUS[result["relationship"]]
    if status == STATUS_NEUTRAL and discrepancies:
        return STATUS_REVIEW
    return status


def discrepancy_notes(discrepancies: list[dict]) -> list[str]:
    names = {"percentage": "percentages", "date": "dates", "duration": "time periods"}
    notes = []
    for d in discrepancies:
        notes.append(f"Different {names[d['kind']]}: {', '.join(d['a'])} vs {', '.join(d['b'])}. "
                     "Differing values are a prompt for human review, not proof of a contradiction.")
    return notes


def build_text_analysis(result: dict, generated_at: datetime | None = None) -> AnalysisResult:
    """Wrap the dict returned by `compare_statements` (Text vs Text)."""
    discrepancies = find_value_discrepancies(result["statement_a"], result["statement_b"])
    status = text_status(result, discrepancies)
    notes = discrepancy_notes(discrepancies)
    flags = [f"differing_{d['kind']}" for d in discrepancies]
    scores = {"entailment": result["entailment_score"], "contradiction": result["contradiction_score"],
              "neutral": result["neutral_score"]}
    row = _row(1, MODE_TEXT, status, result["relationship"], scores,
               result["statement_a"], "Statement A", result["statement_b"], "Statement B",
               result["similarity"], flags=flags, notes=notes)
    return AnalysisResult(
        mode=MODE_TEXT, generated_at=generated_at or datetime.now(), rows=[row],
        summary={s: int(s == status) for s in STATUS_STYLE},
        settings={"NLI direction": "Statement A (premise) -> Statement B (hypothesis)"},
        sources=["Statement A (premise)", "Statement B (hypothesis)"])


def build_report_analysis(report: ComparisonReport, generated_at: datetime | None = None) -> AnalysisResult:
    """Wrap a `ComparisonReport` (Document vs Document / Text vs Document)."""
    rows = []
    for n, p in enumerate(report.pairs, start=1):
        scores = {"entailment": p.entailment_score, "contradiction": p.contradiction_score,
                  "neutral": p.neutral_score}
        rows.append(_row(n, report.mode, p.status, p.relationship, scores,
                         p.premise.text, p.premise.location, p.hypothesis.text, p.hypothesis.location,
                         p.similarity, reverse=p.reverse, flags=p.flags, notes=p.notes))
    cfg = report.config
    return AnalysisResult(
        mode=report.mode, generated_at=generated_at or datetime.now(), rows=rows,
        summary=report.summary, messages=list(report.messages),
        settings={
            "Similarity threshold": f"{cfg.similarity_threshold:.2f}",
            "Maximum sentence pairs": str(cfg.max_pairs),
            "Matches kept per sentence": str(cfg.top_k),
            "Contradiction threshold": f"{cfg.contradiction_threshold:.2f}",
            "Reverse NLI direction": "on" if cfg.bidirectional else "off",
        },
        stats={
            "premise_label": report.premise_label, "hypothesis_label": report.hypothesis_label,
            "premise_count": report.premise_count, "hypothesis_count": report.hypothesis_count,
            "total_possible_pairs": report.total_possible_pairs,
            "candidate_pairs": report.candidate_pairs, "best_similarity": report.best_similarity,
        },
        sources=[report.premise_label, report.hypothesis_label])


def build_analysis(payload, generated_at: datetime | None = None) -> AnalysisResult:
    """Dispatch on the backend return type."""
    if isinstance(payload, ComparisonReport):
        return build_report_analysis(payload, generated_at)
    if isinstance(payload, dict) and "relationship" in payload:
        return build_text_analysis(payload, generated_at)
    raise TypeError("Unsupported analysis payload.")


# -------------------------------------------------------------------- CSV
def _csv_safe(value):
    """Neutralise spreadsheet formula injection: sentences from uploaded files may start with = + - @."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def to_csv_bytes(analysis: AnalysisResult) -> bytes:
    """UTF-8 (with BOM so Excel opens it correctly) CSV of every compared pair."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in analysis.rows:
        writer.writerow({k: _csv_safe(v) for k, v in row.items()})
    return ("﻿" + buf.getvalue()).encode("utf-8")


def export_filename(analysis: AnalysisResult, extension: str) -> str:
    slug = analysis.mode.lower().replace(" ", "_")
    return f"contradix_{slug}_{analysis.generated_at:%Y%m%d_%H%M%S}.{extension}"


# --------------------------------------------------- table for the dashboard
def table_rows(analysis: AnalysisResult) -> list[dict]:
    """Compact, readable rows for `st.dataframe` (full text stays in the expanders / exports)."""
    def short(text: str, n: int = 90) -> str:
        return text if len(text) <= n else text[: n - 1].rstrip() + "…"

    return [{
        "#": r["No"], "Status": r["Status"], "Confidence": r["Confidence"],
        "Premise": short(r["Premise"]), "Hypothesis": short(r["Hypothesis"]),
        "Similarity": r["Similarity"], "Contradiction": r["Contradiction"],
        "Entailment": r["Entailment"], "Neutral": r["Neutral"],
    } for r in analysis.rows]
