"""Small HTML building blocks for the dashboard (pure functions: str in -> HTML str out).

All user / document text goes through `html.escape`, so uploaded sentences can never inject
markup. Colours and numbers always come from the real backend result (see utils/results.py).
HTML is emitted WITHOUT leading indentation or blank lines, otherwise Markdown would turn it
into a code block.
"""
from __future__ import annotations

from html import escape

from utils.results import STATUS_STYLE, confidence_level, uncertainty_message

E = escape


def _compact(html: str) -> str:
    """Strip line indentation / blank lines so Markdown never treats the HTML as a code block."""
    return "".join(line.strip() for line in html.strip().splitlines())


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# --------------------------------------------------------------------- hero
def hero_chips(items: list[str]) -> str:
    return _compact('<div class="cx-chips">' + "".join(f'<span class="cx-chip">{E(i)}</span>' for i in items) + "</div>")


def hero_text(tagline: str, subtitle: str) -> str:
    return _compact(f'<div class="cx-tagline">{E(tagline)}</div><p class="cx-subtitle">{E(subtitle)}</p>')


def mode_cards(cards: list[dict]) -> str:
    """cards: [{icon, title, text, tag}]"""
    inner = "".join(
        f'<div class="cx-card cx-mode-card"><div class="cx-icon">{E(c["icon"])}</div>'
        f'<div class="cx-card-title">{E(c["title"])}</div><p class="cx-card-text">{E(c["text"])}</p>'
        f'<span class="cx-tag">{E(c["tag"])}</span></div>' for c in cards)
    return _compact(f'<div class="cx-grid cx-grid-3">{inner}</div>')


def section_header(title: str, subtitle: str = "", eyebrow: str = "") -> str:
    eb = f'<div class="cx-eyebrow">{E(eyebrow)}</div>' if eyebrow else ""
    sub = f'<p class="cx-section-sub">{E(subtitle)}</p>' if subtitle else ""
    return _compact(f'<div class="cx-section">{eb}<h3 class="cx-section-title">{E(title)}</h3>{sub}</div>')


# ----------------------------------------------------------- result visuals
def classification_card(status: str, nli_label: str, scores: dict, title: str = "Predicted relationship") -> str:
    kind, _ = STATUS_STYLE[status]
    conf = confidence_level(scores["entailment"], scores["contradiction"], scores["neutral"])
    msg = uncertainty_message(conf["level"], nli_label)
    return _compact(f"""
    <div class="cx-class cx-{kind}">
      <div class="cx-class-top"><span class="cx-eyebrow">{E(title)}</span>
        <span class="cx-badge cx-conf-{conf["level"].lower()}">Model confidence: {conf["level"]} ({pct(conf["top"])})</span></div>
      <div class="cx-class-label">{E(status)}</div>
      <div class="cx-class-sub">NLI label: <b>{E(nli_label)}</b></div>
      <p class="cx-class-note">{E(msg)}</p>
    </div>""")


def prob_bar(label: str, value: float, kind: str) -> str:
    width = max(0.0, min(1.0, value)) * 100
    return _compact(f"""
    <div class="cx-bar"><div class="cx-bar-head"><span>{E(label)}</span><b>{pct(value)}</b></div>
      <div class="cx-bar-track"><div class="cx-bar-fill cx-fill-{kind}" style="width:{width:.1f}%"></div></div></div>""")


def prob_bars(entailment: float, contradiction: float, neutral: float) -> str:
    return _compact('<div class="cx-card cx-bars">'
                    + prob_bar("Contradiction probability", contradiction, "danger")
                    + prob_bar("Entailment probability", entailment, "success")
                    + prob_bar("Neutral probability", neutral, "neutral") + "</div>")


def score_strip(similarity: float, entailment: float, contradiction: float, neutral: float) -> str:
    items = [("Similarity", f"{similarity:.3f}", ""), ("Contradiction", f"{contradiction:.3f}", "danger"),
             ("Entailment", f"{entailment:.3f}", "success"), ("Neutral", f"{neutral:.3f}", "neutral")]
    inner = "".join(f'<div class="cx-strip-item cx-strip-{k or "plain"}"><span>{E(n)}</span><b>{E(v)}</b></div>'
                    for n, v, k in items)
    return _compact(f'<div class="cx-strip">{inner}</div>')


def source_paragraphs(premise_ctx: str, premise: str, hypothesis_ctx: str, hypothesis: str) -> str:
    """Collapsible source paragraphs (HTML <details>, so it also works inside an st.expander)."""
    return _compact(
        '<details class="cx-details"><summary>Show source paragraphs</summary>'
        '<div class="cx-eyebrow" style="margin:.6rem 0 .3rem">Premise paragraph</div>'
        + highlight_in_context(premise_ctx or premise, premise)
        + '<div class="cx-eyebrow" style="margin:.8rem 0 .3rem">Hypothesis paragraph</div>'
        + highlight_in_context(hypothesis_ctx or hypothesis, hypothesis) + "</details>")


def statement_compare(premise: str, premise_src: str, hypothesis: str, hypothesis_src: str,
                      premise_label: str = "Premise", hypothesis_label: str = "Hypothesis",
                      kind: str = "neutral") -> str:
    return _compact(f"""
    <div class="cx-grid cx-grid-2">
      <div class="cx-card cx-statement cx-edge-blue"><div class="cx-eyebrow">{E(premise_label)}</div>
        <p class="cx-quote">{E(premise)}</p><span class="cx-src">{E(premise_src)}</span></div>
      <div class="cx-card cx-statement cx-edge-{kind}"><div class="cx-eyebrow">{E(hypothesis_label)}</div>
        <p class="cx-quote">{E(hypothesis)}</p><span class="cx-src">{E(hypothesis_src)}</span></div>
    </div>""")


def note_box(text: str, kind: str = "info") -> str:
    return _compact(f'<div class="cx-note cx-note-{kind}">{E(text)}</div>')


def highlight_in_context(context: str, sentence: str) -> str:
    """Paragraph HTML with the evidence sentence highlighted (escaped first, then marked)."""
    ctx, sent = E(context), E(sentence)
    if sent and sent in ctx:
        return f'<div class="cx-context">{ctx.replace(sent, f"<mark>{sent}</mark>", 1)}</div>'
    return f'<div class="cx-context"><mark>{sent}</mark></div>'


def status_dot(status: str) -> str:
    """Emoji used where HTML cannot be rendered (expander titles, dataframe cells)."""
    return {"danger": "🔴", "warning": "🟠", "success": "🟢", "neutral": "🔵"}[STATUS_STYLE[status][0]]


# -------------------------------------------------------------- methodology
def step_flow(steps: list[tuple[str, str]]) -> str:
    inner = '<span class="cx-arrow">→</span>'.join(
        f'<div class="cx-step"><b>{E(t)}</b><span>{E(d)}</span></div>' for t, d in steps)
    return _compact(f'<div class="cx-flow">{inner}</div>')


def info_cards(cards: list[dict], columns: int = 3) -> str:
    """cards: [{title, text, kind?}] -> grid of small cards."""
    inner = "".join(
        f'<div class="cx-card cx-info {("cx-edge-" + c["kind"]) if c.get("kind") else ""}">'
        f'<div class="cx-card-title">{E(c["title"])}</div><p class="cx-card-text">{E(c["text"])}</p></div>'
        for c in cards)
    return _compact(f'<div class="cx-grid cx-grid-{columns}">{inner}</div>')


def tech_badges(items: list[tuple[str, str]]) -> str:
    inner = "".join(f'<span class="cx-tech"><b>{E(n)}</b><i>{E(d)}</i></span>' for n, d in items)
    return _compact(f'<div class="cx-techs">{inner}</div>')


# ------------------------------------------------------------ results table
def results_table(rows: list[dict]) -> str:
    """Readable HTML table of compared pairs (no pandas / pyarrow needed).

    rows: dicts from utils.results (keys: No, Status, Premise, Hypothesis, Similarity,
    Contradiction, Entailment, Neutral, Confidence). Text is escaped and shortened.
    """
    def short(text: str, n: int = 110) -> str:
        return text if len(text) <= n else text[: n - 1].rstrip() + "\u2026"

    def cell(v: float, kind: str) -> str:
        return (f'<td class="cx-num"><div class="cx-mini"><span class="cx-mini-fill cx-fill-{kind}" '
                f'style="width:{max(0.0, min(1.0, v)) * 100:.0f}%"></span></div><b>{v:.2f}</b></td>')

    body = []
    for r in rows:
        kind = STATUS_STYLE[r["Status"]][0]
        body.append(
            f'<tr><td class="cx-idx">{E(str(r["No"]))}</td>'
            f'<td><span class="cx-pill cx-pill-{kind}">{E(r["Status"])}</span>'
            f'<div class="cx-conf-small">confidence: {E(r["Confidence"])}</div></td>'
            f'<td>{E(short(r["Premise"]))}</td><td>{E(short(r["Hypothesis"]))}</td>'
            + cell(r["Similarity"], "blue") + cell(r["Contradiction"], "danger")
            + cell(r["Entailment"], "success") + cell(r["Neutral"], "neutral") + "</tr>")
    head = "".join(f"<th>{h}</th>" for h in
                   ("#", "Status", "Premise", "Hypothesis", "Similarity", "Contradiction", "Entailment", "Neutral"))
    return _compact(f'<div class="cx-table-wrap"><table class="cx-table"><thead><tr>{head}</tr></thead>'
                    f'<tbody>{"".join(body)}</tbody></table></div>')
