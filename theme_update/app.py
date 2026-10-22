"""CONTRADIX - final Streamlit dashboard (Phase 3).

Semantic Contradiction Detection and Document Consistency Analysis Using NLP.
Modes: Text vs Text | Document vs Document | Text vs Document

Run with:  python -m streamlit run app.py

Architecture (unchanged NLP engine, new presentation layer):
    nlp/*            backend: preprocessing, MiniLM similarity, DeBERTa NLI, retrieval, verdict rules
    utils/results.py reshapes backend output -> rows (table / CSV / PDF)
    utils/report.py  PDF report
    utils/components.py + utils/styles.py   HTML / CSS for the dashboard
    app.py           layout, session state, wiring
"""
from __future__ import annotations

import hashlib
from dataclasses import astuple

import streamlit as st

from nlp import (
    ComparisonConfig,
    compare_documents,
    compare_statements,
    compare_text_to_document,
    extract_document,
    extract_document_from_path,
    inference,
    similarity,
)
from nlp.comparison import STATUS_CONSISTENT, STATUS_CONTRADICTION, STATUS_NEUTRAL, STATUS_REVIEW
from nlp.document_processor import DocumentError
from nlp.preprocessing import analyze_text
from utils import components as ui
from utils.examples import DEMO_EXAMPLES, EXAMPLE_BY_NAME
from utils.file_handler import SAMPLE_CIRCULAR_A, SAMPLE_CIRCULAR_B, UPLOAD_TYPES
from utils.report import build_pdf_report
from utils.results import (
    MODE_DOC,
    MODE_TEXT,
    MODE_TEXT_DOC,
    RELATIONSHIP_STATUS,
    STATUS_STYLE,
    build_analysis,
    export_filename,
    to_csv_bytes,
)
from utils.styles import CSS

st.set_page_config(page_title="CONTRADIX | Beyond Words. Detect the Contradiction.", page_icon="🔍",
                   layout="wide", initial_sidebar_state="collapsed")

MODES = [MODE_TEXT, MODE_DOC, MODE_TEXT_DOC]
DEFAULTS = ComparisonConfig()
EXAMPLE_TEXT_VS_DOC = "Students with 60% attendance are eligible for examinations."
LOADING_HINT = "Running the NLP models... the very first run downloads them (a few hundred MB), so please be patient."


# =============================================================== session state
def init_state() -> None:
    st.session_state.setdefault("results", {})      # mode -> bundle (last result for each mode)
    st.session_state.setdefault("_run_cache", {})   # input fingerprint -> raw backend payload
    st.session_state.setdefault("show_workspace", False)  # landing page first


# ------------------------------------------------------------------ callbacks
def load_example(name: str) -> None:
    """Demo playground: copy the chosen example into the two text boxes (no inference here)."""
    ex = EXAMPLE_BY_NAME[name]
    st.session_state["text_a"] = ex["text_a"]
    st.session_state["text_b"] = ex["text_b"]
    st.session_state["active_example"] = name


def load_sample_documents() -> None:
    st.session_state["dd_a_sample"] = True
    st.session_state["dd_b_sample"] = True


def load_text_doc_demo() -> None:
    st.session_state["tvd_statement"] = EXAMPLE_TEXT_VS_DOC
    st.session_state["td_sample"] = True


# ------------------------------------------------------------- cached helpers
@st.cache_data(show_spinner=False, max_entries=24)
def cached_extract_upload(name: str, data: bytes):
    """Extraction runs once per distinct uploaded file, not on every Streamlit rerun."""
    return extract_document(name, data)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_extract_sample(path: str):
    return extract_document_from_path(path)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_linguistic_info(text: str) -> dict:
    return analyze_text(text)


def doc_fingerprint(doc) -> str:
    return f"{doc.name}:{len(doc.sentences)}:{hashlib.sha1(doc.full_text.encode('utf-8')).hexdigest()[:16]}"


def model_status() -> tuple[bool, bool]:
    """Real status of the two lru_cache'd models (no loading is triggered)."""
    return (similarity.get_embedding_model.cache_info().currsize > 0,
            inference._load.cache_info().currsize > 0)


# =========================================== fixed internal retrieval settings
def sidebar_settings() -> ComparisonConfig:
    """Keep retrieval active without exposing configuration controls in the sidebar."""
    return ComparisonConfig(
        similarity_threshold=0.43,
        max_pairs=25,
        top_k=2,
        contradiction_threshold=0.29,
        bidirectional=True,
    )


# ================================================================ hero / home
def render_hero() -> None:
    st.markdown(ui.hero_chips(["NLP", "Semantic similarity", "Natural language inference"]),
                unsafe_allow_html=True)
    st.title("CONTRADIX")
    st.markdown(ui.hero_text(
        "Beyond Words. Detect the Contradiction.",
        "An NLP-powered semantic analysis platform for identifying potentially conflicting information "
        "across statements and documents."), unsafe_allow_html=True)
    st.markdown(ui.mode_cards([
        {"icon": "⚖️", "title": MODE_TEXT, "tag": "Premise → Hypothesis",
         "text": "Compare two statements and see whether the second follows from, conflicts with, or is unrelated to the first."},
        {"icon": "📑", "title": MODE_DOC, "tag": "PDF · DOCX · TXT",
         "text": "Upload two documents. Relevant sentence pairs are retrieved and checked for potential conflicts, with page and paragraph references."},
        {"icon": "🔎", "title": MODE_TEXT_DOC, "tag": "Evidence retrieval",
         "text": "Test a claim against a reference document and inspect the evidence sentences behind every result."},
    ]), unsafe_allow_html=True)


# ============================================================ run + persistence
def run_analysis(mode: str, key: tuple, compute) -> None:
    """Run the backend (or reuse an identical earlier run) and store the result for this mode."""
    results = st.session_state["results"]
    existing = results.get(mode)
    if existing is not None and existing["key"] == key:
        st.toast("Same inputs and settings as the current result - nothing to recompute.", icon="ℹ️")
        return

    cache = st.session_state["_run_cache"]
    if key in cache:
        payload = cache[key]
    else:
        payload, user_error, crash = None, None, None
        with st.status("Analyzing... running the NLP models", expanded=True) as status:
            st.write(LOADING_HINT)
            try:
                payload = compute()
                status.update(label="Analysis complete", state="complete", expanded=False)
            except (DocumentError, ValueError) as exc:
                user_error = str(exc)
                status.update(label="Analysis stopped", state="error", expanded=False)
            except Exception as exc:  # model download / install problems etc.
                crash = exc
                status.update(label="Analysis failed", state="error", expanded=False)
        # errors are shown OUTSIDE the (collapsed) status box so they are always visible
        if user_error:
            st.error(user_error)
        if crash is not None:
            st.error("Something went wrong while running the NLP models. Check your internet connection "
                     "(needed on first run) and that all requirements are installed.")
            st.exception(crash)
        if payload is None:
            return
        if len(cache) >= 12:                       # keep memory bounded
            cache.pop(next(iter(cache)))
        cache[key] = payload

    results[mode] = {"key": key, "payload": payload, "analysis": build_analysis(payload),
                     "csv": None, "pdf": None, "pdf_error": None,
                     "example": st.session_state.get("active_example") if mode == MODE_TEXT else None}


def stale_notice(mode: str, current_key: tuple) -> None:
    bundle = st.session_state["results"].get(mode)
    if bundle is not None and bundle["key"] != current_key:
        st.markdown(ui.note_box("The inputs or settings changed since this result was produced. "
                                "Press Analyze to refresh it.", "warning"), unsafe_allow_html=True)


# ============================================================ results: shared
def render_exports(mode: str) -> None:
    bundle = st.session_state["results"][mode]
    analysis = bundle["analysis"]
    st.markdown(ui.section_header("Export results", "Download exactly what is shown above."), unsafe_allow_html=True)
    if bundle["csv"] is None:
        bundle["csv"] = to_csv_bytes(analysis)
    if bundle["pdf"] is None and bundle["pdf_error"] is None:
        try:
            bundle["pdf"] = build_pdf_report(analysis)
        except Exception as exc:  # never let report generation crash the dashboard
            bundle["pdf_error"] = str(exc)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇  Download results as CSV", data=bundle["csv"], file_name=export_filename(analysis, "csv"),
                           mime="text/csv", key=f"dl_csv_{mode}")
    with c2:
        if bundle["pdf"] is not None:
            st.download_button("⬇  Download PDF report", data=bundle["pdf"], file_name=export_filename(analysis, "pdf"),
                               mime="application/pdf", key=f"dl_pdf_{mode}")
        else:
            st.error(f"The PDF report could not be generated: {bundle['pdf_error']}")


# ======================================================== results: Text vs Text
def render_text_result() -> None:
    bundle = st.session_state["results"][MODE_TEXT]
    result, analysis = bundle["payload"], bundle["analysis"]
    row = analysis.rows[0]

    st.markdown(ui.section_header("Results", "Prediction from the pretrained NLI model, with its probabilities.",
                                  "Analysis"), unsafe_allow_html=True)
    scores = {"entailment": row["Entailment"], "contradiction": row["Contradiction"], "neutral": row["Neutral"]}
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(ui.classification_card(row["Status"], row["NLI label"], scores), unsafe_allow_html=True)
    with right:
        st.markdown(ui.prob_bars(row["Entailment"], row["Contradiction"], row["Neutral"]), unsafe_allow_html=True)

    c0, c1, c2, c3 = st.columns(4)
    c0.metric("Cosine similarity (MiniLM)", f"{result['similarity']:.4f}")
    c1.metric("Entailment", f"{result['entailment_score']:.4f}")
    c2.metric("Contradiction", f"{result['contradiction_score']:.4f}")
    c3.metric("Neutral", f"{result['neutral_score']:.4f}")
    st.caption("Similarity measures topical closeness, not agreement. High similarity does not mean entailment.")

    kind = STATUS_STYLE[row["Status"]][0]
    st.markdown(ui.statement_compare(result["statement_a"], "Statement A", result["statement_b"], "Statement B",
                                     "Statement A - premise", "Statement B - hypothesis", kind), unsafe_allow_html=True)
    if row["Notes"]:
        st.markdown(ui.note_box(row["Notes"], "warning"), unsafe_allow_html=True)

    ex = EXAMPLE_BY_NAME.get(bundle.get("example") or "")
    if ex and (ex["text_a"], ex["text_b"]) == (result["statement_a"], result["statement_b"]):
        expected_status = RELATIONSHIP_STATUS.get(ex["expected"])
        if expected_status is not None and expected_status != RELATIONSHIP_STATUS[result["relationship"]]:
            st.markdown(ui.note_box(
                f"Demo note: a human reader would label this example '{ex['expected']}', but the model predicted "
                f"'{result['relationship']}'. The prediction above is the model's real output, shown unchanged - "
                "small NLI models can be wrong.", "warning"), unsafe_allow_html=True)

    with st.expander("Basic linguistic analysis (spaCy)"):
        col_a, col_b = st.columns(2)
        for col, title, txt in ((col_a, "Statement A", result["statement_a"]),
                                (col_b, "Statement B", result["statement_b"])):
            info = cached_linguistic_info(txt)
            col.markdown(f"**{title}**")
            col.write(f"Tokens: {info['num_tokens']} | Sentences: {info['num_sentences']}")
            col.write(f"Keywords: {', '.join(info['keywords']) or '-'}")
            col.write(f"Negation cues: {', '.join(info['negations']) or '-'}")
            col.write(f"Entities: {', '.join(f'{t} ({l})' for t, l in info['entities']) or '-'}")
    render_exports(MODE_TEXT)


# ================================================= results: document comparisons
def render_report_result(mode: str) -> None:
    bundle = st.session_state["results"][mode]
    report, analysis = bundle["payload"], bundle["analysis"]

    st.markdown(ui.section_header("Document analysis summary", "How the retrieval step narrowed the work before NLI.",
                                  "Analysis"), unsafe_allow_html=True)
    st.markdown(ui.note_box(f"{report.premise_label}   |   {report.hypothesis_label}", "info"), unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Premise sentences", report.premise_count)
    m2.metric("Hypothesis sentences", report.hypothesis_count)
    m3.metric("Relevant pairs analysed", report.candidate_pairs,
              help=f"Out of {report.total_possible_pairs} possible combinations.")
    m4.metric("Best similarity", "-" if report.best_similarity is None else f"{report.best_similarity:.2f}")
    st.caption(f"Semantic retrieval narrowed {report.total_possible_pairs} possible combinations "
               f"to {report.candidate_pairs} pairs before NLI.")
    for msg in report.messages:
        st.warning(msg)

    if not report.pairs:
        render_exports(mode)
        return

    s = report.summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Potential contradictions", s[STATUS_CONTRADICTION])
    c2.metric("Needs review", s[STATUS_REVIEW])
    c3.metric("Consistent", s[STATUS_CONSISTENT])
    c4.metric("Neutral", s[STATUS_NEUTRAL])
    if s[STATUS_CONTRADICTION] == 0 and s[STATUS_REVIEW] == 0:
        st.markdown(ui.note_box("No potential contradictions were flagged at the current thresholds. "
                                "That does not prove the documents are consistent - lower the thresholds or "
                                "read the neutral pairs to be sure.", "info"), unsafe_allow_html=True)

    # ---- results table (filterable, readable)
    st.markdown(ui.section_header("Results table", "Sorted by severity. Expand a pair below for full detail."),
                unsafe_allow_html=True)
    present = [name for name in STATUS_STYLE if s[name] > 0]
    chosen = st.multiselect("Show statuses", present, default=present, key=f"filter_{mode}")
    shown = [r for r in analysis.rows if r["Status"] in chosen]
    if shown:
        st.markdown(ui.results_table(shown), unsafe_allow_html=True)
    else:
        st.info("No pairs match the selected statuses.")

    # ---- detailed expandable comparisons
    st.markdown(ui.section_header("Detailed comparisons", "Potential contradictions are highlighted and opened first."),
                unsafe_allow_html=True)
    opened = 0
    for pair, row in zip(report.pairs, analysis.rows):
        if row["Status"] not in chosen:
            continue
        kind = STATUS_STYLE[row["Status"]][0]
        title = (f"{ui.status_dot(row['Status'])}  #{row['No']}  {row['Status']}  -  "
                 f"contradiction {row['Contradiction']:.2f}  |  similarity {row['Similarity']:.2f}")
        expand = row["Status"] == STATUS_CONTRADICTION and opened < 3
        opened += int(expand)
        with st.expander(title, expanded=expand):
            st.markdown(ui.statement_compare(
                pair.premise.text, pair.premise.location, pair.hypothesis.text, pair.hypothesis.location,
                "Premise", "Hypothesis", kind), unsafe_allow_html=True)
            st.markdown(ui.score_strip(row["Similarity"], row["Entailment"], row["Contradiction"], row["Neutral"]),
                        unsafe_allow_html=True)
            st.markdown(ui.prob_bars(row["Entailment"], row["Contradiction"], row["Neutral"]), unsafe_allow_html=True)
            st.caption(f"NLI label: {row['NLI label']}  |  model confidence: {row['Confidence']}")
            if pair.reverse:
                st.caption(f"Reverse direction (hypothesis -> premise): {pair.reverse['relationship']} "
                           f"(ent {pair.reverse['entailment']:.3f}, con {pair.reverse['contradiction']:.3f}, "
                           f"neu {pair.reverse['neutral']:.3f})")
            for note in pair.notes:
                st.markdown(ui.note_box(note, "warning"), unsafe_allow_html=True)
            if pair.premise.context or pair.hypothesis.context:
                st.markdown(ui.source_paragraphs(pair.premise.context, pair.premise.text,
                                                 pair.hypothesis.context, pair.hypothesis.text),
                            unsafe_allow_html=True)
    render_exports(mode)


def render_result(mode: str) -> None:
    if mode not in st.session_state["results"]:
        return
    if mode == MODE_TEXT:
        render_text_result()
    else:
        render_report_result(mode)


# ============================================================ input: documents
def pick_document(label: str, key: str, sample_path, sample_name: str):
    """Uploader (or bundled fictional sample). Returns (ExtractedDocument | None, error message | None)."""
    use_sample = st.checkbox(f"Use bundled sample: {sample_name} (fictional test data)", key=f"{key}_sample")
    try:
        if use_sample:
            return cached_extract_sample(str(sample_path)), None
        uploaded = st.file_uploader(label, type=UPLOAD_TYPES, key=f"{key}_upload")
        if uploaded is not None:
            return cached_extract_upload(uploaded.name, uploaded.getvalue()), None
    except DocumentError as exc:
        return None, str(exc)
    except FileNotFoundError:
        return None, f"The bundled sample '{sample_name}' is missing from the sample_data folder."
    return None, None


def describe_doc(doc) -> str:
    extra = f", {doc.page_count} pages" if doc.page_count else ""
    return f"{doc.name}: {doc.num_sentences} sentences, {doc.num_paragraphs} paragraphs{extra}"


# ====================================================================== modes
def text_vs_text() -> None:
    st.markdown(ui.section_header(
        "Demo playground", "Pick an example to fill both boxes, then press Analyze. "
        "The result always comes from the real NLP models - nothing is pre-computed."), unsafe_allow_html=True)
    cols = st.columns(len(DEMO_EXAMPLES))
    for i, (col, ex) in enumerate(zip(cols, DEMO_EXAMPLES), start=1):
        col.button(f"{i}. {ex['name']}", key=f"ex_{i}", on_click=load_example, args=(ex["name"],))

    text_a = st.text_area("Statement A (premise)", key="text_a", height=120,
                          placeholder="e.g. The library is open on Sundays.")
    text_b = st.text_area("Statement B (hypothesis)", key="text_b", height=120,
                          placeholder="e.g. The library is closed every Sunday.")
    st.caption("NLI is directional: Statement A is the premise and Statement B the hypothesis. "
               "Swapping them can change the result.")

    active = EXAMPLE_BY_NAME.get(st.session_state.get("active_example") or "")
    if active and (active["text_a"], active["text_b"]) == (text_a, text_b):
        st.markdown(ui.note_box(f"Demo example '{active['name']}': a human reader would expect "
                                f"'{active['expected']}'. That is only an expectation - the model decides the result.",
                                "info"), unsafe_allow_html=True)

    key = ("tt", text_a.strip(), text_b.strip())
    clicked = st.button("Analyze", type="primary", key="analyze_tt")
    if clicked:
        if not text_a.strip() or not text_b.strip():
            st.error("Please enter text in BOTH Statement A and Statement B.")
        else:
            run_analysis(MODE_TEXT, key, lambda: compare_statements(text_a, text_b))
    stale_notice(MODE_TEXT, key)
    render_result(MODE_TEXT)


def document_vs_document(config: ComparisonConfig) -> None:
    st.caption("Document A sentences are the NLI premises; Document B sentences are the hypotheses.")
    st.button("Load the two bundled sample circulars", key="load_dd_samples", on_click=load_sample_documents)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Document A**")
        doc_a, err_a = pick_document("Upload Document A (PDF, DOCX or TXT)", "dd_a", SAMPLE_CIRCULAR_A, "Circular A (PDF)")
    with col2:
        st.markdown("**Document B**")
        doc_b, err_b = pick_document("Upload Document B (PDF, DOCX or TXT)", "dd_b", SAMPLE_CIRCULAR_B, "Circular B (DOCX)")
    for col, doc, err in ((col1, doc_a, err_a), (col2, doc_b, err_b)):
        if err:
            col.error(err)
        elif doc is not None:
            col.caption(describe_doc(doc))

    key = ("dd", doc_fingerprint(doc_a) if doc_a else None, doc_fingerprint(doc_b) if doc_b else None, astuple(config))
    clicked = st.button("Analyze", type="primary", key="analyze_dd")
    if clicked:
        if doc_a is None or doc_b is None:
            st.error("Please provide BOTH documents (upload files or use the bundled samples).")
        else:
            run_analysis(MODE_DOC, key, lambda: compare_documents(doc_a, doc_b, config))
    stale_notice(MODE_DOC, key)
    render_result(MODE_DOC)


def text_vs_document(config: ComparisonConfig) -> None:
    st.button("Load demo statement + bundled sample circular", key="load_td_demo", on_click=load_text_doc_demo)
    statement = st.text_area("Statement to check", key="tvd_statement", height=100, placeholder=EXAMPLE_TEXT_VS_DOC)
    st.caption("Relevant document sentences are the NLI premises (evidence); your statement is the hypothesis.")
    doc, err = pick_document("Upload a reference document (PDF, DOCX or TXT)", "td", SAMPLE_CIRCULAR_A, "Circular A (PDF)")
    if err:
        st.error(err)
    elif doc is not None:
        st.caption(describe_doc(doc))

    key = ("td", statement.strip(), doc_fingerprint(doc) if doc else None, astuple(config))
    clicked = st.button("Analyze", type="primary", key="analyze_td")
    if clicked:
        if not statement.strip():
            st.error("Please enter a statement to check.")
        elif doc is None:
            st.error("Please upload a reference document (or use the bundled sample).")
        else:
            run_analysis(MODE_TEXT_DOC, key, lambda: compare_text_to_document(statement, doc, config))
    stale_notice(MODE_TEXT_DOC, key)
    render_result(MODE_TEXT_DOC)


def workspace(config: ComparisonConfig) -> None:
    st.markdown(ui.section_header("Analysis workspace", "Choose a comparison mode, provide the inputs, press Analyze."),
                unsafe_allow_html=True)
    mode = st.radio("Comparison mode", MODES, horizontal=True, key="mode", label_visibility="collapsed")
    if mode == MODE_TEXT:
        text_vs_text()
    elif mode == MODE_DOC:
        document_vs_document(config)
    else:
        text_vs_document(config)


# ================================================================== about page
def about() -> None:
    st.markdown(ui.section_header("What CONTRADIX does", eyebrow="About"), unsafe_allow_html=True)
    st.markdown(
        "CONTRADIX reads two statements, or two documents, and checks whether the information in them "
        "**agrees, disagrees or is unrelated**. It combines two NLP techniques: *semantic similarity* to find the "
        "sentences that talk about the same thing, and *natural language inference (NLI)* to judge how those "
        "sentences relate. The output is a list of **potential** conflicts for a human to review - not a legal or factual verdict.")
    st.markdown(ui.step_flow([
        ("1 · Input", "Text, PDF, DOCX or TXT"), ("2 · Structure", "Paragraphs and sentences with page refs"),
        ("3 · Embed", "MiniLM sentence vectors"), ("4 · Retrieve", "Keep only relevant pairs"),
        ("5 · NLI", "Entailment / Contradiction / Neutral"), ("6 · Report", "Dashboard, CSV, PDF")]),
        unsafe_allow_html=True)

    st.markdown(ui.section_header("How semantic similarity works"), unsafe_allow_html=True)
    st.markdown(
        "A Sentence-Transformers model (`all-MiniLM-L6-v2`) turns every sentence into a vector of 384 numbers so that "
        "sentences with similar meaning end up close together. Closeness is measured with **cosine similarity**:")
    st.code("cosine(A, B) = (A · B) / (|A| × |B|)      # from -1 (opposite) to 1 (same direction)", language="text")
    st.markdown(ui.note_box(
        "Similarity answers 'are these about the same thing?', not 'do they agree?'. "
        "'The library is open on Sundays' and 'The library is closed on Sundays' are highly similar and still contradict "
        "each other. That is why CONTRADIX uses similarity only to choose which pairs deserve a closer look.", "info"),
        unsafe_allow_html=True)

    st.markdown(ui.section_header("How natural language inference works"), unsafe_allow_html=True)
    st.markdown(
        "NLI takes a **premise** and a **hypothesis** and asks: *if the premise is true, what can we say about the "
        "hypothesis?* CONTRADIX uses a pretrained cross-encoder (`cross-encoder/nli-deberta-v3-small`). It reads both "
        "sentences together, produces three scores and converts them with softmax into probabilities that add up to 1. "
        "The highest probability gives the label. NLI is **directional**: swapping premise and hypothesis can change the answer.")
    st.markdown(ui.info_cards([
        {"title": "Entailment", "kind": "success",
         "text": "The hypothesis follows from the premise. 'Every employee must carry an ID card' → 'Employees need ID cards'."},
        {"title": "Contradiction", "kind": "danger",
         "text": "The hypothesis cannot be true if the premise is. 'Open every Sunday' vs 'Closed on Sundays'."},
        {"title": "Neutral", "kind": "neutral",
         "text": "The premise neither proves nor disproves the hypothesis. 'The exam begins at 10 AM' vs 'The library has many books'."},
    ]), unsafe_allow_html=True)

    st.markdown(ui.section_header("Reading the results"), unsafe_allow_html=True)
    st.markdown(ui.info_cards([
        {"title": "Red · Potential contradiction", "kind": "danger",
         "text": "The contradiction probability reached the configured threshold (or the reverse direction did)."},
        {"title": "Amber · Needs review", "kind": "warning",
         "text": "The sentences share a subject but give different percentages, dates or durations. Might be an update, a different scope, or a real conflict."},
        {"title": "Green · Consistent", "kind": "success", "text": "The model predicts entailment."},
        {"title": "Grey-blue · Neutral", "kind": "neutral", "text": "No clear logical relationship was found."},
    ], columns=2), unsafe_allow_html=True)

    st.markdown(ui.section_header("NLP syllabus concepts implemented"), unsafe_allow_html=True)
    st.markdown(ui.info_cards([
        {"title": "Unit I · Finding the Structure of Documents",
         "text": "Extracting text from PDF, DOCX and TXT; splitting it into paragraphs and sentences with spaCy; keeping page and paragraph references so every result can be traced back to its source."},
        {"title": "Unit III · Semantic Interpretation and Ambiguity Resolution",
         "text": "NLI interprets how two sentences relate in context, including negation. Similarity is kept separate from entailment, condition words ('if', 'unless', 'only') and differing values are flagged, and low-confidence predictions are labelled as uncertain."},
        {"title": "Unit IV · Meaning Representation",
         "text": "Sentences are represented as dense embedding vectors (distributional meaning) and compared geometrically. Each pair is treated as a premise-hypothesis proposition test. This is vector-based, not formal logic."},
    ]), unsafe_allow_html=True)

    st.markdown(ui.section_header("Technologies used"), unsafe_allow_html=True)
    st.markdown(ui.tech_badges([
        ("Python 3.11", "language"), ("Streamlit", "dashboard + custom CSS"), ("spaCy", "sentence segmentation"),
        ("Sentence-Transformers", "all-MiniLM-L6-v2"), ("Hugging Face Transformers", "nli-deberta-v3-small"),
        ("PyTorch (CPU)", "model inference"), ("PyMuPDF", "PDF text"), ("python-docx", "DOCX text"),
        ("ReportLab", "PDF report"), ("pytest", "automated tests")]), unsafe_allow_html=True)

    st.markdown(ui.section_header("Known limitations"), unsafe_allow_html=True)
    st.markdown(ui.info_cards([
        {"title": "Small models", "text": "They can be confidently wrong, especially with long or domain-specific sentences."},
        {"title": "Context & conditions", "text": "A conflict may disappear once an exception, date or scope is considered. Always read the source paragraphs."},
        {"title": "Numbers & dates", "text": "Different values are not automatically contradictions; they are flagged for review."},
        {"title": "Scanned PDFs", "text": "Image-only PDFs need OCR, which is not included. Use a text-based PDF, DOCX or TXT."},
    ], columns=2), unsafe_allow_html=True)


# ====================================================================== page
st.markdown(CSS, unsafe_allow_html=True)
init_state()
config = sidebar_settings()
# Shared paper-and-ink styling for BOTH the landing page and analysis workspace.
st.markdown("""
<style>
.stApp, [data-testid="stAppViewContainer"] {
  background-color: #faf7f0 !important;
  color: #26343a !important;
}
.stApp::before {
  content: ""; position: fixed; inset: 0; pointer-events: none;
  background: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNjAwIDk1MCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQgc2xpY2UiPjxkZWZzPjxsaW5lYXJHcmFkaWVudCBpZD0icGFwZXIiIHgyPSIwIiB5Mj0iMSI+PHN0b3Agc3RvcC1jb2xvcj0iI2ZhZjdmMCIvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iI2VhZThlMCIvPjwvbGluZWFyR3JhZGllbnQ+PGxpbmVhckdyYWRpZW50IGlkPSJyaWRnZSIgeDI9IjAiIHkyPSIxIj48c3RvcCBzdG9wLWNvbG9yPSIjMzg0NjRiIiBzdG9wLW9wYWNpdHk9Ii41NyIvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzM4NDY0YiIgc3RvcC1vcGFjaXR5PSIuMDM1Ii8+PC9saW5lYXJHcmFkaWVudD48bGluZWFyR3JhZGllbnQgaWQ9ImZyb250IiB4Mj0iMCIgeTI9IjEiPjxzdG9wIHN0b3AtY29sb3I9IiMyODM2M2IiIHN0b3Atb3BhY2l0eT0iLjcyIi8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMjgzNjNiIiBzdG9wLW9wYWNpdHk9Ii4xMyIvPjwvbGluZWFyR3JhZGllbnQ+PGZpbHRlciBpZD0iZ3JhaW4iPjxmZVR1cmJ1bGVuY2UgdHlwZT0iZnJhY3RhbE5vaXNlIiBiYXNlRnJlcXVlbmN5PSIuNjUiIG51bU9jdGF2ZXM9IjMiIHNlZWQ9IjgiIHN0aXRjaFRpbGVzPSJzdGl0Y2giLz48ZmVDb2xvck1hdHJpeCB0eXBlPSJzYXR1cmF0ZSIgdmFsdWVzPSIwIi8+PGZlQ29tcG9uZW50VHJhbnNmZXI+PGZlRnVuY0EgdHlwZT0ibGluZWFyIiBzbG9wZT0iLjA3NSIvPjwvZmVDb21wb25lbnRUcmFuc2Zlcj48L2ZpbHRlcj48L2RlZnM+PHBhdGggZmlsbD0idXJsKCNwYXBlcikiIGQ9Ik0wIDBIMTYwMFY5NTBIMHoiLz48Y2lyY2xlIGN4PSIxMjEwIiBjeT0iMjUwIiByPSI3NiIgZmlsbD0iI2I5NTc0MyIgb3BhY2l0eT0iLjc4Ii8+PHBhdGggZD0iTTAgNTUwUTExMCA1MjggMjIwIDQ3MVQ0MjAgNDkyUTUxNSA0MzAgNjE1IDQ1OVQ4MDUgNDMyUTkyMCAzNTUgMTAyMCA0MjFUMTIxMCAzOTNRMTMzMCAzNTYgMTQ0MCA0MjlMMTYwMCAzODVWOTUwSDBaIiBmaWxsPSJ1cmwoI3JpZGdlKSIgb3BhY2l0eT0iLjM4Ii8+PHBhdGggZD0iTTAgNjUzUTEzMCA1NzAgMjQ4IDYxMlQ0NDMgNTUxUTU1MCA0ODggNjcwIDU2OVQ4NjUgNTEwUTk3MCA0NjUgMTA5MCA1MzVUMTMyMCA0NjlRMTQ1MCA0MzYgMTYwMCA1MjBWOTUwSDBaIiBmaWxsPSJ1cmwoI3JpZGdlKSIgb3BhY2l0eT0iLjY1Ii8+PHBhdGggZD0iTTAgNzQ4UTE0MCA2ODAgMjU1IDcxN1Q0NTMgNjQ0UTU4NSA1OTcgNzA1IDY3MlQ5MDIgNjA0UTEwMzAgNTY4IDExNTAgNjMyVDEzODAgNTU3UTE1MTAgNTI5IDE2MDAgNTgyVjk1MEgwWiIgZmlsbD0idXJsKCNmcm9udCkiIG9wYWNpdHk9Ii43MiIvPjxwYXRoIGQ9Ik0wIDgyOVExNDggNzc2IDI4MCA4MDJUNTAwIDczOFE2MzAgNjk5IDc3MCA3NjBUOTk1IDcxOFExMTM1IDY4MCAxMjUwIDczMlQxNDcwIDY3NUwxNjAwIDY5N1Y5NTBIMFoiIGZpbGw9InVybCgjZnJvbnQpIiBvcGFjaXR5PSIuNTMiLz48cGF0aCBkPSJNMCAwSDE2MDBWOTUwSDB6IiBmaWx0ZXI9InVybCgjZ3JhaW4pIiBvcGFjaXR5PSIuNDUiLz48L3N2Zz4=') center bottom / cover no-repeat;
  opacity: .24; z-index: 0;
}
[data-testid="stAppViewContainer"] > .main {
  position: relative; z-index: 1; background: transparent !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"] {
  background: transparent !important;
}
[data-testid="stMainBlockContainer"] {
  background: transparent !important;
}
h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] {
  color: #26343a;
}
h1, h2 { font-family: Georgia, 'Times New Roman', serif !important; }
[data-testid="stTabs"] [role="tablist"] {
  border-bottom: 1px solid #c9c6bd;
}
[data-testid="stTabs"] button[role="tab"] {
  color: #536061 !important; background: transparent !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: #293b3e !important; border-bottom-color: #a45240 !important;
}
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] > div > div,
[data-testid="stFileUploader"] section {
  background: #fffdf8 !important; color: #26343a !important;
  border-color: #d8d3c8 !important; border-radius: 10px !important;
}
[data-testid="stFileUploader"] section *,
[data-testid="stTextArea"] textarea::placeholder {
  color: #596465 !important;
}
.stButton > button, .stDownloadButton > button {
  border-radius: 24px !important; border: 1px solid #aab2ae !important;
  background: #fffdf8 !important; color: #293b3e !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: #a45240 !important; color: #a45240 !important;
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: #293b3e !important; border-color: #293b3e !important; color: #fff !important;
}
.stButton > button[kind="primary"]:hover {
  background: #42595b !important; color: #fff !important;
}
[data-testid="stExpander"], [data-testid="stMetric"],
[data-testid="stDataFrame"], [data-testid="stAlert"] {
  border-color: #d8d3c8 !important; border-radius: 12px !important;
}
hr { border-color: #d8d3c8 !important; }
</style>
""", unsafe_allow_html=True)
if st.session_state["show_workspace"]:
    if st.button("← Back to home", key="back_home"):
        st.session_state["show_workspace"] = False
        st.rerun()
    st.title("CONTRADIX")
    tab_work, tab_about = st.tabs(["🔬  Analysis workspace", "📘  About & methodology"])
    with tab_work:
        workspace(config)
    with tab_about:
        about()
else:
    # Static sumi-e illustration; no React, JavaScript or external asset required.
    st.markdown(f"""
    <style>
    .stApp {{ background: #faf7f0 !important; color: #26343a !important; }}
    .stApp::before {{ content: ''; position: fixed; inset: 0; pointer-events: none;
        background: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNjAwIDk1MCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQgc2xpY2UiPjxkZWZzPjxsaW5lYXJHcmFkaWVudCBpZD0icGFwZXIiIHgyPSIwIiB5Mj0iMSI+PHN0b3Agc3RvcC1jb2xvcj0iI2ZhZjdmMCIvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iI2VhZThlMCIvPjwvbGluZWFyR3JhZGllbnQ+PGxpbmVhckdyYWRpZW50IGlkPSJyaWRnZSIgeDI9IjAiIHkyPSIxIj48c3RvcCBzdG9wLWNvbG9yPSIjMzg0NjRiIiBzdG9wLW9wYWNpdHk9Ii41NyIvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzM4NDY0YiIgc3RvcC1vcGFjaXR5PSIuMDM1Ii8+PC9saW5lYXJHcmFkaWVudD48bGluZWFyR3JhZGllbnQgaWQ9ImZyb250IiB4Mj0iMCIgeTI9IjEiPjxzdG9wIHN0b3AtY29sb3I9IiMyODM2M2IiIHN0b3Atb3BhY2l0eT0iLjcyIi8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMjgzNjNiIiBzdG9wLW9wYWNpdHk9Ii4xMyIvPjwvbGluZWFyR3JhZGllbnQ+PGZpbHRlciBpZD0iZ3JhaW4iPjxmZVR1cmJ1bGVuY2UgdHlwZT0iZnJhY3RhbE5vaXNlIiBiYXNlRnJlcXVlbmN5PSIuNjUiIG51bU9jdGF2ZXM9IjMiIHNlZWQ9IjgiIHN0aXRjaFRpbGVzPSJzdGl0Y2giLz48ZmVDb2xvck1hdHJpeCB0eXBlPSJzYXR1cmF0ZSIgdmFsdWVzPSIwIi8+PGZlQ29tcG9uZW50VHJhbnNmZXI+PGZlRnVuY0EgdHlwZT0ibGluZWFyIiBzbG9wZT0iLjA3NSIvPjwvZmVDb21wb25lbnRUcmFuc2Zlcj48L2ZpbHRlcj48L2RlZnM+PHBhdGggZmlsbD0idXJsKCNwYXBlcikiIGQ9Ik0wIDBIMTYwMFY5NTBIMHoiLz48Y2lyY2xlIGN4PSIxMjEwIiBjeT0iMjUwIiByPSI3NiIgZmlsbD0iI2I5NTc0MyIgb3BhY2l0eT0iLjc4Ii8+PHBhdGggZD0iTTAgNTUwUTExMCA1MjggMjIwIDQ3MVQ0MjAgNDkyUTUxNSA0MzAgNjE1IDQ1OVQ4MDUgNDMyUTkyMCAzNTUgMTAyMCA0MjFUMTIxMCAzOTNRMTMzMCAzNTYgMTQ0MCA0MjlMMTYwMCAzODVWOTUwSDBaIiBmaWxsPSJ1cmwoI3JpZGdlKSIgb3BhY2l0eT0iLjM4Ii8+PHBhdGggZD0iTTAgNjUzUTEzMCA1NzAgMjQ4IDYxMlQ0NDMgNTUxUTU1MCA0ODggNjcwIDU2OVQ4NjUgNTEwUTk3MCA0NjUgMTA5MCA1MzVUMTMyMCA0NjlRMTQ1MCA0MzYgMTYwMCA1MjBWOTUwSDBaIiBmaWxsPSJ1cmwoI3JpZGdlKSIgb3BhY2l0eT0iLjY1Ii8+PHBhdGggZD0iTTAgNzQ4UTE0MCA2ODAgMjU1IDcxN1Q0NTMgNjQ0UTU4NSA1OTcgNzA1IDY3MlQ5MDIgNjA0UTEwMzAgNTY4IDExNTAgNjMyVDEzODAgNTU3UTE1MTAgNTI5IDE2MDAgNTgyVjk1MEgwWiIgZmlsbD0idXJsKCNmcm9udCkiIG9wYWNpdHk9Ii43MiIvPjxwYXRoIGQ9Ik0wIDgyOVExNDggNzc2IDI4MCA4MDJUNTAwIDczOFE2MzAgNjk5IDc3MCA3NjBUOTk1IDcxOFExMTM1IDY4MCAxMjUwIDczMlQxNDcwIDY3NUwxNjAwIDY5N1Y5NTBIMFoiIGZpbGw9InVybCgjZnJvbnQpIiBvcGFjaXR5PSIuNTMiLz48cGF0aCBkPSJNMCAwSDE2MDBWOTUwSDB6IiBmaWx0ZXI9InVybCgjZ3JhaW4pIiBvcGFjaXR5PSIuNDUiLz48L3N2Zz4=') center bottom / cover no-repeat;
        opacity: .94; z-index: 0; }}
    [data-testid="stAppViewContainer"] > .main {{ position: relative; z-index: 1; background: transparent; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    .contradix-landing {{ text-align: center; padding: clamp(5rem, 16vh, 10rem) 0 1.2rem; }}
    .contradix-landing .eyebrow {{ font-size: .69rem; letter-spacing: .32em; font-weight: 650; color: #785e54; }}
    .contradix-landing h1 {{ font-family: Georgia, serif; font-size: clamp(3.4rem, 8vw, 7.3rem);
        font-weight: 400; letter-spacing: -.055em; line-height: 1.05; color: #24343a; margin: 1.2rem 0; }}
    .contradix-landing .tagline {{ font-family: Georgia, serif; font-size: clamp(1.2rem, 2.5vw, 1.85rem);
        color: #344249; margin: 0 auto 1.15rem; }}
    .contradix-landing .description {{ max-width: 550px; margin: auto; font-size: .94rem;
        line-height: 1.8; color: #4c5657; }}
    .stButton > button[kind="primary"] {{ border-radius: 30px; background: #293b3e; border: 0;
        color: #fff; min-height: 3.2rem; }}
    .stButton > button[kind="primary"]:hover {{ background: #42595b; color: #fff; }}
    </style>
    <section class="contradix-landing">
       <div class="eyebrow">SEMANTIC CONTRADICTION DETECTION</div>
       <h1>CONTRADIX</h1>
       <p class="tagline">Beyond Words. Detect the Contradiction.</p>
       <p class="description">An NLP-powered semantic analysis platform for identifying potentially conflicting information across statements and documents.</p>
    </section>
    """, unsafe_allow_html=True)
    left, middle, right = st.columns([1.2, 1, 1.2])
    with middle:
        if st.button("Start analyzing →", type="primary", use_container_width=True, key="start_analysis"):
            st.session_state["show_workspace"] = True
            st.rerun()
    st.markdown("<div style='height:clamp(8rem,19vh,15rem)'></div>", unsafe_allow_html=True)
    with st.expander("About & methodology"):
        about()
if st.session_state["show_workspace"]:
    st.divider()
    st.caption("CONTRADIX · Semantic Contradiction Detection and Document Consistency Analysis Using NLP · "
               "Results are potential findings for human review.")
