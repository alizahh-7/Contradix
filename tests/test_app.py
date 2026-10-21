"""UI tests with Streamlit's AppTest. Only the two model calls are faked (conftest);
extraction, comparison logic and the real app.py code all run."""
from streamlit.testing.v1 import AppTest

APP = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "app.py")


def fresh():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def test_text_vs_text_mode_empty_and_valid(fake_backend):
    at = fresh()
    assert at.title[0].value == "CONTRADIX"
    at.button(key="analyze_tt").click().run()
    assert any("BOTH" in e.value for e in at.error)
    at.text_area(key="text_a").set_value("The library is open on Sundays.")
    at.text_area(key="text_b").set_value("The library is closed every Sunday.")
    at.button(key="analyze_tt").click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Entailment" in labels and "Contradiction" in labels and "Neutral" in labels
    assert fake_backend.nli_calls  # real backend path was used


def test_document_vs_document_mode_with_samples(fake_backend):
    at = fresh()
    at.radio(key="mode").set_value("Document vs Document").run()
    at.button(key="analyze_dd").click().run()
    assert any("BOTH documents" in e.value for e in at.error)
    at.checkbox(key="dd_a_sample").check()
    at.checkbox(key="dd_b_sample").check().run()
    at.button(key="analyze_dd").click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Relevant pairs analysed" in labels and "Potential contradictions" in labels
    assert fake_backend.nli_calls


def test_text_vs_document_mode_with_sample(fake_backend):
    at = fresh()
    at.radio(key="mode").set_value("Text vs Document").run()
    at.button(key="analyze_td").click().run()
    assert at.error  # nothing provided
    at.text_area(key="tvd_statement").set_value("Students with 60% attendance are eligible for examinations.")
    at.sidebar.slider[0].set_value(0.30)   # similarity threshold (the toy fake embedding scores lower than a real model)
    at.checkbox(key="td_sample").check().run()
    at.button(key="analyze_td").click().run()
    assert not at.exception
    assert "Relevant pairs analysed" in [m.label for m in at.metric]
    assert fake_backend.nli_calls


# ------------------------------------------------------------------ Phase 3 UI
import csv  # noqa: E402
import html  # noqa: E402
import io  # noqa: E402

import pymupdf  # noqa: E402
import pytest  # noqa: E402

from utils.examples import DEMO_EXAMPLES  # noqa: E402

TT, DD, TD = "Text vs Text", "Document vs Document", "Text vs Document"


def all_text(at) -> str:
    """Everything rendered as markdown/caption/error in the main area (for content assertions)."""
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [e.value for e in at.error]
    return html.unescape("\n".join(parts))


def csv_rows(at, mode):
    raw = at.session_state["results"][mode]["csv"].decode("utf-8").lstrip("﻿")
    return list(csv.DictReader(io.StringIO(raw)))


def test_hero_and_sections_render(fake_backend):
    at = fresh()
    text = all_text(at)
    assert "Beyond Words. Detect the Contradiction." in text
    assert "An NLP-powered semantic analysis platform for identifying potentially conflicting information" in text
    for card in (TT, DD, TD):
        assert f'<div class="cx-card-title">{card}</div>' in text
    assert [t.label.strip() for t in at.tabs] == ["🔬  Analysis workspace", "📘  About & methodology"]
    for heading in ("What CONTRADIX does", "How semantic similarity works", "How natural language inference works",
                    "NLP syllabus concepts implemented", "Technologies used"):
        assert heading in text
    assert "Unit I" in text and "Unit III" in text and "Unit IV" in text
    assert not fake_backend.nli_calls                     # opening the app never runs inference


@pytest.mark.parametrize("i,ex", list(enumerate(DEMO_EXAMPLES, start=1)))
def test_demo_examples_populate_inputs_without_running_the_model(fake_backend, i, ex):
    at = fresh()
    at.button(key=f"ex_{i}").click().run()
    assert at.text_area(key="text_a").value == ex["text_a"]
    assert at.text_area(key="text_b").value == ex["text_b"]
    assert not fake_backend.nli_calls                     # selecting an example is not an analysis
    assert "results" not in at.session_state or TT not in at.session_state["results"]


def test_demo_example_then_analyze_uses_the_real_engine_and_shows_its_output(fake_backend):
    at = fresh()
    at.button(key="ex_1").click().run()                   # contradiction demo
    at.button(key="analyze_tt").click().run()
    assert not at.exception
    assert fake_backend.nli_calls == [("The library is open every Sunday.", "The library remains closed on Sundays.")]
    row = at.session_state["results"][TT]["analysis"].rows[0]
    assert row["NLI label"] == "Contradiction"            # produced by the (fake) backend, not by the example table
    assert "Potential contradiction" in all_text(at)


def test_demo_note_when_model_disagrees_with_human_expectation(fake_backend):
    at = fresh()
    at.button(key="ex_2").click().run()                   # entailment demo; the toy fake NLI answers Neutral
    at.button(key="analyze_tt").click().run()
    assert at.session_state["results"][TT]["analysis"].rows[0]["NLI label"] == "Neutral"
    assert "a human reader would label this example 'Entailment', but the model predicted 'Neutral'" in all_text(at)


def test_policy_example_is_flagged_for_review_not_called_a_contradiction(fake_backend):
    at = fresh()
    at.button(key="ex_4").click().run()
    at.button(key="analyze_tt").click().run()
    row = at.session_state["results"][TT]["analysis"].rows[0]
    assert row["Status"] == "Needs review (differing values)" and "75%" in row["Notes"] and "60%" in row["Notes"]


def test_result_survives_interaction_and_does_not_recompute(fake_backend):
    at = fresh()
    at.button(key="ex_1").click().run()
    at.button(key="analyze_tt").click().run()
    calls = len(fake_backend.nli_calls)
    at.radio(key="mode").set_value(DD).run()              # leave the mode ...
    at.radio(key="mode").set_value(TT).run()              # ... and come back
    assert len(fake_backend.nli_calls) == calls
    assert "Potential contradiction" in all_text(at)       # still displayed
    at.button(key="analyze_tt").click().run()             # identical inputs -> no duplicate inference
    assert len(fake_backend.nli_calls) == calls


def test_changed_inputs_show_stale_notice_and_reanalyse(fake_backend):
    at = fresh()
    at.button(key="ex_1").click().run()
    at.button(key="analyze_tt").click().run()
    at.text_area(key="text_b").set_value("The library is open on Sundays too.").run()
    assert "Press Analyze to refresh" in all_text(at)
    calls = len(fake_backend.nli_calls)
    at.button(key="analyze_tt").click().run()
    assert len(fake_backend.nli_calls) == calls + 1
    assert "Press Analyze to refresh" not in all_text(at)


def test_whitespace_only_inputs_are_rejected(fake_backend):
    at = fresh()
    at.text_area(key="text_a").set_value("   ")
    at.text_area(key="text_b").set_value("\n")
    at.button(key="analyze_tt").click().run()
    assert any("BOTH" in e.value for e in at.error) and not fake_backend.nli_calls


def test_overlong_input_shows_friendly_error(fake_backend):
    at = fresh()
    at.text_area(key="text_a").set_value("word " * 5000)
    at.text_area(key="text_b").set_value("The library is closed.")
    at.button(key="analyze_tt").click().run()
    assert not at.exception and at.error


def test_text_mode_exports_are_real_csv_and_pdf(fake_backend):
    at = fresh()
    at.button(key="ex_1").click().run()
    at.button(key="analyze_tt").click().run()
    assert len(at.get("download_button")) == 2
    rows = csv_rows(at, TT)
    assert rows[0]["Premise"] == "The library is open every Sunday." and rows[0]["Status"] == "Potential contradiction"
    pdf = at.session_state["results"][TT]["pdf"]
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        text = doc[0].get_text().replace("\n", " ")
    assert "The library remains closed on Sundays." in text and "Potential contradiction" in text


def test_document_mode_settings_are_connected_and_exports_work(fake_backend):
    at = fresh()
    at.radio(key="mode").set_value(DD).run()
    at.button(key="load_dd_samples").click().run()
    assert at.checkbox(key="dd_a_sample").value and at.checkbox(key="dd_b_sample").value
    at.sidebar.slider(key="cfg_sim").set_value(0.30)
    at.sidebar.slider(key="cfg_max").set_value(5)
    at.button(key="analyze_dd").click().run()
    assert not at.exception
    report = at.session_state["results"][DD]["payload"]
    assert report.candidate_pairs <= 5 and report.config.similarity_threshold == pytest.approx(0.30)
    assert report.config.max_pairs == 5 and not report.config.bidirectional
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Relevant pairs analysed"] == str(report.candidate_pairs)
    assert {"Potential contradictions", "Needs review", "Consistent", "Neutral"} <= set(metrics)
    assert "cx-table" in all_text(at) and "circular_a_original_policy.pdf" in all_text(at)
    rows = csv_rows(at, DD)
    assert len(rows) == len(report.pairs) and rows[0]["Premise source"].startswith("circular_a_original_policy.pdf")
    with pymupdf.open(stream=at.session_state["results"][DD]["pdf"], filetype="pdf") as doc:
        assert "circular_b_revised_notice.docx" in " ".join(p.get_text() for p in doc)


def test_reverse_direction_setting_doubles_nli_calls(fake_backend):
    at = fresh()
    at.radio(key="mode").set_value(DD).run()
    at.button(key="load_dd_samples").click().run()
    at.sidebar.slider(key="cfg_sim").set_value(0.30)
    at.button(key="analyze_dd").click().run()
    one_way = len(fake_backend.nli_calls)
    at.sidebar.checkbox(key="cfg_both").check().run()
    assert "Press Analyze to refresh" in all_text(at)     # settings are part of the result identity
    at.button(key="analyze_dd").click().run()
    assert len(fake_backend.nli_calls) == one_way * 2 + one_way   # +forward +reverse of the new run
    assert any(r["Reverse NLI label"] for r in csv_rows(at, DD))


def test_reset_button_restores_default_settings(fake_backend):
    at = fresh()
    at.sidebar.slider(key="cfg_sim").set_value(0.80)
    at.sidebar.slider(key="cfg_con").set_value(0.90).run()
    at.sidebar.button(key="cfg_reset").click().run()
    assert at.sidebar.slider(key="cfg_sim").value == pytest.approx(0.45)
    assert at.sidebar.slider(key="cfg_con").value == pytest.approx(0.50)


def test_text_vs_document_demo_button_and_no_evidence_message(fake_backend):
    at = fresh()
    at.radio(key="mode").set_value(TD).run()
    at.button(key="load_td_demo").click().run()
    assert at.text_area(key="tvd_statement").value.startswith("Students with 60% attendance")
    assert at.checkbox(key="td_sample").value
    at.sidebar.slider(key="cfg_sim").set_value(0.95)       # nothing can reach this similarity
    at.button(key="analyze_td").click().run()
    assert not at.exception
    assert any("No relevant evidence" in w.value for w in at.warning)
    assert len(at.get("download_button")) == 2             # exports still work for an empty result
    assert not fake_backend.nli_calls                      # no evidence -> NLI never ran


def test_results_are_kept_separately_per_mode(fake_backend):
    at = fresh()
    at.button(key="ex_1").click().run()
    at.button(key="analyze_tt").click().run()
    at.radio(key="mode").set_value(DD).run()
    at.button(key="load_dd_samples").click().run()
    at.sidebar.slider(key="cfg_sim").set_value(0.30)
    at.button(key="analyze_dd").click().run()
    assert set(at.session_state["results"]) == {TT, DD}
    assert "Potential contradictions" in [m.label for m in at.metric]


def test_backend_crash_is_reported_visibly_and_nothing_is_stored(monkeypatch, fake_backend):
    from nlp import inference

    def boom(pairs, batch_size=16):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(inference, "predict_nli_batch", boom)
    monkeypatch.setattr(inference, "predict_nli", lambda premise, hypothesis: boom([]))
    at = fresh()
    at.button(key="ex_1").click().run()
    at.button(key="analyze_tt").click().run()
    assert any("Something went wrong while running the NLP models" in e.value for e in at.error)
    assert TT not in at.session_state["results"] and not at.get("download_button")
