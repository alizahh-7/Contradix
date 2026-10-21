"""CONTRADIX final verification - runs the REAL models end to end.

    python verify_final.py

Checks: document extraction, Text vs Text (all demo examples), Document vs Document,
Text vs Document, the "no evidence" path, CSV export and PDF export.
The first run downloads the models (internet needed). Exit code 0 = OK, 1 = failure.
Predictions come from the pretrained models; the demo "expected" labels are only printed
for comparison and are never used as results.
"""
import csv
import io
import sys
import traceback

failures = 0


def check(name, fn):
    global failures
    try:
        detail = fn()
        print(f"[PASS] {name}" + (f" -> {detail}" if detail else ""))
        return True
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"[FAIL] {name}: {exc}")
        traceback.print_exc(limit=3)
        return False


def extraction():
    from nlp import extract_document_from_path
    from utils.file_handler import SAMPLE_DIR
    out = []
    for fname in ("circular_a_original_policy.pdf", "circular_b_revised_notice.docx",
                  "circular_a_original_policy.txt"):
        d = extract_document_from_path(str(SAMPLE_DIR / fname))
        assert d.num_sentences > 0
        out.append(f"{d.file_type.upper()}:{d.num_sentences} sentences")
    return ", ".join(out)


def demo_examples():
    from nlp import compare_statements
    from utils.examples import DEMO_EXAMPLES
    print()
    print(f"   {'Example':<22}{'Human expects':<32}{'Model predicted':<17}{'Sim':>6}{'Con':>7}{'Ent':>7}{'Neu':>7}")
    for ex in DEMO_EXAMPLES:
        r = compare_statements(ex["text_a"], ex["text_b"])
        total = r["entailment_score"] + r["contradiction_score"] + r["neutral_score"]
        assert abs(total - 1.0) < 1e-4, "NLI probabilities must sum to 1"
        print(f"   {ex['name']:<22}{ex['expected']:<32}{r['relationship']:<17}{r['similarity']:>6.2f}"
              f"{r['contradiction_score']:>7.2f}{r['entailment_score']:>7.2f}{r['neutral_score']:>7.2f}")
    print("   (Predictions come from the model; small NLI models can disagree with a human reader.)")
    return f"{len(DEMO_EXAMPLES)} demo pairs analysed"


def doc_vs_doc():
    from nlp import ComparisonConfig, compare_documents
    from utils.file_handler import SAMPLE_CIRCULAR_A, SAMPLE_CIRCULAR_B, load_sample
    report = compare_documents(load_sample(SAMPLE_CIRCULAR_A), load_sample(SAMPLE_CIRCULAR_B), ComparisonConfig())
    assert report.candidate_pairs > 0, "no relevant sentence pairs were retrieved"
    print(f"\n   {report.candidate_pairs} of {report.total_possible_pairs} possible pairs sent to NLI")
    for p in report.pairs:
        print(f"   [{p.status}] sim={p.similarity:.2f} con={p.contradiction_score:.2f} ent={p.entailment_score:.2f}\n"
              f"      A ({p.premise.location}): {p.premise.text}\n"
              f"      B ({p.hypothesis.location}): {p.hypothesis.text}")
    return report


def text_vs_doc():
    from nlp import ComparisonConfig, compare_text_to_document
    from utils.file_handler import SAMPLE_CIRCULAR_A, load_sample
    report = compare_text_to_document("Students with 60% attendance are eligible for examinations.",
                                      load_sample(SAMPLE_CIRCULAR_A), ComparisonConfig())
    assert report.pairs, f"no evidence retrieved: {report.messages}"
    return report


def no_evidence():
    from nlp import ComparisonConfig, compare_text_to_document
    from utils.file_handler import SAMPLE_CIRCULAR_A, load_sample
    report = compare_text_to_document("Bananas grow in tropical climates.", load_sample(SAMPLE_CIRCULAR_A),
                                      ComparisonConfig(similarity_threshold=0.6))
    assert not report.pairs and report.messages
    return report.messages[0][:70] + "..."


def exports(payload, label):
    """CSV + PDF must be generated from a real backend result and contain its content."""
    import pymupdf

    from utils.report import build_pdf_report
    from utils.results import build_analysis, to_csv_bytes
    analysis = build_analysis(payload)
    rows = list(csv.DictReader(io.StringIO(to_csv_bytes(analysis).decode("utf-8").lstrip("﻿"))))
    assert len(rows) == len(analysis.rows) > 0, "CSV has no rows"
    pdf = build_pdf_report(analysis)
    assert pdf.startswith(b"%PDF")
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        text = " ".join(page.get_text() for page in doc).replace("\n", " ")
    assert rows[0]["Premise"][:30] in text, "PDF does not contain the compared statements"
    return f"{label}: CSV {len(rows)} rows, PDF {max(1, len(pdf) // 1024)} KB"


if __name__ == "__main__":
    from nlp import compare_statements

    print("CONTRADIX final verification\n" + "=" * 34)
    holder = {}
    check("Document extraction (PDF / DOCX / TXT)", extraction)
    check("Text vs Text on the 4 demo examples", demo_examples)
    check("Document vs Document", lambda: holder.setdefault("dd", doc_vs_doc()) and f"summary={holder['dd'].summary}")
    check("Text vs Document", lambda: holder.setdefault("td", text_vs_doc()) and
          f"top evidence: '{holder['td'].pairs[0].premise.text}' -> {holder['td'].pairs[0].status}")
    check("Unrelated statement -> no evidence message", no_evidence)
    check("Export: Text vs Text", lambda: exports(
        compare_statements("The library is open every Sunday.", "The library remains closed on Sundays."), "text"))
    if "dd" in holder:
        check("Export: Document vs Document", lambda: exports(holder["dd"], "doc-vs-doc"))
    if "td" in holder:
        check("Export: Text vs Document", lambda: exports(holder["td"], "text-vs-doc"))
    print("=" * 34)
    if failures:
        print(f"{failures} check(s) FAILED. CONTRADIX is NOT fully working yet.")
        sys.exit(1)
    print("All checks passed. CONTRADIX is working.")
