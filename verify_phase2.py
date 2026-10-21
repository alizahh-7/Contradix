"""Phase 2 verification - runs the REAL models on the bundled fictional sample documents.

    python verify_phase2.py

Checks: PDF/DOCX/TXT extraction, Text vs Text, Document vs Document, Text vs Document.
First run downloads the models (internet needed). Exit code 0 = OK, 1 = failure.
"""
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
        out.append(f"{d.file_type.upper()}:{d.num_sentences} sentences")
    return ", ".join(out)


def text_vs_text():
    from nlp import compare_statements
    r = compare_statements("The library is open on Sundays.", "The library is closed every Sunday.")
    return f"{r['relationship']} (sim {r['similarity']:.3f}, con {r['contradiction_score']:.3f})"


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
    print("   (Statuses depend on the pretrained model; review them - they are not guaranteed.)")
    return f"summary={report.summary}"


def text_vs_doc():
    from nlp import ComparisonConfig, compare_text_to_document
    from utils.file_handler import SAMPLE_CIRCULAR_A, load_sample
    report = compare_text_to_document("Students with 60% attendance are eligible for examinations.",
                                      load_sample(SAMPLE_CIRCULAR_A), ComparisonConfig())
    assert report.pairs, f"no evidence retrieved: {report.messages}"
    top = report.pairs[0]
    return f"top evidence: '{top.premise.text}' -> {top.status} (con {top.contradiction_score:.2f})"


def no_evidence():
    from nlp import ComparisonConfig, compare_text_to_document
    from utils.file_handler import SAMPLE_CIRCULAR_A, load_sample
    report = compare_text_to_document("Bananas grow in tropical climates.", load_sample(SAMPLE_CIRCULAR_A),
                                      ComparisonConfig(similarity_threshold=0.6))
    assert not report.pairs and report.messages
    return report.messages[0][:70] + "..."


if __name__ == "__main__":
    print("CONTRADIX Phase 2 verification\n" + "=" * 34)
    check("Document extraction (PDF / DOCX / TXT)", extraction)
    check("Text vs Text (Phase 1 preserved)", text_vs_text)
    check("Document vs Document", doc_vs_doc)
    check("Text vs Document", text_vs_doc)
    check("Unrelated statement -> no evidence message", no_evidence)
    print("=" * 34)
    if failures:
        print(f"{failures} check(s) FAILED. Phase 2 is NOT complete yet.")
        sys.exit(1)
    print("All checks passed. Phase 2 is working.")
