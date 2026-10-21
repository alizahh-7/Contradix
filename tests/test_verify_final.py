"""verify_final.py is exercised here with the fake backend, so its own logic is tested without downloads."""
import verify_final as vf
from nlp import compare_statements


def test_verify_final_checks_pass_with_fake_backend(fake_backend, capsys):
    assert "sentences" in vf.extraction()
    assert "4 demo pairs" in vf.demo_examples()
    assert vf.no_evidence()
    from nlp import ComparisonConfig, compare_documents, compare_text_to_document
    from utils.file_handler import SAMPLE_CIRCULAR_A, SAMPLE_CIRCULAR_B, load_sample
    cfg = ComparisonConfig(similarity_threshold=0.30)
    dd = compare_documents(load_sample(SAMPLE_CIRCULAR_A), load_sample(SAMPLE_CIRCULAR_B), cfg)
    td = compare_text_to_document("Students with 60% attendance are eligible for examinations.",
                                  load_sample(SAMPLE_CIRCULAR_A), cfg)
    assert vf.exports(dd, "doc-vs-doc").startswith("doc-vs-doc: CSV")
    assert vf.exports(td, "text-vs-doc").startswith("text-vs-doc: CSV")
    assert vf.exports(compare_statements("The library is open every Sunday.",
                                         "The library remains closed on Sundays."), "text")
