"""Phase 1 verification script - run this to check that everything works.

    python verify_phase1.py

It checks: Python version, imports, spaCy model, both pretrained models
(downloads them on first run), the NLI label mapping, and the 3 demo examples.
Exit code 0 = Phase 1 OK, 1 = something failed.
"""
import sys
import traceback

PASS, FAIL = "[PASS]", "[FAIL]"
failures = 0


def check(name, fn):
    global failures
    try:
        detail = fn()
        print(f"{PASS} {name}" + (f" -> {detail}" if detail else ""))
        return True
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"{FAIL} {name}: {exc}")
        traceback.print_exc(limit=2)
        return False


def check_python():
    v = sys.version_info
    if (v.major, v.minor) != (3, 11):
        print(f"[WARN] Python {v.major}.{v.minor} detected; project targets 3.11.")
    return f"{v.major}.{v.minor}.{v.micro}"


def check_spacy():
    from nlp.preprocessing import SPACY_MODEL, analyze_text, get_nlp
    nlp = get_nlp()
    if not nlp.has_pipe("tagger"):
        raise RuntimeError(f"{SPACY_MODEL} not installed. Run: python -m spacy download {SPACY_MODEL}")
    info = analyze_text("The library is not open on Sundays. It closes early.")
    return f"{info['num_sentences']} sentences, negations={info['negations']}"


def check_embedding_model():
    from nlp.similarity import compute_similarity
    return f"sim('cat on mat','a cat sits on a mat') = {compute_similarity('cat on mat', 'a cat sits on a mat'):.3f}"


def check_nli_model():
    from nlp.inference import get_label_mapping
    return f"label mapping from model config = {get_label_mapping()}"


def run_demos():
    from nlp import compare_statements
    from utils.examples import DEMO_EXAMPLES
    print()
    print(f"{'Expected':<14}{'Predicted':<15}{'Sim':>7}{'Ent':>8}{'Con':>8}{'Neu':>8}")
    for ex in DEMO_EXAMPLES:
        r = compare_statements(ex["text_a"], ex["text_b"])
        total = r["entailment_score"] + r["contradiction_score"] + r["neutral_score"]
        assert abs(total - 1.0) < 1e-4, "NLI scores must sum to 1"
        print(f"{ex['expected']:<14}{r['relationship']:<15}{r['similarity']:>7.3f}"
              f"{r['entailment_score']:>8.3f}{r['contradiction_score']:>8.3f}{r['neutral_score']:>8.3f}")
    print("(Demo labels are human expectations, not guarantees - small models can miss on some pairs.)\n")
    return f"{len(DEMO_EXAMPLES)} demo pairs ran"


def check_empty_input():
    from nlp import compare_statements
    try:
        compare_statements("", "hello")
    except ValueError:
        return "empty input rejected with ValueError"
    raise AssertionError("empty input was not rejected")


if __name__ == "__main__":
    print("CONTRADIX Phase 1 verification\n" + "=" * 34)
    check("Python version", check_python)
    check("spaCy preprocessing", check_spacy)
    check("Sentence-Transformers model", check_embedding_model)
    check("NLI model + label mapping", check_nli_model)
    check("Empty-input validation", check_empty_input)
    check("compare_statements on demo pairs", run_demos)
    print("=" * 34)
    if failures:
        print(f"{failures} check(s) FAILED. Phase 1 is NOT complete yet.")
        sys.exit(1)
    print("All checks passed. Phase 1 is working.")
