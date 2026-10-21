"""Demo playground examples."""
from nlp.preprocessing import validate_text
from utils.examples import DEMO_EXAMPLES, EXAMPLE_BY_NAME


def test_four_required_examples_exact_text():
    pairs = {e["name"]: (e["text_a"], e["text_b"]) for e in DEMO_EXAMPLES}
    assert pairs["Contradiction"] == ("The library is open every Sunday.", "The library remains closed on Sundays.")
    assert pairs["Entailment"] == ("Every employee must carry an identification card.",
                                   "Employees are required to carry ID cards.")
    assert pairs["Neutral"] == ("The examination begins at 10 AM.", "The university library contains several thousand books.")
    assert pairs["Policy inconsistency"] == ("Students require 75% attendance to qualify for examinations.",
                                             "Students with 60% attendance are eligible for examinations.")
    assert len(DEMO_EXAMPLES) == 4 and set(EXAMPLE_BY_NAME) == set(pairs)


def test_examples_are_valid_inputs_and_hold_no_model_results():
    for ex in DEMO_EXAMPLES:
        validate_text(ex["text_a"], "A"); validate_text(ex["text_b"], "B")
        assert set(ex) == {"name", "expected", "text_a", "text_b"}      # no scores / predictions stored
