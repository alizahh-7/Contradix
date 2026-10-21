"""Demo statement pairs used by the Streamlit Demo Playground and the verification scripts.

These are DEMONSTRATION inputs. "expected" is what a human reader would conclude; it is
shown in the UI as a clearly-labelled expectation and is NEVER used as a model result.
The real pretrained models always produce the displayed prediction, and they may disagree.
"""

DEMO_EXAMPLES = [
    {
        "name": "Contradiction",
        "expected": "Contradiction",
        "text_a": "The library is open every Sunday.",
        "text_b": "The library remains closed on Sundays.",
    },
    {
        "name": "Entailment",
        "expected": "Entailment",
        "text_a": "Every employee must carry an identification card.",
        "text_b": "Employees are required to carry ID cards.",
    },
    {
        "name": "Neutral",
        "expected": "Neutral",
        "text_a": "The examination begins at 10 AM.",
        "text_b": "The university library contains several thousand books.",
    },
    {
        "name": "Policy inconsistency",
        "expected": "Potential policy inconsistency",
        "text_a": "Students require 75% attendance to qualify for examinations.",
        "text_b": "Students with 60% attendance are eligible for examinations.",
    },
]
EXAMPLE_BY_NAME = {ex["name"]: ex for ex in DEMO_EXAMPLES}
