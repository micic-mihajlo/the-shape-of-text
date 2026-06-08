from scripts.build_curated_social_instructions import build_examples
from shape_of_text.quality import quality_report


def test_curated_social_examples_are_varied_and_pass_quality_gate():
    examples = build_examples(seed=41)

    assert len(examples) >= 100
    assert {example["platform"] for example in examples} == {"LinkedIn", "X"}
    assert len({example["archetype"] for example in examples}) >= 6

    report = quality_report(examples)
    assert report["ok"] is True
