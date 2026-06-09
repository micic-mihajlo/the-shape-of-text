from scripts.build_founder_rewrite_instructions import build_examples
from shape_of_text.quality import founder_rewrite_quality_report


def test_founder_rewrite_examples_are_targeted_and_pass_quality_gate():
    examples = build_examples(seed=73)

    assert len(examples) >= 100
    assert {example["platform"] for example in examples} == {"LinkedIn"}
    assert {example["style_family"] for example in examples} == {"founder_rewrite"}
    assert all(example["required_terms"] for example in examples)
    assert all("Rough draft:" in example["prompt"] for example in examples)

    report = founder_rewrite_quality_report(examples)
    assert report["ok"] is True
