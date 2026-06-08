import argparse
import json

from scripts.prepare_social_instructions import build_examples, write_split


def test_build_examples_from_metadata_and_explicit_prompt(tmp_path):
    source = tmp_path / "posts.jsonl"
    rows = [
        {
            "topic": "a new release",
            "platform": "LinkedIn",
            "audience": "builders",
            "text": "We shipped a cleaner release today with one practical fix users asked for.",
        },
        {
            "prompt": "Write a short launch post.",
            "text": "Today we launched the small thing that makes the daily workflow faster.",
        },
        {"text": "too short"},
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    args = argparse.Namespace(
        inputs=[source],
        output_dir=tmp_path / "out",
        prompt_field="prompt",
        completion_field="text",
        platform_field="platform",
        topic_field="topic",
        audience_field="audience",
        tone_field="tone",
        constraints_field="constraints",
        validation_ratio=0.5,
        min_chars=30,
        max_chars=200,
        keep_urls=False,
        seed=1,
    )
    examples = build_examples(args)
    assert len(examples) == 2
    assert any("Audience: builders." in example["prompt"] for example in examples)
    assert any(example["prompt"] == "Write a short launch post." for example in examples)

    train_path, validation_path = write_split(examples, tmp_path / "out", 0.5)
    assert json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])["completion"]
    assert json.loads(validation_path.read_text(encoding="utf-8").splitlines()[0])["prompt"]
