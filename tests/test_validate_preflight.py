import json

from scripts.validate_preflight import validate_jsonl


def test_validate_jsonl_counts_records(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps({"prompt": "Brief", "completion": "Post"}) + "\n",
        encoding="utf-8",
    )
    assert validate_jsonl(path, {"prompt", "completion"}) == 1


def test_validate_jsonl_accepts_rewrite_brief_metadata(tmp_path):
    path = tmp_path / "briefs.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "rewrite",
                "platform": "LinkedIn",
                "audience": "founders",
                "prompt": "Rewrite this.",
                "required_terms": ["Gemma"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert validate_jsonl(path, {"id", "platform", "audience", "prompt"}) == 1
