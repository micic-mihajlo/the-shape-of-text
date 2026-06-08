import json

from scripts.validate_preflight import validate_jsonl


def test_validate_jsonl_counts_records(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps({"prompt": "Brief", "completion": "Post"}) + "\n",
        encoding="utf-8",
    )
    assert validate_jsonl(path, {"prompt", "completion"}) == 1
