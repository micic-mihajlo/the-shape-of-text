import json

from scripts.compare_social_outputs import comparison_report, metric_distance


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_metric_distance_is_absolute_sum():
    assert metric_distance({"a": 3.0, "b": 1.0}, {"a": 1.0, "b": 4.0}) == 5.0


def test_comparison_report_tracks_pairwise_and_distance(tmp_path):
    adapter = tmp_path / "adapter.jsonl"
    base = tmp_path / "base.jsonl"
    target = tmp_path / "target.jsonl"
    write_jsonl(
        adapter,
        [
            {"id": "a", "completion": "We shipped a practical update. Try it today."},
            {"id": "b", "completion": "A cleaner release landed for builders."},
        ],
    )
    write_jsonl(
        base,
        [
            {"id": "a", "completion": "Update update update update update."},
            {"id": "b", "completion": ""},
        ],
    )
    write_jsonl(
        target,
        [
            {"completion": "We shipped a focused update. Try it today."},
            {"completion": "A small release landed with clearer defaults."},
        ],
    )

    report = comparison_report(adapter_file=adapter, base_file=base, target_file=target)
    assert report["pairwise"]["shared_count"] == 2.0
    assert report["pairwise"]["base_empty_rate"] == 0.5
    assert "style_distance_improvement" in report
