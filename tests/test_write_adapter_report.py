import argparse
import json

from scripts.write_adapter_report import markdown_report, summary_json


def test_adapter_report_includes_counts_and_metrics(tmp_path):
    train = tmp_path / "train.jsonl"
    eval_file = tmp_path / "eval.jsonl"
    generated = tmp_path / "generated.jsonl"
    style_report = tmp_path / "style.json"
    comparison_report = tmp_path / "comparison.json"
    quality_report = tmp_path / "quality.json"
    train.write_text('{"prompt":"a","completion":"b"}\n', encoding="utf-8")
    eval_file.write_text('{"prompt":"a","completion":"b"}\n', encoding="utf-8")
    generated.write_text('{"completion":"b"}\n', encoding="utf-8")
    style_report.write_text(
        json.dumps({"candidate_metrics": {"word_count": 12.0}}),
        encoding="utf-8",
    )
    comparison_report.write_text(
        json.dumps(
            {
                "style_distance_improvement": 3.0,
                "adapter_target_distance": 4.0,
                "base_target_distance": 7.0,
            }
        ),
        encoding="utf-8",
    )
    quality_report.write_text(
        json.dumps({"ok": True, "failed": 0, "total": 1}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        adapter_id="micic-mihajlo/adapter",
        base_model="google/gemma-4-12B",
        training_method="QLoRA + FSDP + MMD/JMQ",
        train_file=train,
        eval_file=eval_file,
        generated_file=generated,
        style_report=style_report,
        comparison_report=comparison_report,
        quality_report=quality_report,
    )

    report = markdown_report(args)
    summary = summary_json(args)
    assert "micic-mihajlo/adapter" in report
    assert "`word_count`" in report
    assert "Style distance improvement" in report
    assert "Generation Quality Gate" in report
    assert summary["train_examples"] == 1
    assert summary["style_report"]["candidate_metrics"]["word_count"] == 12.0
    assert summary["comparison_report"]["style_distance_improvement"] == 3.0
    assert summary["quality_report"]["ok"] is True
