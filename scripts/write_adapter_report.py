#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_jsonl(path: Path | None) -> int | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def markdown_report(args: argparse.Namespace) -> str:
    style_report = read_json(args.style_report)
    comparison_report = read_json(args.comparison_report)
    quality = read_json(args.quality_report)
    train_count = count_jsonl(args.train_file)
    eval_count = count_jsonl(args.eval_file)
    generated_count = count_jsonl(args.generated_file)

    lines = [
        f"# {args.adapter_id}",
        "",
        "LoRA adapter for general-purpose social media post drafting.",
        "",
        "## Base",
        "",
        f"- Base model: `{args.base_model}`",
        f"- Adapter: `{args.adapter_id}`",
        f"- Training method: `{args.training_method}`",
        "",
        "## Data",
        "",
        f"- Train examples: `{train_count if train_count is not None else 'unknown'}`",
        f"- Validation examples: `{eval_count if eval_count is not None else 'unknown'}`",
        (
            "- Generated eval posts: "
            f"`{generated_count if generated_count is not None else 'unknown'}`"
        ),
        "",
        "## Objective",
        "",
        "The adapter is optimized for brief-to-post generation with supervised cross-entropy plus "
        "distribution-alignment regularizers over token-logit shape.",
        "",
        "## Evaluation",
        "",
    ]

    candidate_metrics = style_report.get("candidate_metrics", {})
    if candidate_metrics:
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        for key, value in sorted(candidate_metrics.items()):
            lines.append(f"| `{key}` | {value:.4f} |")
    else:
        lines.append("No style report was provided.")

    if comparison_report:
        lines.extend(
            [
                "",
                "## Base Comparison",
                "",
                (
                    "- Style distance improvement: "
                    f"`{comparison_report.get('style_distance_improvement', 0.0):.4f}`"
                ),
                (
                    "- Adapter-target distance: "
                    f"`{comparison_report.get('adapter_target_distance', 0.0):.4f}`"
                ),
                (
                    "- Base-target distance: "
                    f"`{comparison_report.get('base_target_distance', 0.0):.4f}`"
                ),
            ]
        )

    if quality:
        lines.extend(
            [
                "",
                "## Generation Quality Gate",
                "",
                f"- Status: `{'pass' if quality.get('ok') else 'fail'}`",
                f"- Failed generations: `{quality.get('failed', 0)}`",
                f"- Total generations: `{quality.get('total', 0)}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            (
                "- This adapter should not be used to imitate a private person, brand, "
                "or protected identity."
            ),
            "- Quality depends on the human-reference corpus and held-out brief coverage.",
            "- Distribution metrics are diagnostic signals, not a substitute for human review.",
            "",
        ]
    )
    return "\n".join(lines)


def summary_json(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "adapter_id": args.adapter_id,
        "base_model": args.base_model,
        "training_method": args.training_method,
        "train_examples": count_jsonl(args.train_file),
        "eval_examples": count_jsonl(args.eval_file),
        "generated_examples": count_jsonl(args.generated_file),
        "style_report": read_json(args.style_report),
        "comparison_report": read_json(args.comparison_report),
        "quality_report": read_json(args.quality_report),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write adapter model-card and eval summary")
    parser.add_argument("--adapter-id", required=True)
    parser.add_argument("--base-model", default="google/gemma-4-12B")
    parser.add_argument("--training-method", default="QLoRA + FSDP + MMD/JMQ")
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--generated-file", type=Path, default=None)
    parser.add_argument("--style-report", type=Path, default=None)
    parser.add_argument("--comparison-report", type=Path, default=None)
    parser.add_argument("--quality-report", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    readme_path = args.output_dir / "README.md"
    summary_path = args.output_dir / "eval_summary.json"
    readme_path.write_text(markdown_report(args), encoding="utf-8")
    summary_path.write_text(
        json.dumps(summary_json(args), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"readme={readme_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
