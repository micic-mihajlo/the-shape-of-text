#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from shape_of_text.quality import founder_rewrite_quality_report


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail generic founder-rewrite generations")
    parser.add_argument("generated_file", type=Path)
    parser.add_argument("--completion-field", default="completion")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--min-words", type=int, default=35)
    parser.add_argument("--max-words", type=int, default=260)
    parser.add_argument("--max-prompt-echo-score", type=float, default=0.35)
    parser.add_argument("--max-failure-rate", type=float, default=0.0)
    parser.add_argument("--output-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = founder_rewrite_quality_report(
        read_jsonl(args.generated_file),
        completion_field=args.completion_field,
        prompt_field=args.prompt_field,
        min_words=args.min_words,
        max_words=args.max_words,
        max_prompt_echo_score=args.max_prompt_echo_score,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output_file is not None:
        args.output_file.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["failure_rate"] > args.max_failure_rate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
