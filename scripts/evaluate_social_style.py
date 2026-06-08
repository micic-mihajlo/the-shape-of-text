#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from shape_of_text.style_metrics import (
    aggregate_style_metrics,
    style_delta,
    top_repeated_terms,
)


def read_posts(path: Path, field: str) -> list[str]:
    if path.suffix.lower() == ".jsonl":
        posts: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                value = record.get(field)
                if isinstance(value, str):
                    posts.append(value)
        return posts

    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate social-post style metrics")
    parser.add_argument("candidate_file", type=Path)
    parser.add_argument("--candidate-field", default="completion")
    parser.add_argument("--target-file", type=Path, default=None)
    parser.add_argument("--target-field", default="completion")
    parser.add_argument("--top-terms", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = read_posts(args.candidate_file, args.candidate_field)
    report = {
        "candidate_metrics": aggregate_style_metrics(candidates),
        "candidate_top_terms": top_repeated_terms(candidates, limit=args.top_terms),
    }
    if args.target_file is not None:
        targets = read_posts(args.target_file, args.target_field)
        report["target_metrics"] = aggregate_style_metrics(targets)
        report["candidate_minus_target"] = style_delta(candidates, targets)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
