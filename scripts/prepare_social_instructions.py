#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.prepare_social_corpus import normalize_post
except ModuleNotFoundError:  # Allows direct execution as python scripts/...
    from prepare_social_corpus import normalize_post


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if isinstance(record, dict):
                yield record


def read_csv(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def iter_records(paths: list[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        if path.suffix.lower() == ".jsonl":
            yield from read_jsonl(path)
        elif path.suffix.lower() == ".csv":
            yield from read_csv(path)
        else:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield {"text": line.strip()}


def value(record: dict[str, Any], field: str) -> str:
    raw = record.get(field)
    return raw.strip() if isinstance(raw, str) else ""


def build_prompt(record: dict[str, Any], args: argparse.Namespace) -> str:
    explicit_prompt = value(record, args.prompt_field)
    if explicit_prompt:
        return explicit_prompt

    platform = value(record, args.platform_field) or "social media"
    topic = value(record, args.topic_field) or "the provided update"
    audience = value(record, args.audience_field) or "a general professional audience"
    tone = value(record, args.tone_field) or "clear, human, and concise"
    constraints = value(record, args.constraints_field)

    lines = [
        f"Write a {platform} post about {topic}.",
        f"Audience: {audience}.",
        f"Tone: {tone}.",
    ]
    if constraints:
        lines.append(f"Constraints: {constraints}.")
    return "\n".join(lines)


def build_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    seen: set[str] = set()
    examples: list[dict[str, Any]] = []
    for record in iter_records(args.inputs):
        completion = normalize_post(value(record, args.completion_field), keep_urls=args.keep_urls)
        if not (args.min_chars <= len(completion) <= args.max_chars):
            continue

        prompt = build_prompt(record, args).strip()
        if not prompt:
            continue

        dedupe_key = f"{prompt}\n{completion}".casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        examples.append(
            {
                "prompt": prompt,
                "completion": completion,
                "platform": value(record, args.platform_field),
                "topic": value(record, args.topic_field),
                "audience": value(record, args.audience_field),
            }
        )

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    return examples


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )


def write_split(
    examples: list[dict[str, Any]], output_dir: Path, validation_ratio: float
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_count = max(1, int(len(examples) * validation_ratio)) if len(examples) > 1 else 0
    validation = examples[:validation_count]
    train = examples[validation_count:]

    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    return train_path, validation_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare prompt-to-social-post examples")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/social-instructions"))
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--completion-field", default="text")
    parser.add_argument("--platform-field", default="platform")
    parser.add_argument("--topic-field", default="topic")
    parser.add_argument("--audience-field", default="audience")
    parser.add_argument("--tone-field", default="tone")
    parser.add_argument("--constraints-field", default="constraints")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--keep-urls", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = build_examples(args)
    if not examples:
        raise SystemExit("No instruction examples survived filtering")
    train_path, validation_path = write_split(examples, args.output_dir, args.validation_ratio)
    print(f"examples={len(examples)}")
    print(f"train={train_path}")
    print(f"validation={validation_path}")


if __name__ == "__main__":
    main()
