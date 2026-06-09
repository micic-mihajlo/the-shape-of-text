#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )


def completion_records(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    for index, record in enumerate(records):
        prompt = str(record.get("prompt") or "").strip()
        completion = str(record.get("completion") or "").strip()
        if not prompt or not completion:
            raise ValueError(f"record {index} requires non-empty prompt and completion")
        converted.append({"prompt": prompt, "completion": completion})
    return converted


def convert_dataset(
    *,
    train_file: Path,
    valid_file: Path,
    output_dir: Path,
    test_file: Path | None = None,
) -> dict[str, int]:
    train = completion_records(read_jsonl(train_file))
    valid = completion_records(read_jsonl(valid_file))
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "valid.jsonl", valid)

    counts = {"train": len(train), "valid": len(valid), "test": 0}
    if test_file is not None:
        test = completion_records(read_jsonl(test_file))
        write_jsonl(output_dir / "test.jsonl", test)
        counts["test"] = len(test)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MLX LoRA prompt/completion data")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("examples/founder_rewrite_instructions/train.jsonl"),
    )
    parser.add_argument(
        "--valid-file",
        type=Path,
        default=Path("examples/founder_rewrite_instructions/validation.jsonl"),
    )
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mlx_founder_rewrite_data"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = convert_dataset(
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
    )
    print(f"train={counts['train']}")
    print(f"valid={counts['valid']}")
    if args.test_file is not None:
        print(f"test={counts['test']}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
