#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Iterable


WHITESPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")


def normalize_post(text: str, *, keep_urls: bool) -> str:
    text = text.strip()
    if not keep_urls:
        text = URL_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def read_text_file(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def read_jsonl_file(path: Path, field: str) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            value = record.get(field)
            if isinstance(value, str):
                yield value


def read_csv_file(path: Path, field: str) -> Iterable[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            value = record.get(field)
            if isinstance(value, str):
                yield value


def iter_posts(paths: list[Path], field: str) -> Iterable[str]:
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            yield from read_jsonl_file(path, field)
        elif suffix == ".csv":
            yield from read_csv_file(path, field)
        else:
            yield from read_text_file(path)


def prepare_posts(args: argparse.Namespace) -> list[str]:
    seen: set[str] = set()
    posts: list[str] = []
    for raw_post in iter_posts(args.inputs, args.field):
        post = normalize_post(raw_post, keep_urls=args.keep_urls)
        if not (args.min_chars <= len(post) <= args.max_chars):
            continue
        dedupe_key = post.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        posts.append(post)

    rng = random.Random(args.seed)
    rng.shuffle(posts)
    return posts


def write_split(posts: list[str], output_dir: Path, validation_ratio: float) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_count = max(1, int(len(posts) * validation_ratio)) if len(posts) > 1 else 0
    validation = posts[:validation_count]
    train = posts[validation_count:]

    train_path = output_dir / "train.txt"
    validation_path = output_dir / "validation.txt"
    train_path.write_text("\n".join(train) + ("\n" if train else ""), encoding="utf-8")
    validation_path.write_text(
        "\n".join(validation) + ("\n" if validation else ""), encoding="utf-8"
    )
    return train_path, validation_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a social-post fine-tuning corpus")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--field", default="text")
    parser.add_argument("--output-dir", type=Path, default=Path("data/social"))
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--keep-urls", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    posts = prepare_posts(args)
    if not posts:
        raise SystemExit("No posts survived filtering")
    train_path, validation_path = write_split(posts, args.output_dir, args.validation_ratio)
    print(f"posts={len(posts)}")
    print(f"train={train_path}")
    print(f"validation={validation_path}")


if __name__ == "__main__":
    main()
