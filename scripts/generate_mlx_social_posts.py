#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.generate_social_posts import generation_record, prompt_text, read_jsonl
from shape_of_text.quality import FOUNDER_REWRITE_GLOBAL_AVOID_TERMS

DEFAULT_BAD_PHRASES = FOUNDER_REWRITE_GLOBAL_AVOID_TERMS


def render_generation_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    use_chat_template: bool,
    enable_thinking: bool,
) -> str:
    if not use_chat_template:
        return prompt
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("chat template generation requires tokenizer.apply_chat_template")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt.strip()}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def bad_phrase_token_sequences(tokenizer: Any, phrases: list[str]) -> list[tuple[int, ...]]:
    seen: set[tuple[int, ...]] = set()
    sequences: list[tuple[int, ...]] = []
    for phrase in phrases:
        stripped = phrase.strip()
        if not stripped:
            continue
        variants = {
            stripped,
            stripped.capitalize(),
            stripped.upper(),
            f" {stripped}",
            f" {stripped.capitalize()}",
        }
        for variant in variants:
            ids = tuple(tokenizer.encode(variant, add_special_tokens=False))
            if not ids or ids in seen:
                continue
            seen.add(ids)
            sequences.append(ids)
    return sequences


def make_bad_phrase_processor(sequences: list[tuple[int, ...]]):
    import mlx.core as mx

    filtered = [sequence for sequence in sequences if sequence]

    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        token_list = tokens.tolist()
        for sequence in filtered:
            if len(sequence) == 1:
                logits = logits.at[:, sequence[0]].add(-mx.inf)
                continue
            prefix = sequence[:-1]
            if len(token_list) >= len(prefix) and tuple(token_list[-len(prefix) :]) == prefix:
                logits = logits.at[:, sequence[-1]].add(-mx.inf)
        return logits

    return processor


def make_no_repeat_ngram_processor(ngram_size: int):
    import mlx.core as mx

    if ngram_size <= 0:
        raise ValueError("ngram_size must be positive")

    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        token_list = tokens.tolist()
        prefix_size = ngram_size - 1
        if prefix_size == 0 or len(token_list) < prefix_size:
            return logits

        prefix = tuple(token_list[-prefix_size:])
        blocked: list[int] = []
        for index in range(len(token_list) - ngram_size + 1):
            ngram = tuple(token_list[index : index + ngram_size])
            if ngram[:-1] == prefix:
                blocked.append(ngram[-1])
        if blocked:
            logits = logits.at[:, sorted(set(blocked))].add(-mx.inf)
        return logits

    return processor


def generate_one(model: Any, tokenizer: Any, prompt: str, args: argparse.Namespace) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    rendered_prompt = render_generation_prompt(
        tokenizer,
        prompt,
        use_chat_template=args.use_chat_template,
        enable_thinking=args.enable_thinking,
    )
    sampler = make_sampler(temp=args.temperature, top_p=args.top_p)
    logits_processors = []
    if args.suppress_bad_phrases:
        logits_processors.append(make_bad_phrase_processor(args.bad_phrase_token_sequences))
    if args.no_repeat_ngram_size > 0:
        logits_processors.append(make_no_repeat_ngram_processor(args.no_repeat_ngram_size))
    return generate(
        model,
        tokenizer,
        rendered_prompt,
        max_tokens=args.max_new_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=False,
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate social posts with MLX/MLX-LM")
    parser.add_argument("--model-id", default="mlx-community/gemma-4-12B-it-4bit")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument(
        "--briefs-file",
        type=Path,
        default=Path("configs/founder_rewrite_eval_briefs.jsonl"),
    )
    parser.add_argument("--output-file", type=Path, default=Path("outputs/mlx_adapter_posts.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--suppress-bad-phrases",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    return parser.parse_args()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    from mlx_lm import load

    args = parse_args()
    model, tokenizer = load(args.model_id, adapter_path=args.adapter_path)
    args.bad_phrase_token_sequences = bad_phrase_token_sequences(
        tokenizer, list(DEFAULT_BAD_PHRASES)
    )

    records = []
    for brief in read_jsonl(args.briefs_file):
        prompt = prompt_text(brief)
        completion = generate_one(model, tokenizer, prompt, args)
        records.append(generation_record(brief, completion))
        print(f"generated {brief.get('id')}")

    write_jsonl(args.output_file, records)


if __name__ == "__main__":
    main()
