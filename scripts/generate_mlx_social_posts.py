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

from scripts.generate_social_posts import generation_record, prompt_text, read_jsonl


def render_generation_prompt(tokenizer: Any, prompt: str, *, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("chat template generation requires tokenizer.apply_chat_template")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt.strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_one(model: Any, tokenizer: Any, prompt: str, args: argparse.Namespace) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    rendered_prompt = render_generation_prompt(
        tokenizer, prompt, use_chat_template=args.use_chat_template
    )
    sampler = make_sampler(temp=args.temperature, top_p=args.top_p)
    return generate(
        model,
        tokenizer,
        rendered_prompt,
        max_tokens=args.max_new_tokens,
        sampler=sampler,
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

    records = []
    for brief in read_jsonl(args.briefs_file):
        prompt = prompt_text(brief)
        completion = generate_one(model, tokenizer, prompt, args)
        records.append(generation_record(brief, completion))
        print(f"generated {brief.get('id')}")

    write_jsonl(args.output_file, records)


if __name__ == "__main__":
    main()
