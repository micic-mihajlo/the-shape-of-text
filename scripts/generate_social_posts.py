#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def model_class(name: str):
    from transformers import AutoModelForCausalLM

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:  # pragma: no cover - depends on installed Transformers version
        AutoModelForImageTextToText = None

    if name == "causal-lm":
        return AutoModelForCausalLM
    if AutoModelForImageTextToText is None:
        raise ImportError(
            "AutoModelForImageTextToText is unavailable. Upgrade transformers "
            "or pass --model-class causal-lm for a causal-LM checkpoint."
        )
    return AutoModelForImageTextToText


def model_dtype(dtype_name: str):
    import torch

    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def load_model(args: argparse.Namespace):
    from peft import PeftModel

    cls = model_class(args.model_class)
    model = cls.from_pretrained(
        args.model_id,
        dtype=model_dtype(args.torch_dtype),
        device_map=None if args.device_map == "none" else args.device_map,
    )
    if args.adapter_id:
        model = PeftModel.from_pretrained(model, args.adapter_id)
    model.eval()
    return model


def prompt_text(brief: dict[str, Any]) -> str:
    platform = brief.get("platform")
    audience = brief.get("audience")
    lines = []
    if platform:
        lines.append(f"Platform: {platform}")
    if audience:
        lines.append(f"Audience: {audience}")
    lines.append(str(brief["prompt"]).strip())
    return "\n".join(lines).strip() + "\n\n"


def generation_prompt(tokenizer: Any, prompt: str, *, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("--use-chat-template requires a tokenizer with apply_chat_template")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt.strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generation_stop_token_ids(tokenizer: Any) -> list[int]:
    ids: list[int] = []
    for token_id in (getattr(tokenizer, "eos_token_id", None),):
        if isinstance(token_id, int) and token_id >= 0:
            ids.append(token_id)

    for name in ("eot_token",):
        value = getattr(tokenizer, name, None)
        if not isinstance(value, str):
            continue
        token_ids = tokenizer.encode(value, add_special_tokens=False)
        if len(token_ids) == 1:
            ids.append(token_ids[0])

    return sorted(set(ids))


def control_token_bad_words(
    tokenizer: Any, *, allowed_token_ids: set[int] | None = None
) -> list[list[int]]:
    allowed_token_ids = allowed_token_ids or set()
    token_values = {
        "<|image|>",
        "<|audio|>",
        "<|video|>",
        "<|tool>",
        "<tool|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|tool_response>",
        "<tool_response|>",
        "<|channel>",
        "<channel|>",
        "<|turn>",
        "<turn|>",
        "<|think|>",
        "<|image>",
        "<image|>",
        "<|audio>",
        "<audio|>",
    }
    for name in (
        "image_token",
        "audio_token",
        "video_token",
        "boi_token",
        "eoi_token",
        "boa_token",
        "eoa_token",
        "soc_token",
        "eoc_token",
        "sot_token",
        "eot_token",
        "stc_token",
        "etc_token",
        "std_token",
        "etd_token",
        "str_token",
        "etr_token",
        "think_token",
    ):
        value = getattr(tokenizer, name, None)
        if isinstance(value, str):
            token_values.add(value)

    bad_words: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for token in sorted(token_values):
        ids = tokenizer.encode(token, add_special_tokens=False)
        if not ids:
            continue
        if len(ids) == 1 and ids[0] in allowed_token_ids:
            continue
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        bad_words.append(list(ids))
    return bad_words


def generate_one(model, tokenizer, prompt: str, args: argparse.Namespace) -> str:
    import torch

    with torch.no_grad():
        rendered_prompt = generation_prompt(
            tokenizer, prompt, use_chat_template=args.use_chat_template
        )
        inputs = tokenizer(
            rendered_prompt,
            return_tensors="pt",
            add_special_tokens=not args.use_chat_template,
        )
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        stop_token_ids = generation_stop_token_ids(tokenizer)
        eos_token_id = (
            stop_token_ids
            if len(stop_token_ids) > 1
            else stop_token_ids[0]
            if stop_token_ids
            else tokenizer.eos_token_id
        )
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "repetition_penalty": args.repetition_penalty,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": eos_token_id,
        }
        if args.no_repeat_ngram_size > 0:
            generation_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p
        if args.suppress_control_tokens:
            generation_kwargs["bad_words_ids"] = control_token_bad_words(
                tokenizer,
                allowed_token_ids=set(stop_token_ids),
            )
        output_ids = model.generate(**inputs, **generation_kwargs)
        generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate social posts from eval briefs")
    parser.add_argument("--model-id", default="google/gemma-4-12B")
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument(
        "--model-class",
        choices=("image-text-to-text", "causal-lm"),
        default="image-text-to-text",
    )
    parser.add_argument("--adapter-id", default=None)
    parser.add_argument(
        "--briefs-file",
        type=Path,
        default=Path("configs/social_eval_briefs.jsonl"),
    )
    parser.add_argument("--output-file", type=Path, default=Path("outputs/generated_posts.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=5)
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument(
        "--suppress-control-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id or args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        for brief in read_jsonl(args.briefs_file):
            prompt = prompt_text(brief)
            completion = generate_one(model, tokenizer, prompt, args)
            record = {
                "id": brief.get("id"),
                "platform": brief.get("platform"),
                "audience": brief.get("audience"),
                "prompt": brief["prompt"],
                "completion": completion,
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            print(f"generated {record['id']}")


if __name__ == "__main__":
    main()
