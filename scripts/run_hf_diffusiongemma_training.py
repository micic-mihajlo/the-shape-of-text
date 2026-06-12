#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from shape_of_text.quality import founder_rewrite_quality_report


DEFAULT_BASE_MODEL = "unsloth/diffusiongemma-26B-A4B-it"
DEFAULT_HUB_MODEL_ID = "micic-mihajlo/diffusiongemma-social-writer-lora"


@dataclass(frozen=True)
class TrainingRecord:
    record_id: str
    prompt: str
    completion: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_from_row(row: dict[str, Any], *, fallback_id: str) -> TrainingRecord:
    prompt = str(row.get("prompt", "")).strip()
    completion = str(row.get("completion", "")).strip()
    if not prompt or not completion:
        raise ValueError(f"Training row {fallback_id} must include prompt and completion")
    return TrainingRecord(
        record_id=str(row.get("id") or fallback_id),
        prompt=prompt,
        completion=completion,
    )


def chat_messages(record: TrainingRecord) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": record.prompt},
        {"role": "assistant", "content": record.completion},
    ]


def eval_prompt(row: dict[str, Any]) -> str:
    return str(row.get("prompt", "")).strip()


def clean_generated_text(text: Any, prompt: str) -> str:
    if isinstance(text, (list, tuple)):
        text = "\n".join(str(part) for part in text)
    else:
        text = str(text)
    cleaned = text.replace("\r\n", "\n").strip()
    markers = (
        "<|channel>final",
        "<channel|>",
        "<turn|>",
        "<eos>",
        "<end_of_turn>",
    )
    for marker in markers:
        cleaned = cleaned.replace(marker, "\n")
    if prompt and prompt in cleaned:
        cleaned = cleaned.split(prompt, 1)[-1]
    for marker in ("model\n", "assistant\n", "Final answer:", "Draft:"):
        if marker in cleaned:
            cleaned = cleaned.split(marker)[-1]
    first_newline = cleaned.find("\n")
    first_line = cleaned if first_newline == -1 else cleaned[:first_newline]
    if first_line.strip().casefold() in {"thought", "final", "analysis"}:
        cleaned = cleaned[first_newline + 1 :].strip() if first_newline != -1 else ""
    lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        lowered = stripped.casefold()
        if lowered.startswith(("prompt:", "platform:", "audience:", "analysis:", "thought:")):
            continue
        lines.append(stripped)
    cleaned = "\n".join(lines).strip()
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned


def adapter_card(
    *,
    base_model: str,
    hub_model_id: str,
    train_examples: int,
    skipped_examples: int,
    max_steps: int,
    lora_rank: int,
) -> str:
    return f"""---
license: gemma
base_model: {base_model}
library_name: peft
pipeline_tag: text-generation
language:
- en
tags:
- diffusiongemma
- block-diffusion
- lora
- social-writing
- founder-writing
datasets:
- micic-mihajlo/the-shape-of-text-founder-rewrite-instructions
---

# DiffusionGemma Social Writer LoRA

This is a LoRA adapter for `{base_model}` trained with Unsloth's
DiffusionGemma block-diffusion objective on founder/social rewrite examples
from `the-shape-of-text`.

The adapter is intentionally unmerged. Load it with the DiffusionGemma-capable
Transformers/Unsloth stack and keep the base model separate.

Training summary:

- Hub repo: `{hub_model_id}`
- Train examples used: {train_examples}
- Train examples skipped for 256-token canvas overflow: {skipped_examples}
- Optimizer steps: {max_steps}
- LoRA rank: {lora_rank}
- Precision: bf16
- Target task: concise founder-style social post rewrites with no headings,
  options, placeholders, or generic template language.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remote-only Unsloth DiffusionGemma LoRA training for social rewrites."
    )
    parser.add_argument("--base-model", default=os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("examples/founder_rewrite_instructions/train.jsonl"),
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("examples/founder_rewrite_instructions/validation.jsonl"),
    )
    parser.add_argument(
        "--eval-briefs-file",
        type=Path,
        default=Path("configs/founder_rewrite_eval_briefs.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diffusiongemma_lora"))
    parser.add_argument("--hub-model-id", default=os.environ.get("HUB_MODEL_ID", DEFAULT_HUB_MODEL_ID))
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "160")))
    parser.add_argument("--grad-accum", type=int, default=int(os.environ.get("GRAD_ACCUM", "4")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("LR", "1e-4")))
    parser.add_argument("--lora-r", type=int, default=int(os.environ.get("LORA_R", "32")))
    parser.add_argument("--lora-alpha", type=int, default=int(os.environ.get("LORA_ALPHA", "64")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "13")))
    parser.add_argument("--t-low", type=float, default=float(os.environ.get("T_LOW", "0.1")))
    parser.add_argument("--min-free-gb", type=float, default=float(os.environ.get("MIN_FREE_GB", "50")))
    parser.add_argument("--eval-limit", type=int, default=int(os.environ.get("EVAL_LIMIT", "10")))
    parser.add_argument(
        "--max-denoising-steps",
        type=int,
        default=int(os.environ.get("MAX_DENOISING_STEPS", "32")),
    )
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", "256")))
    parser.add_argument("--min-train-examples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    import torch
    import torch.nn.functional as F
    from huggingface_hub import HfApi
    from unsloth import FastModel

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. This script is for remote A100/H100 jobs, not laptops.")
    free_gb, total_gb = (value / 1e9 for value in torch.cuda.mem_get_info())
    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA device: {device_name} | free={free_gb:.1f}GB total={total_gb:.1f}GB", flush=True)
    if free_gb < args.min_free_gb:
        raise RuntimeError(
            f"{free_gb:.1f}GB free GPU memory is below {args.min_free_gb:.1f}GB. "
            "Use A100 80GB / H100 for real DiffusionGemma training."
        )

    model, processor = FastModel.from_pretrained(
        model_name=args.base_model,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    vocab_size = model.config.text_config.vocab_size
    canvas_len = model.config.canvas_length
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id or 1
    if isinstance(eos, (list, tuple)):
        eos = eos[0]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_gradient_checkpointing=False,
    )
    model.config.use_cache = True
    model.train()

    def encode_training_rows(rows: list[dict[str, Any]]) -> tuple[list[tuple[Any, Any, Any, str]], int]:
        examples = []
        skipped = 0
        for index, row in enumerate(rows):
            record = record_from_row(row, fallback_id=f"row-{index}")
            prompt_ids = processor.apply_chat_template(
                [chat_messages(record)[0]],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )[0].cpu()
            content = tokenizer.encode(record.completion, add_special_tokens=False) + [int(eos)]
            if len(content) > canvas_len:
                skipped += 1
                continue
            x0 = torch.tensor(content + [int(pad)] * (canvas_len - len(content)), dtype=torch.long)
            mask = torch.zeros(canvas_len, dtype=torch.bool)
            mask[: len(content)] = True
            examples.append((prompt_ids, x0, mask, record.record_id))
        return examples, skipped

    train_examples, skipped_train = encode_training_rows(read_jsonl(args.train_file))
    val_examples, skipped_val = encode_training_rows(read_jsonl(args.validation_file))
    if len(train_examples) < args.min_train_examples:
        raise RuntimeError(
            f"Only {len(train_examples)} train examples fit the {canvas_len}-token canvas; "
            f"minimum is {args.min_train_examples}."
        )
    print(
        "Encoded examples: "
        f"train={len(train_examples)} skipped_train={skipped_train} "
        f"val={len(val_examples)} skipped_val={skipped_val} canvas={canvas_len}",
        flush=True,
    )

    device = next(parameter.device for parameter in model.parameters() if parameter.device.type != "meta")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=args.max_steps,
        pct_start=0.03,
        anneal_strategy="cos",
    )

    def corrupt(x0: Any) -> Any:
        t = random.uniform(args.t_low, 1.0)
        xt = x0.to(device).clone()
        selected = torch.rand(canvas_len, device=device) < t
        noise = torch.randint(0, vocab_size, (canvas_len,), device=device)
        xt[selected] = noise[selected]
        return xt.unsqueeze(0)

    def diffusion_loss(example: tuple[Any, Any, Any, str]) -> Any:
        prompt_ids, x0, loss_mask, _record_id = example
        output = model(
            input_ids=prompt_ids.unsqueeze(0).to(device),
            canvas_ids=corrupt(x0),
            self_conditioning_logits=None,
        )
        logits = output.logits[0].float()
        active = loss_mask.to(device)
        return F.cross_entropy(logits[active], x0.to(device)[active])

    @torch.no_grad()
    def validation_loss(limit: int = 8) -> float | None:
        if not val_examples:
            return None
        model.eval()
        losses = []
        for example in val_examples[:limit]:
            losses.append(float(diffusion_loss(example).detach().cpu()))
        model.train()
        return sum(losses) / len(losses)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = args.output_dir / "train_loss.jsonl"
    if train_log_path.exists():
        train_log_path.unlink()

    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.max_steps + 1):
        step_loss = 0.0
        for _ in range(args.grad_accum):
            loss = diffusion_loss(random.choice(train_examples)) / args.grad_accum
            loss.backward()
            step_loss += float(loss.detach().cpu())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        row = {
            "step": step,
            "train_loss": step_loss,
            "grad_norm": float(grad_norm.detach().cpu()),
            "lr": scheduler.get_last_lr()[0],
        }
        if step == 1 or step % 20 == 0 or step == args.max_steps:
            row["validation_loss"] = validation_loss()
            print(json.dumps(row, sort_keys=True), flush=True)
        with train_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print("Saving adapter...", flush=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model": args.base_model,
        "hub_model_id": args.hub_model_id,
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "train_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "skipped_train_examples": skipped_train,
        "skipped_validation_examples": skipped_val,
        "max_steps": args.max_steps,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "canvas_length": canvas_len,
        "device": device_name,
        "bf16": True,
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        adapter_card(
            base_model=args.base_model,
            hub_model_id=args.hub_model_id,
            train_examples=len(train_examples),
            skipped_examples=skipped_train,
            max_steps=args.max_steps,
            lora_rank=args.lora_r,
        ),
        encoding="utf-8",
    )

    api = None
    if not args.no_push:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required to push the trained adapter.")
        api = HfApi(token=token)
        api.create_repo(repo_id=args.hub_model_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=args.hub_model_id,
            repo_type="model",
            folder_path=str(args.output_dir),
            commit_message="Upload trained DiffusionGemma social writer LoRA",
        )
        print(f"Uploaded trained adapter to https://huggingface.co/{args.hub_model_id}", flush=True)

    eval_rows = read_jsonl(args.eval_briefs_file)[: args.eval_limit]
    generations = []
    model.eval()
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_denoising_steps = args.max_denoising_steps
    generation_config.max_new_tokens = min(args.max_new_tokens, canvas_len)
    for row in eval_rows:
        prompt = eval_prompt(row)
        prompt_ids = processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=prompt_ids.to(device),
                generation_config=generation_config,
            )
        decoded = tokenizer.decode(output_ids[0].detach().cpu().tolist(), skip_special_tokens=True)
        completion = clean_generated_text(decoded, prompt)
        generations.append(
            {
                "id": row.get("id"),
                "prompt": prompt,
                "completion": completion,
                "required_terms": row.get("required_terms", []),
                "avoid_terms": row.get("avoid_terms", []),
            }
        )
        print(json.dumps({"eval_id": row.get("id"), "completion": completion[:200]}), flush=True)

    generations_path = args.output_dir / "eval_generations.jsonl"
    with generations_path.open("w", encoding="utf-8") as handle:
        for row in generations:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    quality = founder_rewrite_quality_report(generations)
    (args.output_dir / "eval_quality_report.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"quality": quality}, sort_keys=True), flush=True)

    if api is not None:
        api.upload_folder(
            repo_id=args.hub_model_id,
            repo_type="model",
            folder_path=str(args.output_dir),
            commit_message="Upload DiffusionGemma social writer eval artifacts",
        )
        print(f"Uploaded eval artifacts to https://huggingface.co/{args.hub_model_id}", flush=True)


if __name__ == "__main__":
    main()
