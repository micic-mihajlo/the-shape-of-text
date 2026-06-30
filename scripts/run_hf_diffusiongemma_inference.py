#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from shape_of_text.quality import founder_rewrite_quality_report

from scripts.run_hf_diffusiongemma_training import clean_generated_text, eval_prompt, read_jsonl


DEFAULT_BASE_MODEL = "unsloth/diffusiongemma-26B-A4B-it"
DEFAULT_ADAPTER_ID = "micic-mihajlo/diffusiongemma-social-writer-lora"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remote-only DiffusionGemma LoRA inference for founder/social rewrite prompts."
    )
    parser.add_argument("--base-model", default=os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument("--adapter-id", default=os.environ.get("ADAPTER_ID", DEFAULT_ADAPTER_ID))
    parser.add_argument(
        "--eval-briefs-file",
        type=Path,
        default=Path("configs/founder_rewrite_eval_briefs.jsonl"),
    )
    parser.add_argument("--prompt", default=os.environ.get("PROMPT", ""))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diffusiongemma_inference"))
    parser.add_argument("--eval-limit", type=int, default=int(os.environ.get("EVAL_LIMIT", "10")))
    parser.add_argument(
        "--max-denoising-steps",
        type=int,
        default=int(os.environ.get("MAX_DENOISING_STEPS", "32")),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("MAX_NEW_TOKENS", "256")),
    )
    parser.add_argument("--min-free-gb", type=float, default=float(os.environ.get("MIN_FREE_GB", "50")))
    parser.add_argument("--artifact-repo", default=os.environ.get("ARTIFACT_REPO", ""))
    parser.add_argument("--artifact-repo-type", default=os.environ.get("ARTIFACT_REPO_TYPE", "model"))
    parser.add_argument(
        "--artifact-path-prefix",
        default=os.environ.get("ARTIFACT_PATH_PREFIX", "inference-runs"),
    )
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", ""))
    return parser.parse_args()


def rows_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.prompt.strip():
        return [{"id": "manual_prompt", "prompt": args.prompt.strip()}]
    return read_jsonl(args.eval_briefs_file)[: args.eval_limit]


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    import torch
    from peft import PeftModel
    from unsloth import FastModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run this on Colab Pro or Hugging Face Jobs, not locally.")
    free_gb, total_gb = (value / 1e9 for value in torch.cuda.mem_get_info())
    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA device: {device_name} | free={free_gb:.1f}GB total={total_gb:.1f}GB", flush=True)
    if free_gb < args.min_free_gb:
        raise RuntimeError(
            f"{free_gb:.1f}GB free GPU memory is below {args.min_free_gb:.1f}GB. "
            "Use A100 80GB / H100 for this adapter path."
        )

    base_model, processor = FastModel.from_pretrained(
        model_name=args.base_model,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_id)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    device = next(parameter.device for parameter in model.parameters() if parameter.device.type != "meta")
    canvas_len = getattr(model.config, "canvas_length", base_model.config.canvas_length)

    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_denoising_steps = args.max_denoising_steps
    generation_config.max_new_tokens = min(args.max_new_tokens, canvas_len)

    generations = []
    model.eval()
    for row in rows_from_args(args):
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
        record = {
            "id": row.get("id"),
            "prompt": prompt,
            "completion": completion,
            "required_terms": row.get("required_terms", []),
            "avoid_terms": row.get("avoid_terms", []),
        }
        generations.append(record)
        print(json.dumps({"id": row.get("id"), "completion": completion}, ensure_ascii=True), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = args.output_dir / "eval_generations.jsonl"
    with generations_path.open("w", encoding="utf-8") as handle:
        for row in generations:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    quality = founder_rewrite_quality_report(generations)
    quality_path = args.output_dir / "eval_quality_report.json"
    write_json(quality_path, quality)

    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model": args.base_model,
        "adapter_id": args.adapter_id,
        "device": device_name,
        "eval_count": len(generations),
        "max_denoising_steps": args.max_denoising_steps,
        "max_new_tokens": generation_config.max_new_tokens,
        "quality_ok": quality.get("ok"),
    }
    metadata_path = args.output_dir / "run_metadata.json"
    write_json(metadata_path, metadata)

    if args.artifact_repo:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required to upload inference artifacts.")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        prefix = f"{args.artifact_path_prefix.strip('/')}/{run_id}"
        api.create_repo(repo_id=args.artifact_repo, repo_type=args.artifact_repo_type, exist_ok=True)
        for path in (generations_path, quality_path, metadata_path):
            api.upload_file(
                repo_id=args.artifact_repo,
                repo_type=args.artifact_repo_type,
                path_or_fileobj=str(path),
                path_in_repo=f"{prefix}/{path.name}",
                commit_message=f"Upload DiffusionGemma inference artifact {run_id}",
            )
        print(
            f"Uploaded inference artifacts to https://huggingface.co/{args.artifact_repo}/tree/main/{prefix}",
            flush=True,
        )


if __name__ == "__main__":
    main()
