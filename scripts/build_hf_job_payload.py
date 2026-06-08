#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


DEFAULT_IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"
HF_TOKEN_PLACEHOLDER = "${HF_TOKEN}"


def shell_join(parts: list[str]) -> str:
    return " ".join(
        HF_TOKEN_PLACEHOLDER if part == HF_TOKEN_PLACEHOLDER else shlex.quote(part)
        for part in parts
        if part
    )


def training_args(args: argparse.Namespace) -> list[str]:
    max_steps = 10 if args.mode == "smoke" else args.max_steps
    max_length = 512 if args.mode == "smoke" else args.max_length
    eval_steps = (
        min(args.eval_steps, max_steps) if args.mode == "full" else max(1, max_steps // 2)
    )
    save_steps = min(args.save_steps, max_steps) if args.mode == "full" else max_steps
    logging_steps = args.logging_steps if args.mode == "full" else 1
    lora_r = min(args.lora_r, 8) if args.mode == "smoke" else args.lora_r
    lora_alpha = min(args.lora_alpha, 16) if args.mode == "smoke" else args.lora_alpha
    output_dir = f"/workspace/runs/{args.adapter_name}-{args.mode}"

    return [
        "accelerate",
        "launch",
        "--config_file",
        "configs/accelerate_fsdp_qlora_gemma4_12b.yaml",
        "-m",
        "shape_of_text.train",
        "--model-id",
        args.model_id,
        "--model-class",
        "image-text-to-text",
        "--dataset-format",
        "instruction-jsonl",
        "--train-file",
        str(args.train_file),
        "--eval-file",
        str(args.eval_file),
        "--output-dir",
        output_dir,
        "--fsdp",
        "full_shard auto_wrap",
        "--fsdp-config",
        "configs/trainer_fsdp_qlora_gemma4_12b.json",
        "--max-length",
        str(max_length),
        "--max-steps",
        str(max_steps),
        "--per-device-train-batch-size",
        "1",
        "--per-device-eval-batch-size",
        "1",
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--eval-steps",
        str(eval_steps),
        "--save-steps",
        str(save_steps),
        "--logging-steps",
        str(logging_steps),
        "--lora-r",
        str(lora_r),
        "--lora-alpha",
        str(lora_alpha),
        "--mmd-weight",
        str(args.mmd_weight),
        "--jmq-weight",
        str(args.jmq_weight),
        "--mmd-warmup-steps",
        str(args.mmd_warmup_steps),
        "--jmq-warmup-steps",
        str(args.jmq_warmup_steps),
        "--kl-eval-batches",
        str(args.kl_eval_batches),
        "--push-to-hub",
        "--hub-model-id",
        args.hub_model_id,
        "--hub-token",
        HF_TOKEN_PLACEHOLDER,
    ]


def clone_and_install_shell(args: argparse.Namespace) -> str:
    return f"""
set -euo pipefail
if ! command -v git >/dev/null 2>&1 || ! command -v gcc >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git build-essential
fi
git clone {shlex.quote(args.repo_url)} /workspace/the-shape-of-text
cd /workspace/the-shape-of-text
git checkout {shlex.quote(args.git_ref)}
python -m pip install -U pip
python -m pip install -e ".[dev]"
""".strip()


def build_preflight_command(args: argparse.Namespace) -> list[str]:
    payload_command = shell_join(
        [
            "python",
            "scripts/build_hf_job_payload.py",
            "--git-ref",
            args.git_ref,
            "--mode",
            "smoke",
            "--train-file",
            str(args.train_file),
            "--eval-file",
            str(args.eval_file),
            "--hub-model-id",
            args.hub_model_id,
            "--detach",
        ]
    )
    shell = f"""
{clone_and_install_shell(args)}
gcc --version >/tmp/gcc_version.txt
python -m shape_of_text.train --help >/tmp/train_help.txt
python scripts/validate_preflight.py \\
  --train-file {shlex.quote(str(args.train_file))} \\
  --eval-file {shlex.quote(str(args.eval_file))}
{payload_command} >/tmp/hf_payload.json
python -m json.tool /tmp/hf_payload.json >/dev/null
python -m pytest -q
printf '\\nREMOTE_CPU_PREFLIGHT_OK\\n'
""".strip()
    return ["/bin/bash", "-lc", shell]


def build_command(args: argparse.Namespace) -> list[str]:
    train_command = shell_join(training_args(args))
    shell = f"""
{clone_and_install_shell(args)}
{train_command}
python scripts/generate_social_posts.py \\
  --model-id {shlex.quote(args.model_id)} \\
  --briefs-file configs/social_eval_briefs.jsonl \\
  --output-file /workspace/base_posts.jsonl
python scripts/generate_social_posts.py \\
  --model-id {shlex.quote(args.model_id)} \\
  --adapter-id {shlex.quote(args.hub_model_id)} \\
  --briefs-file configs/social_eval_briefs.jsonl \\
  --output-file /workspace/adapter_posts.jsonl
python scripts/evaluate_social_style.py \\
  /workspace/adapter_posts.jsonl \\
  --target-file {shlex.quote(str(args.eval_file))} \\
  --target-field completion \\
  > /workspace/style_report.json
python scripts/compare_social_outputs.py \\
  --adapter-file /workspace/adapter_posts.jsonl \\
  --base-file /workspace/base_posts.jsonl \\
  --target-file {shlex.quote(str(args.eval_file))} \\
  > /workspace/comparison_report.json
python scripts/write_adapter_report.py \\
  --adapter-id {shlex.quote(args.hub_model_id)} \\
  --base-model {shlex.quote(args.model_id)} \\
  --train-file {shlex.quote(str(args.train_file))} \\
  --eval-file {shlex.quote(str(args.eval_file))} \\
  --generated-file /workspace/adapter_posts.jsonl \\
  --style-report /workspace/style_report.json \\
  --comparison-report /workspace/comparison_report.json \\
  --output-dir /workspace/adapter_report
""".strip()
    return ["/bin/bash", "-lc", shell]


def build_payload(args: argparse.Namespace) -> dict:
    command = build_preflight_command(args) if args.mode == "preflight" else build_command(args)
    payload = {
        "operation": "run",
        "args": {
            "image": args.image,
            "command": command,
            "flavor": selected_flavor(args),
            "timeout": selected_timeout(args),
        },
    }
    if args.mode != "preflight":
        payload["args"]["secrets"] = {"HF_TOKEN": "$HF_TOKEN"}
    if args.detach:
        payload["args"]["detach"] = True
    if args.volume:
        payload["args"]["volumes"] = args.volume
    return payload


def selected_flavor(args: argparse.Namespace) -> str:
    if args.flavor:
        return args.flavor
    return "cpu-upgrade" if args.mode == "preflight" else "l40sx1"


def selected_timeout(args: argparse.Namespace) -> str:
    if args.timeout:
        return args.timeout
    return "45m" if args.mode == "preflight" else "2h"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Hugging Face Jobs payload")
    parser.add_argument(
        "--repo-url",
        default="https://github.com/micic-mihajlo/the-shape-of-text.git",
    )
    parser.add_argument("--git-ref", required=True)
    parser.add_argument("--mode", choices=("preflight", "smoke", "full"), default="smoke")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--flavor", default=None)
    parser.add_argument("--timeout", default=None)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--volume", action="append", default=[])
    parser.add_argument("--model-id", default="google/gemma-4-12B")
    parser.add_argument("--adapter-name", default="gemma-4-12b-social-post-lora")
    parser.add_argument("--hub-model-id", required=True)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/social-instructions/train.jsonl"),
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=Path("data/social-instructions/validation.jsonl"),
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--mmd-weight", type=float, default=0.03)
    parser.add_argument("--jmq-weight", type=float, default=0.03)
    parser.add_argument("--mmd-warmup-steps", type=int, default=100)
    parser.add_argument("--jmq-warmup-steps", type=int, default=100)
    parser.add_argument("--kl-eval-batches", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_payload(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
