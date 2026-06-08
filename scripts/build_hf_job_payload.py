#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


DEFAULT_IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"
HF_TOKEN_PLACEHOLDER = "${HF_TOKEN}"
GENERATION_EVAL_ARGS = (
    "  --max-new-tokens 140 \\\n"
    "  --repetition-penalty 1.12 \\\n"
    "  --no-repeat-ngram-size 5 \\\n"
)


def shell_join(parts: list[str]) -> str:
    return " ".join(
        HF_TOKEN_PLACEHOLDER if part == HF_TOKEN_PLACEHOLDER else shlex.quote(part)
        for part in parts
        if part
    )


def training_output_dir(args: argparse.Namespace) -> str:
    return f"/workspace/runs/{args.adapter_name}-{args.mode}"


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
    output_dir = training_output_dir(args)

    command = [
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
    ]
    if args.use_chat_template:
        command.append("--use-chat-template")
    if args.trainer_push_to_hub:
        command.extend(
            [
                "--push-to-hub",
                "--hub-model-id",
                args.hub_model_id,
                "--hub-token",
                HF_TOKEN_PLACEHOLDER,
            ]
        )
    return command


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


def hub_upload_access_check_shell() -> str:
    return f"""
python - <<'PY'
import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN is required for adapter upload")

info = HfApi(token=token).whoami()
access = info.get("auth", {{}}).get("accessToken", {{}})
role = access.get("role")
fine_grained = access.get("fineGrained") or {{}}
permissions = set(fine_grained.get("global") or [])
for scoped in fine_grained.get("scoped") or []:
    permissions.update(scoped.get("permissions") or [])

has_write = role == "write" or "repo.write" in permissions
if not has_write:
    raise SystemExit("HF_TOKEN must include repo.write permission for adapter upload")

print("HF_UPLOAD_PERMISSION_OK")
PY
""".strip()


def build_command(args: argparse.Namespace) -> list[str]:
    train_command = shell_join(training_args(args))
    output_dir = training_output_dir(args)
    chat_template_arg = "  --use-chat-template \\\n" if args.use_chat_template else ""
    upload_args = [
        "python",
        "scripts/upload_hf_adapter.py",
        "--repo-id",
        args.hub_model_id,
        "--folder",
        output_dir,
        "--token",
        HF_TOKEN_PLACEHOLDER,
    ]
    if args.hub_create_pr:
        upload_args.append("--create-pr")
    upload_command = shell_join(upload_args)
    shell = f"""
{clone_and_install_shell(args)}
{hub_upload_access_check_shell()}
{train_command}
python scripts/generate_social_posts.py \\
  --model-id {shlex.quote(args.model_id)} \\
{chat_template_arg}\
{GENERATION_EVAL_ARGS}\
  --briefs-file configs/social_eval_briefs.jsonl \\
  --output-file /workspace/base_posts.jsonl
python scripts/generate_social_posts.py \\
  --model-id {shlex.quote(args.model_id)} \\
  --adapter-id {shlex.quote(output_dir)} \\
{chat_template_arg}\
{GENERATION_EVAL_ARGS}\
  --briefs-file configs/social_eval_briefs.jsonl \\
  --output-file /workspace/adapter_posts.jsonl
python scripts/check_generation_quality.py \\
  /workspace/adapter_posts.jsonl \\
  --output-file /workspace/generation_quality_report.json
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
  --quality-report /workspace/generation_quality_report.json \\
  --output-dir /workspace/adapter_report
cp /workspace/adapter_report/README.md {shlex.quote(output_dir)}/README.md
cp /workspace/adapter_report/eval_summary.json {shlex.quote(output_dir)}/eval_summary.json
cp /workspace/adapter_posts.jsonl {shlex.quote(output_dir)}/adapter_posts.jsonl
cp /workspace/style_report.json {shlex.quote(output_dir)}/style_report.json
cp /workspace/comparison_report.json {shlex.quote(output_dir)}/comparison_report.json
cp /workspace/generation_quality_report.json {shlex.quote(output_dir)}/generation_quality_report.json
{upload_command}
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
    parser.add_argument("--model-id", default="google/gemma-4-12B-it")
    parser.add_argument("--adapter-name", default="gemma-4-12b-it-social-post-lora")
    parser.add_argument("--hub-model-id", required=True)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("examples/social_instructions/train.jsonl"),
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=Path("examples/social_instructions/validation.jsonl"),
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
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trainer-push-to-hub", action="store_true")
    parser.add_argument("--hub-create-pr", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_payload(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
