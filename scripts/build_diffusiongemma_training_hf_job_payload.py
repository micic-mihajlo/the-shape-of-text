#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex


DEFAULT_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
DEFAULT_REPO_URL = "https://github.com/micic-mihajlo/the-shape-of-text.git"
DEFAULT_HUB_MODEL_ID = "micic-mihajlo/diffusiongemma-social-writer-lora"


def clone_and_train_shell(args: argparse.Namespace) -> str:
    return f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export HF_HUB_ENABLE_HF_TRANSFER=1
export UNSLOTH_RETURN_LOGITS=1
apt-get update
apt-get install -y python3 python3-pip git build-essential libssl-dev
git clone {shlex.quote(args.repo_url)} /workspace/the-shape-of-text
cd /workspace/the-shape-of-text
git checkout {shlex.quote(args.git_ref)}
python3 -m pip install -U pip
python3 -m pip install unsloth
python3 -m pip install --no-deps --upgrade --force-reinstall git+https://github.com/unslothai/unsloth-zoo.git git+https://github.com/unslothai/unsloth.git
python3 -m pip install --no-deps transformers==5.11.0 "tokenizers>=0.22.0,<=0.23.0"
python3 -m pip install -e ".[dev]"
python3 scripts/run_hf_diffusiongemma_training.py \\
  --hub-model-id {shlex.quote(args.hub_model_id)} \\
  --max-steps {shlex.quote(str(args.max_steps))} \\
  --grad-accum {shlex.quote(str(args.grad_accum))} \\
  --lora-r {shlex.quote(str(args.lora_r))} \\
  --lora-alpha {shlex.quote(str(args.lora_alpha))} \\
  --eval-limit {shlex.quote(str(args.eval_limit))} \\
  --min-free-gb {shlex.quote(str(args.min_free_gb))}
""".strip()


def build_payload(args: argparse.Namespace) -> dict:
    payload = {
        "operation": "run",
        "args": {
            "image": args.image,
            "command": ["/bin/bash", "-lc", clone_and_train_shell(args)],
            "flavor": args.flavor,
            "timeout": args.timeout,
            "secrets": {"HF_TOKEN": "$HF_TOKEN"},
        },
    }
    if args.detach:
        payload["args"]["detach"] = True
    if args.volume:
        payload["args"]["volumes"] = args.volume
    return payload


def hf_cli_command(args: argparse.Namespace) -> str:
    payload = build_payload(args)
    command = payload["args"]["command"]
    parts = [
        "hf",
        "jobs",
        "run",
        "--detach" if args.detach else "",
        "--flavor",
        args.flavor,
        "--timeout",
        args.timeout,
        "--secrets",
        "HF_TOKEN",
        args.image,
        *command,
    ]
    return " ".join(shlex.quote(part) for part in parts if part)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Hugging Face Jobs payload for remote DiffusionGemma LoRA training."
    )
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--git-ref", required=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--flavor", default="a100-large")
    parser.add_argument("--timeout", default="4h")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--volume", action="append", default=[])
    parser.add_argument("--hub-model-id", default=DEFAULT_HUB_MODEL_ID)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--eval-limit", type=int, default=10)
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    parser.add_argument("--cli", action="store_true", help="Print an hf CLI command instead of JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cli:
        print(hf_cli_command(args))
    else:
        print(json.dumps(build_payload(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
