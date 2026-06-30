#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex


DEFAULT_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
DEFAULT_REPO_URL = "https://github.com/micic-mihajlo/the-shape-of-text.git"
DEFAULT_ADAPTER_ID = "micic-mihajlo/diffusiongemma-social-writer-lora"


def clone_and_infer_shell(args: argparse.Namespace) -> str:
    prompt_flag = ""
    if args.prompt:
        prompt_flag = f" \\\n  --prompt {shlex.quote(args.prompt)}"
    command = f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export HF_HUB_ENABLE_HF_TRANSFER=1
apt-get update
apt-get install -y python3 python3-pip git build-essential libssl-dev
git clone {shlex.quote(args.repo_url)} /workspace/the-shape-of-text
cd /workspace/the-shape-of-text
git checkout {shlex.quote(args.git_ref)}
python3 -m pip install -U pip
python3 -m pip install unsloth peft
python3 -m pip install --no-deps --upgrade --force-reinstall git+https://github.com/unslothai/unsloth-zoo.git git+https://github.com/unslothai/unsloth.git
python3 -m pip install --no-deps transformers==5.11.0 "tokenizers>=0.22.0,<=0.23.0"
python3 -m pip install -e ".[dev]"
python3 scripts/run_hf_diffusiongemma_inference.py \\
  --adapter-id {shlex.quote(args.adapter_id)} \\
  --eval-limit {shlex.quote(str(args.eval_limit))} \\
  --max-denoising-steps {shlex.quote(str(args.max_denoising_steps))} \\
  --max-new-tokens {shlex.quote(str(args.max_new_tokens))} \\
  --min-free-gb {shlex.quote(str(args.min_free_gb))} \\
  --artifact-repo {shlex.quote(args.artifact_repo)} \\
  --artifact-repo-type {shlex.quote(args.artifact_repo_type)} \\
  --artifact-path-prefix {shlex.quote(args.artifact_path_prefix)} \\
  --run-id {shlex.quote(args.run_id)}
""".strip()
    return command + prompt_flag


def build_payload(args: argparse.Namespace) -> dict:
    payload = {
        "operation": "run",
        "args": {
            "image": args.image,
            "command": ["/bin/bash", "-lc", clone_and_infer_shell(args)],
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Hugging Face Jobs payload for remote DiffusionGemma LoRA inference."
    )
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--git-ref", required=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--flavor", default="a100-large")
    parser.add_argument("--timeout", default="90m")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--volume", action="append", default=[])
    parser.add_argument("--adapter-id", default=DEFAULT_ADAPTER_ID)
    parser.add_argument("--eval-limit", type=int, default=10)
    parser.add_argument("--max-denoising-steps", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    parser.add_argument("--artifact-repo", default=DEFAULT_ADAPTER_ID)
    parser.add_argument("--artifact-repo-type", default="model")
    parser.add_argument("--artifact-path-prefix", default="inference-runs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompt", default="")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_payload(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
