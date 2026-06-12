#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex


DEFAULT_IMAGE = "nvidia/cuda:12.5.1-devel-ubuntu22.04"


def clone_and_run_shell(args: argparse.Namespace) -> str:
    return f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3 python3-pip git build-essential cmake ninja-build
git clone {shlex.quote(args.repo_url)} /workspace/the-shape-of-text
cd /workspace/the-shape-of-text
git checkout {shlex.quote(args.git_ref)}
export COLAB_WORKDIR=/workspace
export REPO_DIR=/workspace/the-shape-of-text
export SKIP_REPO_CLONE=1
export REQUIRE_CUDA=1
export DIFFUSIONGEMMA_GGUF_REPO={shlex.quote(args.gguf_repo)}
export DIFFUSIONGEMMA_GGUF_QUANT={shlex.quote(args.gguf_quant)}
python3 scripts/run_colab_diffusiongemma_smoke.py
""".strip()


def build_payload(args: argparse.Namespace) -> dict:
    payload = {
        "operation": "run",
        "args": {
            "image": args.image,
            "command": ["/bin/bash", "-lc", clone_and_run_shell(args)],
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
        description="Build a Hugging Face Jobs payload for remote DiffusionGemma smoke eval."
    )
    parser.add_argument(
        "--repo-url",
        default="https://github.com/micic-mihajlo/the-shape-of-text.git",
    )
    parser.add_argument("--git-ref", required=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--flavor", default="l40sx1")
    parser.add_argument("--timeout", default="2h")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--volume", action="append", default=[])
    parser.add_argument(
        "--gguf-repo",
        default="unsloth/diffusiongemma-26B-A4B-it-GGUF",
    )
    parser.add_argument("--gguf-quant", default="Q4_K_M")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_payload(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
