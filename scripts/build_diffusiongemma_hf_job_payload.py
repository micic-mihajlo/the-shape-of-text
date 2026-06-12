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
apt-get install -y python3 python3-pip git build-essential cmake ninja-build libssl-dev
git clone {shlex.quote(args.repo_url)} /workspace/the-shape-of-text
cd /workspace/the-shape-of-text
git checkout {shlex.quote(args.git_ref)}
export COLAB_WORKDIR=/workspace
export REPO_DIR=/workspace/the-shape-of-text
export SKIP_REPO_CLONE=1
export REQUIRE_CUDA=1
export DIFFUSIONGEMMA_GGUF_REPO={shlex.quote(args.gguf_repo)}
export DIFFUSIONGEMMA_GGUF_QUANT={shlex.quote(args.gguf_quant)}
export DIFFUSIONGEMMA_RUN_ID={shlex.quote(args.run_id)}
export DIFFUSIONGEMMA_ARTIFACT_REPO={shlex.quote(args.artifact_repo)}
export DIFFUSIONGEMMA_ARTIFACT_REPO_TYPE={shlex.quote(args.artifact_repo_type)}
export DIFFUSIONGEMMA_ARTIFACT_FALLBACK_REPO_TYPE={shlex.quote(args.artifact_fallback_repo_type)}
export DIFFUSIONGEMMA_ARTIFACT_PATH_PREFIX={shlex.quote(args.artifact_path_prefix)}
export LLAMA_CPP_DIFFUSION_REF={shlex.quote(args.llama_cpp_ref)}
export CMAKE_CUDA_ARCHITECTURES={shlex.quote(args.cuda_arch)}
export GENERATION_N_PREDICT={shlex.quote(str(args.n_predict))}
export GENERATION_MAX_ATTEMPTS={shlex.quote(str(args.max_attempts))}
export GENERATION_TEMPERATURE={shlex.quote(str(args.temperature))}
export GENERATION_TOP_P={shlex.quote(str(args.top_p))}
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
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--artifact-repo", default="")
    parser.add_argument("--artifact-repo-type", default="dataset")
    parser.add_argument("--artifact-fallback-repo-type", default="model")
    parser.add_argument("--artifact-path-prefix", default="runs")
    parser.add_argument(
        "--gguf-repo",
        default="unsloth/diffusiongemma-26B-A4B-it-GGUF",
    )
    parser.add_argument("--gguf-quant", default="Q4_K_M")
    parser.add_argument("--llama-cpp-ref", default="pull/24423/head")
    parser.add_argument("--cuda-arch", default="80")
    parser.add_argument("--n-predict", type=int, default=768)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_payload(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
