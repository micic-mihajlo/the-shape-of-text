#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys


DEFAULT_REPO_URL = "https://github.com/micic-mihajlo/the-shape-of-text.git"
DEFAULT_GIT_REF = "mihajlo/social-style-alignment-framework"
DEFAULT_GGUF_REPO = "unsloth/diffusiongemma-26B-A4B-it-GGUF"
PYTHON = sys.executable


def env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def shell(command: str, *, cwd: Path | None = None) -> None:
    print("+ " + command, flush=True)
    subprocess.run(["/bin/bash", "-lc", command], cwd=cwd, check=True)


def clone_repo(repo_dir: Path, repo_url: str, git_ref: str) -> None:
    if env_flag("SKIP_REPO_CLONE", False):
        return
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    run(["git", "clone", repo_url, str(repo_dir)])
    run(["git", "checkout", git_ref], cwd=repo_dir)


def install_repo(repo_dir: Path) -> None:
    run([PYTHON, "-m", "pip", "install", "-U", "pip"])
    run([PYTHON, "-m", "pip", "install", "-e", ".[dev]"], cwd=repo_dir)


def ensure_llama_cpp(llama_cpp_dir: Path) -> Path:
    if not llama_cpp_dir.exists():
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/ggml-org/llama.cpp.git",
                str(llama_cpp_dir),
            ]
        )
    build_dir = llama_cpp_dir / "build"
    cuda_flag = "-DGGML_CUDA=ON" if shutil.which("nvcc") else "-DGGML_CUDA=OFF"
    if cuda_flag.endswith("OFF") and env_flag("REQUIRE_CUDA", True):
        raise RuntimeError(
            "CUDA Toolkit was not found. Switch Colab to a GPU runtime and reconnect before "
            "running the DiffusionGemma smoke path."
        )
    run(
        [
            "cmake",
            "-S",
            ".",
            "-B",
            "build",
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            cuda_flag,
            "-DGGML_CUDA_ARCHITECTURES=native",
        ],
        cwd=llama_cpp_dir,
    )
    run(["cmake", "--build", "build", "--target", "llama-server", "-j"], cwd=llama_cpp_dir)
    server = build_dir / "bin" / "llama-server"
    if not server.exists():
        server = build_dir / "llama-server"
    if not server.exists():
        raise FileNotFoundError("llama-server build output was not found")
    return server


def launch_server(server: Path, *, gguf_repo: str, gguf_quant: str, port: str) -> subprocess.Popen:
    command = [
        str(server),
        "-hf",
        f"{gguf_repo}:{gguf_quant}",
        "--host",
        "127.0.0.1",
        "--port",
        port,
        "--ctx-size",
        env("LLAMA_CTX_SIZE", "8192"),
        "--parallel",
        env("LLAMA_PARALLEL", "1"),
        "--threads",
        env("LLAMA_THREADS", "8"),
        "--n-gpu-layers",
        env("LLAMA_N_GPU_LAYERS", "999"),
        "--jinja",
    ]
    print("+ " + " ".join(shlex.quote(part) for part in command), flush=True)
    return subprocess.Popen(command)


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=20)


def main() -> None:
    workdir = Path(env("COLAB_WORKDIR", "/content")).resolve()
    repo_dir = Path(env("REPO_DIR", str(workdir / "the-shape-of-text"))).resolve()
    llama_cpp_dir = Path(env("LLAMA_CPP_DIR", str(workdir / "llama.cpp"))).resolve()
    repo_url = env("REPO_URL", DEFAULT_REPO_URL)
    git_ref = env("GIT_REF", DEFAULT_GIT_REF)
    gguf_repo = env("DIFFUSIONGEMMA_GGUF_REPO", DEFAULT_GGUF_REPO)
    gguf_quant = env("DIFFUSIONGEMMA_GGUF_QUANT", "Q4_K_M")
    port = env("LLAMA_SERVER_PORT", "8000")
    generated_file = workdir / "diffusiongemma_founder_posts.jsonl"
    quality_report = workdir / "diffusiongemma_founder_quality_report.json"

    run(["nvidia-smi"])
    shell("apt-get update && apt-get install -y git build-essential cmake ninja-build libssl-dev")
    clone_repo(repo_dir, repo_url, git_ref)
    install_repo(repo_dir)
    run(
        [
            PYTHON,
            "scripts/validate_preflight.py",
            "--train-file",
            "examples/founder_rewrite_instructions/train.jsonl",
            "--eval-file",
            "examples/founder_rewrite_instructions/validation.jsonl",
            "--briefs-file",
            "configs/founder_rewrite_eval_briefs.jsonl",
        ],
        cwd=repo_dir,
    )

    server = ensure_llama_cpp(llama_cpp_dir)
    process = launch_server(server, gguf_repo=gguf_repo, gguf_quant=gguf_quant, port=port)
    try:
        run(
            [
                PYTHON,
                "scripts/generate_diffusiongemma_llamacpp_posts.py",
                "--endpoint",
                f"http://127.0.0.1:{port}/v1/chat/completions",
                "--briefs-file",
                "configs/founder_rewrite_eval_briefs.jsonl",
                "--output-file",
                str(generated_file),
                "--temperature",
                env("GENERATION_TEMPERATURE", "0.4"),
                "--top-p",
                env("GENERATION_TOP_P", "0.9"),
                "--server-timeout",
                env("SERVER_TIMEOUT", "1200"),
            ],
            cwd=repo_dir,
        )
        quality_ok = (
            subprocess.run(
                [
                    PYTHON,
                    "scripts/check_founder_rewrite_quality.py",
                    str(generated_file),
                    "--output-file",
                    str(quality_report),
                ],
                cwd=repo_dir,
            ).returncode
            == 0
        )
    finally:
        terminate(process)

    print("DIFFUSIONGEMMA_POSTS_JSONL_BEGIN", flush=True)
    print(generated_file.read_text(encoding="utf-8"), flush=True)
    print("DIFFUSIONGEMMA_POSTS_JSONL_END", flush=True)
    print(f"DIFFUSIONGEMMA_QUALITY_REPORT={quality_report}", flush=True)
    if not quality_ok:
        raise SystemExit("DiffusionGemma founder quality gate failed.")
    print("COLAB_DIFFUSIONGEMMA_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
