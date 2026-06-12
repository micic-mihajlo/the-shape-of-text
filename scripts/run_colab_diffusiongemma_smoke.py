#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


DEFAULT_REPO_URL = "https://github.com/micic-mihajlo/the-shape-of-text.git"
DEFAULT_GIT_REF = "mihajlo/social-style-alignment-framework"
DEFAULT_GGUF_REPO = "unsloth/diffusiongemma-26B-A4B-it-GGUF"
DEFAULT_LLAMA_CPP_DIFFUSION_REF = "pull/24423/head"
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
    diffusion_ref = env("LLAMA_CPP_DIFFUSION_REF", DEFAULT_LLAMA_CPP_DIFFUSION_REF)
    if diffusion_ref.startswith("pull/"):
        run(["git", "fetch", "origin", f"{diffusion_ref}:diffusiongemma"], cwd=llama_cpp_dir)
        run(["git", "checkout", "diffusiongemma"], cwd=llama_cpp_dir)
    else:
        run(["git", "checkout", diffusion_ref], cwd=llama_cpp_dir)
    build_dir = llama_cpp_dir / "build"
    cuda_flag = "-DGGML_CUDA=ON" if shutil.which("nvcc") else "-DGGML_CUDA=OFF"
    if cuda_flag.endswith("OFF") and env_flag("REQUIRE_CUDA", True):
        raise RuntimeError(
            "CUDA Toolkit was not found. Switch Colab to a GPU runtime and reconnect before "
            "running the DiffusionGemma smoke path."
        )
    cuda_arch = env("CMAKE_CUDA_ARCHITECTURES", "80")
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
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
        ],
        cwd=llama_cpp_dir,
    )
    run(["cmake", "--build", "build", "--target", "llama-diffusion-cli", "-j"], cwd=llama_cpp_dir)
    cli = build_dir / "bin" / "llama-diffusion-cli"
    if not cli.exists():
        cli = build_dir / "examples" / "diffusion" / "llama-diffusion-cli"
    if not cli.exists():
        raise FileNotFoundError("llama-diffusion-cli build output was not found")
    return cli


def default_gguf_filename(gguf_quant: str) -> str:
    return f"diffusiongemma-26B-A4B-it-{gguf_quant}.gguf"


def download_gguf(repo_id: str, filename: str, model_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        run([PYTHON, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or None
    print(f"Downloading GGUF: repo={repo_id} file={filename}", flush=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(model_dir),
        token=token,
    )
    resolved = Path(path).resolve()
    print(f"Downloaded GGUF to {resolved}", flush=True)
    return resolved


def main() -> None:
    workdir = Path(env("COLAB_WORKDIR", "/content")).resolve()
    repo_dir = Path(env("REPO_DIR", str(workdir / "the-shape-of-text"))).resolve()
    llama_cpp_dir = Path(env("LLAMA_CPP_DIR", str(workdir / "llama.cpp"))).resolve()
    model_dir = Path(env("DIFFUSIONGEMMA_MODEL_DIR", str(workdir / "models"))).resolve()
    repo_url = env("REPO_URL", DEFAULT_REPO_URL)
    git_ref = env("GIT_REF", DEFAULT_GIT_REF)
    gguf_repo = env("DIFFUSIONGEMMA_GGUF_REPO", DEFAULT_GGUF_REPO)
    gguf_quant = env("DIFFUSIONGEMMA_GGUF_QUANT", "Q4_K_M")
    gguf_file = env("DIFFUSIONGEMMA_GGUF_FILE", default_gguf_filename(gguf_quant))
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

    cli = ensure_llama_cpp(llama_cpp_dir)
    model_file = download_gguf(gguf_repo, gguf_file, model_dir)
    run(
        [
            PYTHON,
            "scripts/generate_diffusiongemma_llamacpp_cli_posts.py",
            "--cli",
            str(cli),
            "--model-file",
            str(model_file),
            "--briefs-file",
            "configs/founder_rewrite_eval_briefs.jsonl",
            "--output-file",
            str(generated_file),
            "--n-predict",
            env("GENERATION_N_PREDICT", "220"),
            "--n-gpu-layers",
            env("LLAMA_N_GPU_LAYERS", "99"),
            "--temperature",
            env("GENERATION_TEMPERATURE", "0.4"),
            "--top-p",
            env("GENERATION_TOP_P", "0.9"),
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

    print("DIFFUSIONGEMMA_POSTS_JSONL_BEGIN", flush=True)
    print(generated_file.read_text(encoding="utf-8"), flush=True)
    print("DIFFUSIONGEMMA_POSTS_JSONL_END", flush=True)
    print(f"DIFFUSIONGEMMA_QUALITY_REPORT={quality_report}", flush=True)
    if not quality_ok:
        raise SystemExit("DiffusionGemma founder quality gate failed.")
    print("COLAB_DIFFUSIONGEMMA_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
