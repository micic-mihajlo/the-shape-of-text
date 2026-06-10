#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


DEFAULT_REPO_URL = "https://github.com/micic-mihajlo/the-shape-of-text.git"
DEFAULT_GIT_REF = "mihajlo/social-style-alignment-framework"
DEFAULT_HUB_MODEL_ID = "micic-mihajlo/gemma-4-12b-it-founder-rewrite-lora"


def env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def shell(command: str, *, cwd: Path | None = None) -> None:
    print("+ " + command, flush=True)
    subprocess.run(["/bin/bash", "-lc", command], cwd=cwd, check=True)


def colab_secret(name: str) -> str | None:
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return None
    try:
        value = userdata.get(name)
    except Exception:
        return None
    return value or None


def ensure_hf_token() -> str:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or colab_secret("HF_TOKEN")
        or colab_secret("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required. In Colab, add it under Secrets as HF_TOKEN "
            "with read access to Gemma and write access to the target adapter repo."
        )
    os.environ["HF_TOKEN"] = token
    return token


def main() -> None:
    token = ensure_hf_token()
    repo_url = env("REPO_URL", DEFAULT_REPO_URL)
    git_ref = env("GIT_REF", DEFAULT_GIT_REF)
    hub_model_id = env("HUB_MODEL_ID", DEFAULT_HUB_MODEL_ID)
    workdir = Path(env("COLAB_WORKDIR", "/content")).resolve()
    repo_dir = workdir / "the-shape-of-text"
    output_dir = workdir / "runs" / "gemma4-founder-rewrite-colab"

    max_steps = env("MAX_STEPS", "220")
    max_length = env("MAX_LENGTH", "768")
    learning_rate = env("LEARNING_RATE", "8e-5")
    lora_r = env("LORA_R", "16")
    gradient_accumulation_steps = env("GRADIENT_ACCUMULATION_STEPS", "8")
    eval_steps = env("EVAL_STEPS", "55")
    save_steps = env("SAVE_STEPS", "55")

    run(["nvidia-smi"])
    shell("python -m pip install -U pip")
    if repo_dir.exists():
        shell(f"rm -rf {shlex.quote(str(repo_dir))}")
    run(["git", "clone", repo_url, str(repo_dir)])
    run(["git", "checkout", git_ref], cwd=repo_dir)
    shell('python -m pip install -e ".[dev]"', cwd=repo_dir)

    run(
        [
            "python",
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

    run(
        [
            "python",
            "-m",
            "shape_of_text.train",
            "--model-id",
            "google/gemma-4-12B-it",
            "--model-class",
            "image-text-to-text",
            "--dataset-format",
            "instruction-jsonl",
            "--train-file",
            "examples/founder_rewrite_instructions/train.jsonl",
            "--eval-file",
            "examples/founder_rewrite_instructions/validation.jsonl",
            "--output-dir",
            str(output_dir),
            "--use-chat-template",
            "--max-length",
            max_length,
            "--max-steps",
            max_steps,
            "--per-device-train-batch-size",
            "1",
            "--per-device-eval-batch-size",
            "1",
            "--gradient-accumulation-steps",
            gradient_accumulation_steps,
            "--learning-rate",
            learning_rate,
            "--eval-steps",
            eval_steps,
            "--save-steps",
            save_steps,
            "--logging-steps",
            "10",
            "--lora-r",
            lora_r,
            "--lora-alpha",
            str(int(lora_r) * 2),
            "--mmd-weight",
            "0.01",
            "--jmq-weight",
            "0.01",
            "--mmd-warmup-steps",
            "30",
            "--jmq-warmup-steps",
            "30",
            "--kl-eval-batches",
            "4",
        ],
        cwd=repo_dir,
    )

    adapter_posts = workdir / "adapter_posts.jsonl"
    quality_report = workdir / "founder_rewrite_quality_report.json"
    run(
        [
            "python",
            "scripts/generate_social_posts.py",
            "--model-id",
            "google/gemma-4-12B-it",
            "--adapter-id",
            str(output_dir),
            "--use-chat-template",
            "--load-in-4bit",
            "--no-enable-thinking",
            "--briefs-file",
            "configs/founder_rewrite_eval_briefs.jsonl",
            "--output-file",
            str(adapter_posts),
        ],
        cwd=repo_dir,
    )
    quality_ok = subprocess.run(
        [
            "python",
            "scripts/check_founder_rewrite_quality.py",
            str(adapter_posts),
            "--output-file",
            str(quality_report),
        ],
        cwd=repo_dir,
    ).returncode == 0

    report_dir = workdir / "adapter_report"
    run(
        [
            "python",
            "scripts/write_adapter_report.py",
            "--adapter-id",
            hub_model_id,
            "--base-model",
            "google/gemma-4-12B-it",
            "--training-method",
            "Colab single-GPU 4-bit QLoRA + MMD/JMQ",
            "--train-file",
            "examples/founder_rewrite_instructions/train.jsonl",
            "--eval-file",
            "examples/founder_rewrite_instructions/validation.jsonl",
            "--generated-file",
            str(adapter_posts),
            "--quality-report",
            str(quality_report),
            "--output-dir",
            str(report_dir),
        ],
        cwd=repo_dir,
    )
    report_files = (
        adapter_posts,
        quality_report,
        report_dir / "README.md",
        report_dir / "eval_summary.json",
    )
    for path in report_files:
        if path.exists():
            run(["cp", str(path), str(output_dir / path.name)])

    run(
        [
            "python",
            "scripts/upload_hf_adapter.py",
            "--repo-id",
            hub_model_id,
            "--folder",
            str(output_dir),
            "--token",
            token,
            "--create-pr",
        ],
        cwd=repo_dir,
    )

    if not quality_ok:
        raise SystemExit("Founder rewrite quality gate failed; uploaded artifacts for inspection.")
    print("COLAB_FOUNDER_REWRITE_TRAINING_OK")


if __name__ == "__main__":
    main()
