#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import shlex
import subprocess
from pathlib import Path


DEFAULT_REPO_URL = "https://github.com/micic-mihajlo/the-shape-of-text.git"
DEFAULT_GIT_REF = "mihajlo/social-style-alignment-framework"
DEFAULT_HUB_MODEL_ID = "micic-mihajlo/gemma-4-12b-it-founder-rewrite-lora"


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


def ensure_hf_token(*, required: bool) -> str | None:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or colab_secret("HF_TOKEN")
        or colab_secret("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        if not required:
            return None
        raise RuntimeError(
            "HF_TOKEN is required. In Colab, add it under Secrets as HF_TOKEN "
            "with read access to Gemma and write access to the target adapter repo."
        )
    os.environ["HF_TOKEN"] = token
    return token


def selected_dtype() -> str:
    requested = os.environ.get("TORCH_DTYPE")
    if requested in {"bfloat16", "float16"}:
        return requested
    try:
        import torch

        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    except Exception:
        return "float16"


def gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return ""


def download_artifact(path: Path) -> None:
    try:
        from google.colab import files  # type: ignore
    except Exception:
        print(f"Artifact zip written to {path}")
        return
    files.download(str(path))


def main() -> None:
    upload_to_hub = env_flag("UPLOAD_TO_HUB", True)
    token = ensure_hf_token(required=upload_to_hub)
    repo_url = env("REPO_URL", DEFAULT_REPO_URL)
    git_ref = env("GIT_REF", DEFAULT_GIT_REF)
    hub_model_id = env("HUB_MODEL_ID", DEFAULT_HUB_MODEL_ID)
    workdir = Path(env("COLAB_WORKDIR", "/content")).resolve()
    repo_dir = workdir / "the-shape-of-text"

    device_name = gpu_name()
    low_memory_gpu = "T4" in device_name
    model_id = env("MODEL_ID", "google/gemma-4-12B-it")
    target_model_id = env("TARGET_MODEL_ID", "" if low_memory_gpu else model_id)
    output_name = env("OUTPUT_NAME", "gemma4-founder-rewrite-colab")
    output_dir = workdir / "runs" / output_name
    resume_from_checkpoint = env("RESUME_FROM_CHECKPOINT", "")
    dtype = selected_dtype()
    max_steps = env("MAX_STEPS", "260")
    max_length = env("MAX_LENGTH", "512" if low_memory_gpu else "768")
    learning_rate = env("LEARNING_RATE", "4e-5")
    lora_r = env("LORA_R", "16")
    gradient_accumulation_steps = env(
        "GRADIENT_ACCUMULATION_STEPS", "12" if low_memory_gpu else "8"
    )
    eval_steps = env("EVAL_STEPS", "65")
    save_steps = env("SAVE_STEPS", "65")
    mmd_weight = env("MMD_WEIGHT", "0" if not target_model_id else "0.0005")
    jmq_weight = env("JMQ_WEIGHT", "0" if not target_model_id else "0.0005")
    mmd_warmup_steps = env("MMD_WARMUP_STEPS", "180")
    jmq_warmup_steps = env("JMQ_WARMUP_STEPS", "180")
    mmd_vocab_sample_size = env("MMD_VOCAB_SAMPLE_SIZE", "1024")
    jmq_vocab_sample_size = env("JMQ_VOCAB_SAMPLE_SIZE", "2048")
    kl_vocab_sample_size = env("KL_VOCAB_SAMPLE_SIZE", "2048")
    generation_temperature = env("GENERATION_TEMPERATURE", "0")
    generation_top_p = env("GENERATION_TOP_P", "0.85")
    generation_repetition_penalty = env("GENERATION_REPETITION_PENALTY", "1.0")
    generation_no_repeat_ngram_size = env("GENERATION_NO_REPEAT_NGRAM_SIZE", "5")

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
            model_id,
            *(
                [
                    "--target-model-id",
                    target_model_id,
                ]
                if target_model_id
                else []
            ),
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
            *(
                [
                    "--resume-from-checkpoint",
                    resume_from_checkpoint,
                ]
                if resume_from_checkpoint
                else []
            ),
            "--use-chat-template",
            "--max-length",
            max_length,
            "--torch-dtype",
            dtype,
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
            mmd_weight,
            "--jmq-weight",
            jmq_weight,
            "--mmd-warmup-steps",
            mmd_warmup_steps,
            "--jmq-warmup-steps",
            jmq_warmup_steps,
            "--mmd-vocab-sample-size",
            mmd_vocab_sample_size,
            "--jmq-vocab-sample-size",
            jmq_vocab_sample_size,
            "--kl-vocab-sample-size",
            kl_vocab_sample_size,
            "--kl-eval-batches",
            "4",
            "--bf16" if dtype == "bfloat16" else "--no-bf16",
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
            model_id,
            "--adapter-id",
            str(output_dir),
            "--use-chat-template",
            "--load-in-4bit",
            "--torch-dtype",
            dtype,
            "--bnb-4bit-compute-dtype",
            dtype,
            "--no-enable-thinking",
            "--briefs-file",
            "configs/founder_rewrite_eval_briefs.jsonl",
            "--output-file",
            str(adapter_posts),
            "--temperature",
            generation_temperature,
            "--top-p",
            generation_top_p,
            "--repetition-penalty",
            generation_repetition_penalty,
            "--no-repeat-ngram-size",
            generation_no_repeat_ngram_size,
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
            model_id,
            "--training-method",
            "Colab single-GPU 4-bit QLoRA + base-logit MMD/JMQ",
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

    if upload_to_hub:
        if token is None:
            raise RuntimeError("HF_TOKEN is required when UPLOAD_TO_HUB is enabled")
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
    else:
        archive_base = workdir / output_dir.name
        zip_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=output_dir))
        print(f"COLAB_ADAPTER_ZIP={zip_path}", flush=True)
        if env_flag("DOWNLOAD_ARTIFACT", True):
            download_artifact(zip_path)

    if not quality_ok:
        raise SystemExit("Founder rewrite quality gate failed; artifacts were saved for inspection.")
    print("COLAB_FOUNDER_REWRITE_TRAINING_OK")


if __name__ == "__main__":
    main()
