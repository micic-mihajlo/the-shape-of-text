#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.generate_social_posts import generation_record, prompt_text


SYSTEM_PROMPT = (
    "You rewrite rough founder notes into direct, specific social posts. "
    "Return only the final post."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cli_prompt(brief: dict[str, Any]) -> str:
    return f"{SYSTEM_PROMPT}\n\n{prompt_text(brief).strip()}\n\nFinal post:\n"


def clean_completion(raw: str, prompt: str) -> str:
    text = raw.replace("\r\n", "\n").strip()
    if prompt.strip() in text:
        text = text.rsplit(prompt.strip(), 1)[-1].strip()
    if "Final post:" in text:
        text = text.rsplit("Final post:", 1)[-1].strip()
    return text.strip()


def run_cli(prompt: str, args: argparse.Namespace) -> str:
    command = [
        str(args.cli),
        "-m",
        str(args.model_file),
        "-p",
        prompt,
        "-ngl",
        str(args.n_gpu_layers),
        "-n",
        str(args.n_predict),
    ]
    if args.context_size:
        command.extend(["-c", str(args.context_size)])
    if args.batch_size:
        command.extend(["-b", str(args.batch_size)])
    if args.ubatch_size:
        command.extend(["-ub", str(args.ubatch_size)])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.temperature is not None:
        command.extend(["--temp", str(args.temperature)])
    if args.top_p is not None:
        command.extend(["--top-p", str(args.top_p)])

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=args.request_timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "llama-diffusion-cli failed with exit code "
            f"{result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return clean_completion(result.stdout, prompt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate founder/social eval posts with llama-diffusion-cli."
    )
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument(
        "--briefs-file",
        type=Path,
        default=Path("configs/founder_rewrite_eval_briefs.jsonl"),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("outputs/diffusiongemma_posts.jsonl"),
    )
    parser.add_argument("--n-predict", type=int, default=220)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--context-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--ubatch-size", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--request-timeout", type=float, default=360)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        for brief in read_jsonl(args.briefs_file):
            prompt = cli_prompt(brief)
            completion = run_cli(prompt, args)
            record = generation_record(brief, completion)
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            print(f"generated {record['id']}", flush=True)


if __name__ == "__main__":
    main()
