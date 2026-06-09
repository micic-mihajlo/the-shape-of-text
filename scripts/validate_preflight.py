#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


REQUIRED_FILES = [
    "pyproject.toml",
    "src/shape_of_text/losses.py",
    "src/shape_of_text/train.py",
    "src/shape_of_text/evaluation.py",
    "src/shape_of_text/style_metrics.py",
    "scripts/prepare_social_instructions.py",
    "scripts/generate_social_posts.py",
    "scripts/evaluate_social_style.py",
    "scripts/compare_social_outputs.py",
    "scripts/write_adapter_report.py",
    "scripts/upload_hf_adapter.py",
    "configs/accelerate_fsdp_qlora_gemma4_12b.yaml",
    "configs/trainer_fsdp_qlora_gemma4_12b.json",
    "configs/social_eval_briefs.jsonl",
]


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"missing file: {path}")


def validate_jsonl(path: Path, required_fields: set[str], *, min_records: int = 1) -> int:
    require_file(path)
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number} invalid JSON: {exc}") from exc
            missing = required_fields - set(record)
            if missing:
                raise SystemExit(f"{path}:{line_number} missing fields: {sorted(missing)}")
            count += 1
    if count < min_records:
        raise SystemExit(f"{path} has {count} records; expected at least {min_records}")
    return count


def validate_configs(root: Path) -> None:
    with (root / "configs/accelerate_fsdp_qlora_gemma4_12b.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        accelerate = yaml.safe_load(handle)
    layer = accelerate["fsdp_config"]["fsdp_transformer_layer_cls_to_wrap"]
    if layer != "Gemma4UnifiedTextDecoderLayer":
        raise SystemExit(f"unexpected FSDP layer class: {layer}")

    with (root / "configs/trainer_fsdp_qlora_gemma4_12b.json").open(
        "r", encoding="utf-8"
    ) as handle:
        trainer = json.load(handle)
    if not trainer.get("use_orig_params"):
        raise SystemExit("trainer FSDP config must keep use_orig_params=true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate fine-tuning preflight artifacts")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument(
        "--briefs-file",
        type=Path,
        default=Path("configs/social_eval_briefs.jsonl"),
    )
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--min-eval-briefs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    for relative in REQUIRED_FILES:
        require_file(root / relative)
    validate_configs(root)

    briefs_file = args.briefs_file
    if not briefs_file.is_absolute():
        briefs_file = root / briefs_file
    brief_count = validate_jsonl(
        briefs_file,
        {"id", "platform", "audience", "prompt"},
        min_records=args.min_eval_briefs,
    )
    print(f"eval_briefs={brief_count}")

    if args.train_file is not None:
        train_count = validate_jsonl(args.train_file, {"prompt", "completion"})
        print(f"train_examples={train_count}")
    if args.eval_file is not None:
        eval_count = validate_jsonl(args.eval_file, {"prompt", "completion"})
        print(f"eval_examples={eval_count}")
    if args.adapter_dir is not None:
        require_file(args.adapter_dir / "adapter_config.json")
        print(f"adapter_dir={args.adapter_dir}")

    print("preflight=ok")


if __name__ == "__main__":
    main()
