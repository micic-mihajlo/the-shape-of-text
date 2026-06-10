#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def snapshot_path(args: argparse.Namespace) -> Path:
    if args.source_dir is not None:
        return args.source_dir.expanduser().resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            args.model_id,
            revision=args.revision,
            local_files_only=args.local_files_only,
        )
    ).resolve()


def replace_file(dest: Path, *, force: bool) -> None:
    if not dest.exists() and not dest.is_symlink():
        return
    if not force:
        raise FileExistsError(f"{dest} already exists; pass --force to replace it")
    if dest.is_dir() and not dest.is_symlink():
        raise IsADirectoryError(f"refusing to replace directory: {dest}")
    dest.unlink()


def link_or_copy_file(src: Path, dest: Path, *, link_mode: str, force: bool) -> None:
    replace_file(dest, force=force)
    if link_mode == "copy":
        shutil.copy2(src, dest)
    else:
        dest.symlink_to(src)


def write_patched_config(source_config: Path, dest: Path, *, force: bool) -> None:
    replace_file(dest, force=force)
    config = json.loads(source_config.read_text(encoding="utf-8"))
    config["model_type"] = "gemma4"
    config["architectures"] = ["Gemma4ForConditionalGeneration"]
    dest.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_alias(
    *,
    source_dir: Path,
    output_dir: Path,
    link_mode: str,
    force: bool,
) -> list[str]:
    if not source_dir.exists():
        raise FileNotFoundError(f"source model directory not found: {source_dir}")
    source_config = source_dir / "config.json"
    if not source_config.exists():
        raise FileNotFoundError(f"source config not found: {source_config}")

    output_dir.mkdir(parents=True, exist_ok=True)
    linked: list[str] = []
    for src in sorted(source_dir.iterdir()):
        if not src.is_file():
            continue
        dest = output_dir / src.name
        if src.name == "config.json":
            write_patched_config(src, dest, force=force)
        else:
            link_or_copy_file(src, dest, link_mode=link_mode, force=force)
        linked.append(src.name)

    return linked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an MLX-LM-compatible alias for mlx-community Gemma 4."
    )
    parser.add_argument("--model-id", default="mlx-community/gemma-4-12B-it-4bit")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mlx_gemma4_12b_it_4bit_alias"),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = snapshot_path(args)
    linked = prepare_alias(
        source_dir=source_dir,
        output_dir=args.output_dir,
        link_mode=args.link_mode,
        force=args.force,
    )
    print(f"source_dir={source_dir}")
    print(f"output_dir={args.output_dir}")
    print(f"files={len(linked)}")


if __name__ == "__main__":
    main()
