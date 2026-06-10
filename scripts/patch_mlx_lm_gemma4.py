#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def package_root(package: str) -> Path:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(f"could not find installed package: {package}")
    return Path(spec.origin).resolve().parent


def patch_gemma4_sanitizer(path: Path, *, dry_run: bool) -> bool:
    text = path.read_text(encoding="utf-8")
    if '"vision_embedder",' in text:
        return False

    anchors = (
        '"embed_vision",\n',
        '"multi_modal_projector",\n',
    )
    for anchor in anchors:
        if anchor in text:
            updated = text.replace(anchor, anchor + '                    "vision_embedder",\n', 1)
            if not dry_run:
                path.write_text(updated, encoding="utf-8")
            return True

    raise RuntimeError(
        f"could not find Gemma4 sanitizer skip-list anchor in {path}; inspect MLX-LM manually"
    )


def patch_no_thinking_dataset(path: Path, *, dry_run: bool) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    updated: list[str] = []
    changed = False
    for index, line in enumerate(lines):
        if (
            "return_dict=False)" in line
            and "enable_thinking=False" not in line
        ):
            line = line.replace(
                "return_dict=False)",
                "return_dict=False, enable_thinking=False)",
            )
            changed = True
        updated.append(line)
        if "return_dict=False," not in line or "enable_thinking=False" in line:
            continue
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if "enable_thinking=False" in next_line:
            continue
        indent = line[: len(line) - len(line.lstrip())]
        updated.append(f"{indent}enable_thinking=False,\n")
        changed = True

    if changed and not dry_run:
        path.write_text("".join(updated), encoding="utf-8")
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch MLX-LM for local Gemma 4 LoRA training on Apple Silicon."
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = package_root("mlx_lm")
    gemma4_path = root / "models" / "gemma4.py"
    datasets_path = root / "tuner" / "datasets.py"

    changed_sanitizer = patch_gemma4_sanitizer(gemma4_path, dry_run=args.dry_run)
    changed_dataset = patch_no_thinking_dataset(datasets_path, dry_run=args.dry_run)

    print(f"mlx_lm_root={root}")
    print(f"gemma4_sanitizer_changed={changed_sanitizer}")
    print(f"datasets_no_thinking_changed={changed_dataset}")


if __name__ == "__main__":
    main()
