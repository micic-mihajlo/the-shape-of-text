#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex


DEFAULT_IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"
HF_TOKEN_PLACEHOLDER = "${HF_TOKEN}"


def conversion_shell(args: argparse.Namespace) -> str:
    gguf_filename = f"{args.output_name}.{args.quantization}.gguf"
    f16_filename = f"{args.output_name}.f16.gguf"
    return f"""
set -euo pipefail
apt-get update
apt-get install -y git cmake ninja-build build-essential
python -m pip install -U pip
python -m pip install -U "huggingface_hub[hf_xet]" transformers accelerate peft safetensors sentencepiece protobuf
rm -rf /workspace/llama.cpp /workspace/merged-text
git clone --depth 1 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp
python - <<'PY'
import json
import os
from pathlib import Path

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

base_model = {args.base_model!r}
adapter_model = {args.adapter_model!r}
adapter_revision = {args.adapter_revision!r}
output_dir = Path("/workspace/merged-text")
token = os.environ["HF_TOKEN"]

base = AutoModelForImageTextToText.from_pretrained(
    base_model,
    dtype=torch.bfloat16,
    device_map="auto",
    token=token,
)
model = PeftModel.from_pretrained(
    base,
    adapter_model,
    revision=adapter_revision,
    token=token,
)
merged = model.merge_and_unload()

if hasattr(merged, "model") and hasattr(merged.model, "language_model"):
    text_model = merged.model.language_model
    if hasattr(merged, "lm_head"):
        text_model.lm_head = merged.lm_head
    text_model.config.architectures = ["Gemma4ForCausalLM"]
else:
    text_model = merged

output_dir.mkdir(parents=True, exist_ok=True)
text_model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="4GB")
tokenizer = AutoTokenizer.from_pretrained(base_model, token=token, use_fast=True)
tokenizer.save_pretrained(output_dir)

manifest = {{
    "base_model": base_model,
    "adapter_model": adapter_model,
    "adapter_revision": adapter_revision,
    "merged_text_dir": str(output_dir),
}}
(output_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("MERGE_OK")
PY
python -m pip install -r /workspace/llama.cpp/requirements/requirements-convert_legacy_llama.txt
cmake -S /workspace/llama.cpp -B /workspace/llama.cpp/build -DLLAMA_CURL=OFF -G Ninja
cmake --build /workspace/llama.cpp/build --target llama-quantize -j 2
python /workspace/llama.cpp/convert_hf_to_gguf.py /workspace/merged-text \\
  --outfile /workspace/{shlex.quote(f16_filename)} \\
  --outtype f16 \\
  --model-name {shlex.quote(args.output_name)} \\
  --fuse-gate-up-exps
/workspace/llama.cpp/build/bin/llama-quantize \\
  /workspace/{shlex.quote(f16_filename)} \\
  /workspace/{shlex.quote(gguf_filename)} \\
  {shlex.quote(args.quantization)}
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

repo_id = {args.hub_model_id!r}
base_model = {args.base_model!r}
adapter_model = {args.adapter_model!r}
adapter_revision = {args.adapter_revision!r}
quantization = {args.quantization!r}
gguf_path = Path("/workspace") / {gguf_filename!r}
token = os.environ["HF_TOKEN"]

sha = hashlib.sha256(gguf_path.read_bytes()).hexdigest()
manifest = {{
    "base_model": base_model,
    "adapter_model": adapter_model,
    "adapter_revision": adapter_revision,
    "format": "GGUF",
    "quantization": quantization,
    "gguf_file": gguf_path.name,
    "gguf_size_bytes": gguf_path.stat().st_size,
    "gguf_sha256": sha,
}}
Path("/workspace/conversion_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True),
    encoding="utf-8",
)
readme = '''---
license: gemma
language:
- en
base_model: {{base_model}}
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- lm-studio
- text-generation
- social-media
- gemma4
- {{quantization_lower}}
---

# {{repo_id}}

GGUF export for `{{adapter_model}}` merged into `{{base_model}}`.

Adapter revision: `{{adapter_revision}}`

Use `{{gguf_name}}` in LM Studio or llama.cpp.
'''.format(
    adapter_model=adapter_model,
    adapter_revision=adapter_revision,
    base_model=base_model,
    gguf_name=gguf_path.name,
    quantization_lower=quantization.lower(),
    repo_id=repo_id,
)
Path("/workspace/README.md").write_text(readme, encoding="utf-8")

api = HfApi(token=token)
create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
api.upload_file(
    path_or_fileobj=str(gguf_path),
    path_in_repo=gguf_path.name,
    repo_id=repo_id,
    repo_type="model",
    token=token,
)
api.upload_file(
    path_or_fileobj="/workspace/conversion_manifest.json",
    path_in_repo="conversion_manifest.json",
    repo_id=repo_id,
    repo_type="model",
    token=token,
)
api.upload_file(
    path_or_fileobj="/workspace/README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
    token=token,
)
print("GGUF_UPLOAD_OK")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
""".strip()


def build_payload(args: argparse.Namespace) -> dict:
    payload = {
        "operation": "run",
        "args": {
            "image": args.image,
            "command": ["/bin/bash", "-lc", conversion_shell(args)],
            "flavor": args.flavor,
            "timeout": args.timeout,
            "secrets": {"HF_TOKEN": "$HF_TOKEN"},
        },
    }
    if args.detach:
        payload["args"]["detach"] = True
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a GGUF conversion HF Jobs payload")
    parser.add_argument("--base-model", default="google/gemma-4-12B-it")
    parser.add_argument("--adapter-model", required=True)
    parser.add_argument("--adapter-revision", default=None)
    parser.add_argument("--hub-model-id", required=True)
    parser.add_argument("--output-name", default="gemma-4-12b-it-social-post-lora")
    parser.add_argument("--quantization", default="Q4_K_M")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--flavor", default="a100-large")
    parser.add_argument("--timeout", default="4h")
    parser.add_argument("--detach", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_payload(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
