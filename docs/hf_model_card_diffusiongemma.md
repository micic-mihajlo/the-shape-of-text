---
license: gemma
base_model: unsloth/diffusiongemma-26B-A4B-it
library_name: peft
pipeline_tag: text-generation
language:
- en
tags:
- diffusiongemma
- block-diffusion
- peft
- lora
- social-writing
- founder-writing
- writing-assistant
model-index:
- name: DiffusionGemma Social Writer LoRA
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      type: custom
      name: Founder rewrite quality gate
    metrics:
    - type: pass_rate
      name: Founder rewrite pass rate
      value: 0.6
    - type: failure_rate
      name: Founder rewrite failure rate
      value: 0.4
---

# DiffusionGemma Social Writer LoRA

This repository contains a PEFT LoRA adapter for
[`unsloth/diffusiongemma-26B-A4B-it`](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it),
trained for concise founder-style social post rewrites.

It is not a merged base model. The adapter is intentionally small and separate
from DiffusionGemma so the base model remains untouched.

## What This Is

The model is part of
[`the-shape-of-text`](https://github.com/micic-mihajlo/the-shape-of-text), a
hackathon research project for style alignment. The core idea is to move beyond
generic instruction-following and train/evaluate against concrete writing
distribution features:

- short, uneven but readable paragraph rhythm;
- preservation of named facts and exact anchors;
- no headings, option lists, placeholders, hashtags, or template scaffolding;
- reduced generic LinkedIn/product-update phrasing;
- clean endings without special-token leaks or repeated tails.

DiffusionGemma is a discrete text diffusion model, so it is trained through a
diffusion-aware denoising objective rather than a standard causal-LM next-token
objective.

## Intended Use

Use this adapter for draft generation and rewriting of short founder/social
posts from a rough brief.

Example target prompt shape:

```text
Rewrite this rough post into one finished founder-style LinkedIn post.
Preserve the point, the concrete facts, and the human rhythm.

Rough draft:
The model finally loads in LM Studio, but the first answer still sounds like a
template. That is not a win. If the local model needs five retries and a perfect
prompt to write a normal post, we did not train it enough. The goal is Gemma
writing well on the first try.

Return only one finished post. No headings, options, hashtags, placeholders, or
analysis. Write 45-130 words in 3-7 short paragraphs with at least one short
standalone line. End cleanly.
```

## Loading

This is a PEFT adapter for DiffusionGemma. Use a DiffusionGemma-capable
Unsloth/Transformers stack and keep the base model separate:

```python
import torch
from peft import PeftModel
from unsloth import FastModel

base_model, processor = FastModel.from_pretrained(
    model_name="unsloth/diffusiongemma-26B-A4B-it",
    dtype=torch.bfloat16,
    load_in_4bit=False,
)

model = PeftModel.from_pretrained(
    base_model,
    "micic-mihajlo/diffusiongemma-social-writer-lora",
)
```

Do not load this with `AutoModelForCausalLM`: DiffusionGemma is not a normal
autoregressive causal language model.

## Training Details

Best live adapter snapshot:

- Hub repo: `micic-mihajlo/diffusiongemma-social-writer-lora`
- Base model: `unsloth/diffusiongemma-26B-A4B-it`
- Hardware: Hugging Face Jobs `a100-large` (`NVIDIA A100-SXM4-80GB`)
- Precision: bf16
- LoRA rank: 32
- LoRA alpha: 64
- Optimizer steps: 300
- Learning rate: `7e-5`
- Canvas length: 256
- Raw founder rewrite examples: 112
- Natural prompt augmentations: 112
- Hard-case examples: 200
- Total encoded training examples: 424
- Skipped examples: 0

Training data lives in the GitHub repository under
`examples/founder_rewrite_instructions/` and
`configs/founder_rewrite_eval_briefs.jsonl`. It is synthetic/generic founder
rewrite data and is intentionally decoupled from any private company corpus,
brand voice, customer data, or protected identity.

## Evaluation

The live adapter currently passes 6 of 10 held-out founder rewrite checks:

```json
{
  "passed": 6,
  "failed": 4,
  "failure_rate": 0.4,
  "total": 10
}
```

The quality gate checks for:

- required anchor preservation;
- forbidden/template phrase avoidance;
- paragraph rhythm and short standalone lines;
- repeated phrase failures;
- non-ASCII/token artifacts;
- clean terminal punctuation;
- prompt/template leakage.

The uploaded files in this repo include:

- `training_metadata.json`
- `train_loss.jsonl`
- `eval_generations.jsonl`
- `eval_quality_report.json`

There is also a separate remote DiffusionGemma GGUF smoke test in the GitHub
repo that passed the same founder rewrite gate 10/10 with a prompt/runtime
configuration. That smoke test is useful evidence for the runtime path, while
this repository is the trained LoRA adapter artifact.

## Known Limitations

This is a hackathon adapter, not a production writing system.

- Exact phrase retention is improved but not solved.
- Some prompts still need a quality gate or retry loop.
- Overweighting hard cases caused repetition/non-ASCII artifacts in later runs,
  so the live repo was restored to the best balanced snapshot.
- Standard LM Studio workflows may not support this LoRA directly unless the
  DiffusionGemma base and adapter are converted through a compatible runtime.

For practical use, run generation behind a small validator that rejects outputs
with missing anchors, repeated phrases, placeholders, or special-token leaks.

## Out-of-Scope Use

This adapter is not intended for impersonation, deceptive authorship claims, or
automated posting without human review. It should be treated as a drafting tool
for generic founder/social writing.

## Citation

```bibtex
@misc{shape_of_text_diffusiongemma_lora_2026,
  title = {DiffusionGemma Social Writer LoRA},
  author = {Mihajlo Micic},
  year = {2026},
  howpublished = {\url{https://huggingface.co/micic-mihajlo/diffusiongemma-social-writer-lora}},
}
```
