# Hackathon Submission Packet

## Project

**The Shape of Text: DiffusionGemma Social Writer LoRA**

Diffusion-aware style alignment for founder/social post rewriting. The project
trains a PEFT LoRA adapter on top of DiffusionGemma and evaluates outputs with a
strict no-template founder rewrite quality gate.

## Submit These Links

- Model: https://huggingface.co/micic-mihajlo/diffusiongemma-social-writer-lora
- Code: https://github.com/micic-mihajlo/the-shape-of-text
- Best adapter evidence:
  https://huggingface.co/micic-mihajlo/diffusiongemma-social-writer-lora/blob/main/eval_quality_report.json
- Generated examples:
  https://huggingface.co/micic-mihajlo/diffusiongemma-social-writer-lora/blob/main/eval_generations.jsonl
- Remote GGUF smoke evidence:
  `docs/evidence/diffusiongemma_a100_smoke_20260612.md`

## Short Description

The Shape of Text fine-tunes DiffusionGemma for concise founder-style social
rewrites. Instead of optimizing for generic instruction-following, it trains and
evaluates against concrete writing-shape constraints: anchor preservation,
short-paragraph rhythm, no headings/options/placeholders, no special-token
leaks, and reduced template language.

## What Was Built

- A diffusion-aware remote training script for `unsloth/diffusiongemma-26B-A4B-it`.
- A PEFT LoRA adapter published to Hugging Face.
- A founder rewrite quality gate with deterministic failure reasons.
- Hugging Face Jobs orchestration for A100 training without running model
  workloads on the laptop.
- A restored best Hub snapshot after later sweeps overfit into repetition.

## Current Results

Live LoRA adapter:

- Base model: `unsloth/diffusiongemma-26B-A4B-it`
- Hardware: Hugging Face Jobs `a100-large`
- LoRA rank: 32
- LoRA alpha: 64
- Steps: 300
- Training examples after augmentation: 424
- Held-out founder rewrite gate: 6/10 passed

Separate remote GGUF smoke path:

- Base/runtime: `unsloth/diffusiongemma-26B-A4B-it-GGUF`
- Founder rewrite gate: 10/10 passed
- Evidence file: `docs/evidence/diffusiongemma_a100_smoke_20260612.md`

## Honest Status

This is a real trained adapter, not just a base model link. It is also not fully
solved. The best live LoRA passes 6/10 of the strict held-out rewrite gate. Later
training sweeps improved some anchors but introduced repeated tails and
non-ASCII artifacts, so the Hub repo was restored to the best balanced snapshot.

The strongest next step is a quality-gated inference wrapper and more targeted
anchor-retention data, not blind extra training.

## Suggested Demo Prompt

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

## Submission Form Copy

**Title:** The Shape of Text: DiffusionGemma Social Writer LoRA

**Tagline:** A diffusion-aware LoRA adapter for writing concise, non-template
founder social posts from rough drafts.

**What makes it interesting:** The project treats writing style as a measurable
output shape rather than a brand voice. It trains DiffusionGemma with a
denoising objective and evaluates with explicit quality checks for anchor
retention, rhythm, template leakage, repetition, and clean endings.

**Limitations:** The adapter is still hackathon-grade. It passes 6/10 of the
strict held-out gate and should be used with a validator/retry loop for serious
drafting.
