# DiffusionGemma Remote Runbook

This project now treats DiffusionGemma as a separate backend from the Gemma 4
causal-LM/QLoRA trainer.

## Why This Is A Separate Lane

DiffusionGemma is a discrete text diffusion model, not a standard
token-by-token causal LM. The existing `shape_of_text.train` entrypoint trains
shifted causal-LM labels and applies MMD/JMQ to autoregressive logits. That is
valid for Gemma 4 causal checkpoints, but it is not the right supervised
objective for DiffusionGemma.

For DiffusionGemma, the research-backed training shape is:

- corrupt supervised canvas tokens at sampled diffusion timesteps;
- denoise the whole canvas in parallel;
- keep prompt/prefix positions fixed;
- optimize recovery of clean target tokens across supervised canvas positions;
- evaluate with the same no-slop founder quality gate used by the causal path.

## Repos To Use

- Inference / LM Studio style testing:
  `unsloth/diffusiongemma-26B-A4B-it-GGUF`
- Safetensors / future fine-tuning:
  `unsloth/diffusiongemma-26B-A4B-it`
- Upstream base:
  `google/diffusiongemma-26B-A4B-it`

## Remote Colab Smoke Test

Do not run this on the laptop. Use a Colab GPU runtime.

```bash
%env GIT_REF=mihajlo/social-style-alignment-framework
%env DIFFUSIONGEMMA_GGUF_QUANT=Q4_K_M
!python scripts/run_colab_diffusiongemma_smoke.py
```

The script clones the repo into Colab, builds the DiffusionGemma `llama.cpp`
PR (`pull/24423/head`) with CUDA, downloads the Unsloth GGUF, runs
`llama-diffusion-cli` for each founder rewrite eval brief, and fails if
`scripts/check_founder_rewrite_quality.py` rejects any output.

The standard `llama-server` / OpenAI-compatible runner is intentionally not
used here: current mainline builds fail to load this GGUF with
`unknown model architecture: 'diffusion-gemma'`.

## Hugging Face Jobs Smoke Payload

Generate a payload without touching local GPU:

```bash
python scripts/build_diffusiongemma_hf_job_payload.py \
  --git-ref mihajlo/social-style-alignment-framework \
  --artifact-repo micic-mihajlo/diffusiongemma-social-writing-artifacts \
  --run-id smoke-YYYYMMDD-HHMMSS \
  --detach \
  > outputs/hf-diffusiongemma-smoke-job.json
```

Submit it with the Hugging Face Jobs MCP or CLI only after confirming the budget.
The default flavor is `l40sx1`; the job downloads a large GGUF, so it is useful
for proving runtime behavior but not for long training loops on a small credit
budget.

When `DIFFUSIONGEMMA_ARTIFACT_REPO` is set, the remote runner uploads a Hub
dataset artifact folder containing:

- `diffusiongemma_founder_posts.jsonl`
- `diffusiongemma_founder_quality_report.json`
- `run_metadata.json`
- a dataset card with valid YAML metadata

That makes the remote gate auditable without keeping the Colab/HF job logs open.

## Fine-Tuning Direction

Use the GGUF smoke result as a gate before spending training money. If base
DiffusionGemma already clears the founder quality gate with the right prompt and
sampling settings, treat it as the hackathon model and save the fine-tune budget.

If it fails, the fine-tune path should use an official diffusion-aware trainer,
not `shape_of_text.train`. Current viable routes:

- Unsloth Studio / Unsloth code path if their DiffusionGemma fine-tuning API is
  exposed for the selected Colab GPU.
- NVIDIA NeMo AutoModel DiffusionGemma LoRA if an 8-GPU job is available.

The old MMD/JMQ idea can still be reused, but only after the diffusion SFT loss
exists: compute MMD/JMQ over denoised supervised-position logits or over sampled
post-generation statistics, not over shifted causal-LM logits.
