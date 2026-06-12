# The Shape of Text

Distribution style-alignment scaffolding for fine-tuning Gemma-class text
models with PyTorch, QLoRA, bf16, and FSDP, plus a remote-only DiffusionGemma
smoke/evaluation path.

The project is intentionally general-purpose: it optimizes measurable alignment
to human reference text distributions while keeping the model decoupled from any
company brand, private corpus, or protected identity.

## What Is Implemented

- `MMDLoss`: maximum mean discrepancy over sampled per-token logit or probability
  distributions.
- `JMQLoss`: joint moment and quantile matching over entropy, confidence, logit
  moments, covariance, and effective support features.
- `CausalLMAlignmentLoss`: standard shifted causal-LM cross entropy plus MMD and
  JMQ regularizers.
- `kl_from_logits` and `evaluate_kl`: KL(target || student) evaluation against
  either frozen target logits or smoothed empirical human-token targets.
- `shape_of_text.train`: Hugging Face Trainer entrypoint for Gemma-compatible
  QLoRA fine-tuning. The default loader is `AutoModelForImageTextToText`, which
  matches the verified Gemma 4 Hub metadata; pass
  `--model-class causal-lm` for text-only causal-LM checkpoints.
- `scripts/run_colab_diffusiongemma_smoke.py`: remote Colab/HF smoke path for
  Unsloth DiffusionGemma GGUF inference plus the founder no-slop quality gate.
- FSDP/QLoRA launcher configs under `configs/`.

DiffusionGemma is intentionally not routed through `shape_of_text.train`.
It is a discrete diffusion text model, so shifted causal-LM labels are the wrong
training objective. See `docs/diffusiongemma_runbook.md` for the remote-only
runtime and fine-tuning plan.

## Loss Objective

The training loss is:

```text
L = L_ce + lambda_mmd * MMD(logits_student, logits_target)
         + lambda_jmq * JMQ(logits_student, logits_target)
```

If target logits are unavailable, the MMD, JMQ, and KL paths build smoothed
empirical targets from the human labels in the batch. Cached target logits from a
frozen reference model are preferred when hardware allows it, because they expose
more of the target distribution than one-hot labels can.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

On Linux CUDA hosts, `bitsandbytes` is installed by the project dependencies. On
macOS it is skipped by default because 4-bit CUDA quantization is not available.

## Data Format

Use plain text files with one or more human-authored documents:

```text
data/train.txt
data/validation.txt
```

The training script tokenizes, concatenates, and chunks the corpus into
fixed-length causal-LM sequences.

For JSONL/CSV/TXT post exports, prepare train and validation files with:

```bash
python scripts/prepare_social_corpus.py \
  exports/posts.jsonl \
  --field text \
  --output-dir data/social \
  --min-chars 40 \
  --max-chars 2000
```

The prep script normalizes whitespace, optionally removes URLs, dedupes exact
case-insensitive matches, filters very short or very long posts, shuffles with a
fixed seed, and writes `train.txt` plus `validation.txt`.

For brief-to-post training, prepare instruction JSONL:

```bash
python scripts/prepare_social_instructions.py \
  exports/posts.jsonl \
  --completion-field text \
  --output-dir data/social-instructions
```

Then train with `--dataset-format instruction-jsonl`. Prompt tokens remain in
the context but are masked from the supervised cross-entropy loss by default.

## Single Or Multi-GPU Training

Start with a short smoke run before increasing context length or rank:

```bash
accelerate launch \
  --config_file configs/accelerate_fsdp_qlora_gemma4_12b.yaml \
  -m shape_of_text.train \
  --model-id google/gemma-4-12B \
  --train-file data/train.txt \
  --eval-file data/validation.txt \
  --output-dir runs/gemma4-12b-style-alignment \
  --fsdp "full_shard auto_wrap" \
  --fsdp-config configs/trainer_fsdp_qlora_gemma4_12b.json \
  --max-steps 100 \
  --max-length 1024 \
  --mmd-weight 0.03 \
  --jmq-weight 0.03 \
  --mmd-warmup-steps 100 \
  --jmq-warmup-steps 100
```

The default FSDP auto-wrap class is `Gemma4UnifiedTextDecoderLayer`, matching
the text decoder layer exposed by `google/gemma-4-12B` in Transformers 5.10. If
the selected checkpoint exposes a different decoder class, update both FSDP
config files before launching.

The committed Accelerate config uses `num_processes: 1` so the default HF Jobs
smoke payload matches one-GPU flavors such as `l40sx1`. For multi-GPU FSDP
sharding, raise `num_processes` to the GPU count before launching.

## Hardware Notes

The memory-critical switches are already set for the intended QLoRA path:

- 4-bit NF4 quantization with bf16 compute and bf16 quant storage.
- FSDP/QLoRA runs use `adamw_torch`; non-FSDP QLoRA can still use the
  bitsandbytes paged 8-bit optimizer.
- LoRA only on attention and MLP projection modules.
- Gradient checkpointing with non-reentrant checkpointing where supported.
- FSDP `FULL_SHARD`, transformer auto-wrap, sharded state dicts, and
  `use_orig_params=true` for PEFT compatibility.
- Bounded MMD/JMQ token and vocab sampling so distribution losses do not dominate
  activation memory.

For CUDA OOM, reduce in this order: `--max-length`,
`--mmd-vocab-sample-size`, `--jmq-vocab-sample-size`, LoRA rank, then
per-device batch size.

## DiffusionGemma Remote Path

Use this when targeting Unsloth DiffusionGemma instead of Gemma 4 12B. Do not
run it on the laptop; run it in Colab Pro or Hugging Face Jobs.

```bash
GIT_REF=mihajlo/social-style-alignment-framework \
DIFFUSIONGEMMA_GGUF_QUANT=Q4_K_M \
python scripts/run_colab_diffusiongemma_smoke.py
```

That runner builds `llama.cpp` with CUDA, serves
`unsloth/diffusiongemma-26B-A4B-it-GGUF`, generates the founder rewrite eval
set, and fails unless `scripts/check_founder_rewrite_quality.py` accepts every
completion.

To create a Hugging Face Jobs payload for the same remote smoke test:

```bash
python scripts/build_diffusiongemma_hf_job_payload.py \
  --git-ref YOUR_COMMITTED_SHA \
  --artifact-repo micic-mihajlo/diffusiongemma-social-writing-artifacts \
  --run-id smoke-YYYYMMDD-HHMMSS \
  --detach \
  > outputs/hf-diffusiongemma-smoke-job.json
```

If the base DiffusionGemma GGUF clears the founder gate, use it as the hackathon
runtime and avoid spending training credits. With `--artifact-repo`, the remote
job also uploads generated posts, the quality report, and run metadata to a Hub
dataset for later comparison. If the base model fails, fine-tuning must use a
diffusion-aware trainer such as Unsloth's DiffusionGemma path or NeMo AutoModel,
not the causal-LM trainer in this repo.

## Evaluation

At each eval step, the Trainer reports regular eval loss plus:

- `eval_kl_divergence`
- `eval_student_entropy`
- `eval_target_entropy`
- `eval_num_tokens`

Use `--kl-eval-batches` to control evaluation cost.

For generated posts or held-out completions, run deterministic style metrics:

```bash
python scripts/evaluate_social_style.py \
  outputs/generated_posts.jsonl \
  --candidate-field completion \
  --target-file data/social-instructions/validation.jsonl \
  --target-field completion
```

Generate posts from the held-out brief file with a base model or LoRA adapter:

```bash
python scripts/generate_social_posts.py \
  --model-id google/gemma-4-12B-it \
  --use-chat-template \
  --briefs-file configs/social_eval_briefs.jsonl \
  --output-file outputs/base_posts.jsonl

python scripts/generate_social_posts.py \
  --model-id google/gemma-4-12B-it \
  --adapter-id micic-mihajlo/gemma-4-12b-it-social-post-lora \
  --use-chat-template \
  --briefs-file configs/social_eval_briefs.jsonl \
  --output-file outputs/adapter_posts.jsonl
```

Compare base and adapter outputs:

```bash
python scripts/compare_social_outputs.py \
  --adapter-file outputs/adapter_posts.jsonl \
  --base-file outputs/base_posts.jsonl \
  --target-file data/social-instructions/validation.jsonl \
  > outputs/comparison_report.json
```

## Tests

```bash
pytest
```

## Local Smoke Test

This command validates the training stack on a tiny causal-LM checkpoint without
using GPU credits:

```bash
mkdir -p data
printf '%s\n' \
  "Launch note: we shipped a compact update today." \
  "Field report: social posts need a hook, a concrete detail, and a clean ask." \
  > data/smoke_train.txt

python -m shape_of_text.train \
  --model-id sshleifer/tiny-gpt2 \
  --model-class causal-lm \
  --train-file data/smoke_train.txt \
  --output-dir runs/smoke-tiny-gpt2 \
  --max-steps 1 \
  --max-length 16 \
  --no-4bit \
  --no-gradient-checkpointing \
  --no-bf16 \
  --torch-dtype float32 \
  --lora-r 4 \
  --lora-alpha 8 \
  --lora-target-modules c_attn c_proj c_fc \
  --mmd-max-positions 8 \
  --jmq-max-positions 8 \
  --mmd-vocab-sample-size 64 \
  --jmq-vocab-sample-size 64 \
  --mmd-weight 0.01 \
  --jmq-weight 0.01 \
  --mmd-warmup-steps 1 \
  --jmq-warmup-steps 1
```

Current local validation status:

- Unit tests pass with `torch 2.12.0`.
- Tiny one-step LoRA training completes on this Mac, including MMD/JMQ warm-in.
- Tiny validation smoke logs `eval_kl_divergence`, `eval_student_entropy`,
  `eval_target_entropy`, and `eval_num_tokens`.
- Tiny generation smoke writes held-out brief completions and the style metrics
  script produces valid JSON.
- Full Gemma 4 12B FSDP/QLoRA has not been run on this Mac because there is no
  CUDA GPU available.

See `docs/hf_job_runbook.md` for the paid GPU job path that trains, evaluates,
and uploads the LoRA adapter to Hugging Face.

For a preflight check before the paid run:

```bash
python scripts/validate_preflight.py \
  --train-file data/social-instructions/train.jsonl \
  --eval-file data/social-instructions/validation.jsonl
```

For a clean-clone smoke check, use the committed fake examples:

```bash
python scripts/validate_preflight.py \
  --train-file examples/social_instructions/train.jsonl \
  --eval-file examples/social_instructions/validation.jsonl
```

For the hackathon-quality social-writing run, use the original seed corpus under
`examples/hackathon_social_instructions/`. It is generated from generic
product/workflow topics, not scraped private posts or brand-specific data:

```bash
python scripts/build_curated_social_instructions.py \
  --output-dir examples/hackathon_social_instructions

python scripts/validate_preflight.py \
  --train-file examples/hackathon_social_instructions/train.jsonl \
  --eval-file examples/hackathon_social_instructions/validation.jsonl
```

Generated eval posts are blocked from upload unless
`scripts/check_generation_quality.py` passes. The gate rejects prompt echo,
Gemma control-token leakage, code/template output, placeholders, and basic
length failures.

For the stricter founder-rewrite target, use the no-slop rewrite corpus and
held-out rewrite briefs:

```bash
python scripts/build_founder_rewrite_instructions.py \
  --output-dir examples/founder_rewrite_instructions

python scripts/validate_preflight.py \
  --train-file examples/founder_rewrite_instructions/train.jsonl \
  --eval-file examples/founder_rewrite_instructions/validation.jsonl \
  --briefs-file configs/founder_rewrite_eval_briefs.jsonl
```

That profile trains on rough-draft-to-finished-post pairs and carries required
anchor terms through generation. The founder quality gate rejects assistant
prefaces, generic product-update filler, missing anchors, dense single-block
posts, and long posts without any short emphasis line:

```bash
python scripts/check_founder_rewrite_quality.py \
  outputs/adapter_posts.jsonl \
  --output-file outputs/founder_rewrite_quality_report.json
```

On Apple Silicon, the same founder-rewrite corpus can be trained locally with
MLX when Hugging Face Jobs credits are unavailable. This is a local fallback for
producing an on-machine adapter; the CUDA/FSDP path above remains the primary
distributed-training route.

```bash
python -m pip install -e ".[mlx]"

python scripts/prepare_mlx_lora_data.py \
  --train-file examples/founder_rewrite_instructions/train.jsonl \
  --valid-file examples/founder_rewrite_instructions/validation.jsonl \
  --output-dir outputs/mlx_founder_rewrite_data

python -m mlx_lm lora \
  --model mlx-community/gemma-4-12B-it-4bit \
  --train \
  --data outputs/mlx_founder_rewrite_data \
  --adapter-path outputs/mlx_founder_rewrite_lora \
  --mask-prompt \
  --batch-size 1 \
  --grad-accumulation-steps 8 \
  --iters 110 \
  --learning-rate 3e-5 \
  --steps-per-report 10 \
  --steps-per-eval 55 \
  --save-every 55 \
  --num-layers 16 \
  --max-seq-length 1024 \
  --grad-checkpoint

python scripts/generate_mlx_social_posts.py \
  --model-id mlx-community/gemma-4-12B-it-4bit \
  --adapter-path outputs/mlx_founder_rewrite_lora \
  --briefs-file configs/founder_rewrite_eval_briefs.jsonl \
  --output-file outputs/mlx_adapter_posts.jsonl

python scripts/check_founder_rewrite_quality.py \
  outputs/mlx_adapter_posts.jsonl \
  --output-file outputs/mlx_generation_quality_report.json
```

For day-to-day work, prefer Colab Pro or Hugging Face Jobs over local MLX. The
local MLX path is useful for a short systems proof, but it can make the laptop
unusable during training.

Run the founder-rewrite job on Colab Pro by executing the runner below in a
Colab GPU runtime. Put a Hugging Face token in Colab Secrets as `HF_TOKEN` with
Gemma read access and write access to the adapter repo.

```bash
GIT_REF=mihajlo/social-style-alignment-framework \
HUB_MODEL_ID=micic-mihajlo/gemma-4-12b-it-founder-rewrite-lora \
MAX_STEPS=220 \
MAX_LENGTH=768 \
python scripts/run_colab_founder_rewrite_training.py
```

That script clones the pushed repo ref, runs PyTorch 4-bit QLoRA with the
MMD/JMQ alignment loss, generates held-out founder rewrites, fails on the strict
founder quality gate, and uploads the adapter plus reports to Hugging Face as a
model PR.

To generate a Hugging Face Jobs payload:

```bash
python scripts/build_hf_job_payload.py \
  --git-ref YOUR_COMMITTED_SHA \
  --mode preflight \
  --hub-model-id micic-mihajlo/gemma-4-12b-it-social-post-lora \
  --detach

python scripts/build_hf_job_payload.py \
  --git-ref YOUR_COMMITTED_SHA \
  --mode full \
  --train-file examples/hackathon_social_instructions/train.jsonl \
  --eval-file examples/hackathon_social_instructions/validation.jsonl \
  --hub-model-id micic-mihajlo/gemma-4-12b-it-social-post-lora \
  --max-steps 180 \
  --learning-rate 8e-5 \
  --mmd-weight 0.01 \
  --jmq-weight 0.01 \
  --mmd-warmup-steps 30 \
  --jmq-warmup-steps 30 \
  --detach

python scripts/build_hf_job_payload.py \
  --git-ref YOUR_COMMITTED_SHA \
  --mode full \
  --adapter-name gemma-4-12b-it-founder-rewrite-lora \
  --train-file examples/founder_rewrite_instructions/train.jsonl \
  --eval-file examples/founder_rewrite_instructions/validation.jsonl \
  --briefs-file configs/founder_rewrite_eval_briefs.jsonl \
  --quality-script scripts/check_founder_rewrite_quality.py \
  --hub-model-id micic-mihajlo/gemma-4-12b-it-founder-rewrite-lora \
  --max-steps 300 \
  --learning-rate 8e-5 \
  --mmd-weight 0.01 \
  --jmq-weight 0.01 \
  --mmd-warmup-steps 30 \
  --jmq-warmup-steps 30 \
  --detach
```

By default the generated HF Jobs payload uses the committed fake examples under
`examples/social_instructions/` so a clean remote clone can run a systems smoke.
For the hackathon-quality run, pass the committed
`examples/hackathon_social_instructions/` train and validation files.

The GPU payload checks that the `HF_TOKEN` secret has Hub `repo.write` before it
downloads Gemma or starts training. A read-only token can validate downloads but
cannot upload the adapter folder at the end.

The generated GPU job trains and saves the adapter locally, evaluates generated
posts from that saved adapter, then uploads the adapter folder with
`scripts/upload_hf_adapter.py --create-pr`. PR-mode upload is the default
because Hugging Face Jobs tokens may be allowed to open Hub PRs while direct
commits to `main` are forbidden.

After generation and style evaluation, write adapter report artifacts:

```bash
python scripts/write_adapter_report.py \
  --adapter-id micic-mihajlo/gemma-4-12b-it-social-post-lora \
  --base-model google/gemma-4-12B-it \
  --train-file examples/hackathon_social_instructions/train.jsonl \
  --eval-file examples/hackathon_social_instructions/validation.jsonl \
  --generated-file outputs/adapter_posts.jsonl \
  --style-report outputs/style_report.json \
  --comparison-report outputs/comparison_report.json \
  --quality-report outputs/generation_quality_report.json \
  --output-dir outputs/adapter_report
```
