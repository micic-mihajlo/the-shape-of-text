# Hugging Face Job Runbook

Use this after the repo changes are committed and pushed. The job should consume
prepared `instruction-jsonl` files and publish a LoRA adapter, not a merged base
model.

## Inputs

- Smoke/preflight: committed fake examples under `examples/social_instructions/`
- Quality run: committed original examples under
  `examples/hackathon_social_instructions/`, or prepared files such as
  `data/social-instructions/train.jsonl` and
  `data/social-instructions/validation.jsonl`
- Hugging Face token with access to `google/gemma-4-12B-it`
- Target adapter repo, for example:
  `micic-mihajlo/gemma-4-12b-it-social-post-lora`

## Local Command Shape

```bash
accelerate launch \
  --config_file configs/accelerate_fsdp_qlora_gemma4_12b.yaml \
  -m shape_of_text.train \
  --model-id google/gemma-4-12B-it \
  --model-class image-text-to-text \
  --dataset-format instruction-jsonl \
  --train-file examples/hackathon_social_instructions/train.jsonl \
  --eval-file examples/hackathon_social_instructions/validation.jsonl \
  --output-dir runs/gemma4-12b-social-post-lora \
  --use-chat-template \
  --fsdp "full_shard auto_wrap" \
  --fsdp-config configs/trainer_fsdp_qlora_gemma4_12b.json \
  --max-length 1024 \
  --max-steps 1000 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --mmd-weight 0.03 \
  --jmq-weight 0.03 \
  --mmd-warmup-steps 100 \
  --jmq-warmup-steps 100 \
  --kl-eval-batches 8
```

After local training, upload the saved adapter folder with
`scripts/upload_hf_adapter.py`. On Hugging Face Jobs, prefer the generated job
payload because Jobs write tokens may require pull-request uploads instead of
direct commits to `main`.

## HF Jobs Shape

Use Colab Pro for interactive training when it is available. The Colab runner is
the lowest-friction remote path for this repo:

```bash
GIT_REF=mihajlo/social-style-alignment-framework \
HUB_MODEL_ID=micic-mihajlo/gemma-4-12b-it-founder-rewrite-lora \
MAX_STEPS=220 \
MAX_LENGTH=768 \
python scripts/run_colab_founder_rewrite_training.py
```

It expects `HF_TOKEN` in Colab Secrets. The token must read gated Gemma weights
and write to the target adapter repo. Do not use the local MLX path for long
training while the laptop is needed for other work.

Hugging Face ZeroGPU is useful after training for a small Space/demo endpoint.
It is not the right place to train a 12B QLoRA adapter; use Colab Pro or HF Jobs
for the training run, then use ZeroGPU only to serve or test the uploaded
adapter if needed.

For the first paid run, use a short detached smoke job on `l40sx1` or
`a10g-large`, with `--max-steps 10` and `--max-length 512`. The committed
Accelerate config uses `num_processes: 1` so the default payload matches those
one-GPU flavors. Only increase GPU count, length, and steps after the smoke logs
show a clean train step, eval step, and adapter save.

In smoke mode, `scripts/build_hf_job_payload.py` also lowers LoRA to rank 8,
logs every step, evaluates at step 5, and saves at step 10. That makes the smoke
run a systems proof, not a quality run.

Before launching, run preflight locally:

```bash
python scripts/validate_preflight.py \
  --train-file data/social-instructions/train.jsonl \
  --eval-file data/social-instructions/validation.jsonl
```

For a systems-only smoke job from a clean clone, use the committed fake examples
instead of local ignored data. These are the payload defaults:

```bash
python scripts/build_hf_job_payload.py \
  --git-ref YOUR_COMMITTED_SHA \
  --mode preflight \
  --train-file examples/social_instructions/train.jsonl \
  --eval-file examples/social_instructions/validation.jsonl \
  --hub-model-id micic-mihajlo/gemma-4-12b-it-social-post-lora \
  --detach \
  > hf-preflight-job.json
```

The preflight payload uses a CPU flavor by default. It clones the committed SHA,
installs system packages needed by the PyTorch runtime image (`git` and
`build-essential`), installs the package, checks `gcc`, checks the module
entrypoint, validates example data, validates a generated smoke payload, and
runs the test suite. It does not need `HF_TOKEN` because it does not download
Gemma or push an adapter.

For the GPU Gemma smoke job, use:

```bash
python scripts/build_hf_job_payload.py \
  --git-ref YOUR_COMMITTED_SHA \
  --mode smoke \
  --train-file examples/social_instructions/train.jsonl \
  --eval-file examples/social_instructions/validation.jsonl \
  --hub-model-id micic-mihajlo/gemma-4-12b-it-social-post-lora \
  --detach \
  > hf-smoke-job.json
```

For the first quality run, use the instruction-tuned Gemma checkpoint and the
committed original social-writing examples:

```bash
python scripts/build_hf_job_payload.py \
  --git-ref YOUR_COMMITTED_SHA \
  --mode full \
  --model-id google/gemma-4-12B-it \
  --train-file examples/hackathon_social_instructions/train.jsonl \
  --eval-file examples/hackathon_social_instructions/validation.jsonl \
  --hub-model-id micic-mihajlo/gemma-4-12b-it-social-post-lora \
  --max-steps 180 \
  --learning-rate 8e-5 \
  --mmd-weight 0.01 \
  --jmq-weight 0.01 \
  --mmd-warmup-steps 30 \
  --jmq-warmup-steps 30 \
  --detach \
  > hf-quality-job.json
```

Generate the HF Jobs payload from the committed SHA:

```bash
python scripts/build_hf_job_payload.py \
  --git-ref YOUR_COMMITTED_SHA \
  --mode smoke \
  --hub-model-id micic-mihajlo/gemma-4-12b-it-social-post-lora \
  --detach \
  > hf-smoke-job.json
```

The generated job command will:

1. Clone this repository at a committed SHA.
2. Install `git` and `build-essential` if the CUDA image does not include them.
3. Install `pip install -e ".[dev]"`.
4. Check that `HF_TOKEN` exposes Hub `repo.write` before spending GPU time.
5. Run the `accelerate launch` command above with the smoke settings.
6. Pass `HF_TOKEN` as a secret so the job can read gated Gemma weights.
7. Generate held-out posts from `configs/social_eval_briefs.jsonl`.
8. Run deterministic style metrics against the validation completions.
9. Compare base-model generations against adapter generations.
10. Run `scripts/check_generation_quality.py` against adapter generations. This
    fails the job before upload if outputs echo prompts, leak control tokens,
    emit code/template text, or miss basic social-post length bounds.
11. Write a model-card draft and `eval_summary.json` into
   `/workspace/adapter_report`.
12. Copy report artifacts into the saved adapter folder.
13. Upload the adapter folder with `scripts/upload_hf_adapter.py --create-pr`;
    this works with Jobs tokens that can open Hub PRs but cannot write directly
    to `main`.

Do not start with a long run. The first paid run is a systems test, not a model
quality run.

## Post-Job Evaluation

After the adapter is pushed, generate held-out posts:

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

Then compare style metrics against validation completions:

```bash
python scripts/evaluate_social_style.py \
  outputs/adapter_posts.jsonl \
  --candidate-field completion \
  --target-file examples/hackathon_social_instructions/validation.jsonl \
  --target-field completion \
  > outputs/style_report.json

python scripts/compare_social_outputs.py \
  --adapter-file outputs/adapter_posts.jsonl \
  --base-file outputs/base_posts.jsonl \
  --target-file examples/hackathon_social_instructions/validation.jsonl \
  > outputs/comparison_report.json

python scripts/check_generation_quality.py \
  outputs/adapter_posts.jsonl \
  --output-file outputs/generation_quality_report.json
```

Write the adapter report artifacts:

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
