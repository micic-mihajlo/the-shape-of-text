# Hugging Face Job Runbook

Use this after the repo changes are committed and pushed. The job should consume
prepared `instruction-jsonl` files and publish a LoRA adapter, not a merged base
model.

## Inputs

- `data/social-instructions/train.jsonl`
- `data/social-instructions/validation.jsonl`
- Hugging Face token with access to `google/gemma-4-12B`
- Target adapter repo, for example:
  `micic-mihajlo/gemma-4-12b-social-post-lora`

## Local Command Shape

```bash
accelerate launch \
  --config_file configs/accelerate_fsdp_qlora_gemma4_12b.yaml \
  -m shape_of_text.train \
  --model-id google/gemma-4-12B \
  --model-class image-text-to-text \
  --dataset-format instruction-jsonl \
  --train-file data/social-instructions/train.jsonl \
  --eval-file data/social-instructions/validation.jsonl \
  --output-dir runs/gemma4-12b-social-post-lora \
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
  --kl-eval-batches 8 \
  --push-to-hub \
  --hub-model-id micic-mihajlo/gemma-4-12b-social-post-lora
```

## HF Jobs Shape

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
instead of local ignored data:

```bash
python scripts/build_hf_job_payload.py \
  --git-ref YOUR_COMMITTED_SHA \
  --mode preflight \
  --train-file examples/social_instructions/train.jsonl \
  --eval-file examples/social_instructions/validation.jsonl \
  --hub-model-id micic-mihajlo/gemma-4-12b-social-post-lora \
  --detach \
  > hf-preflight-job.json
```

The preflight payload uses a CPU flavor by default. It clones the committed SHA,
installs the package, checks the module entrypoint, validates example data,
validates a generated smoke payload, and runs the test suite. It does not need
`HF_TOKEN` because it does not download Gemma or push an adapter.

For the GPU Gemma smoke job, use:

```bash
python scripts/build_hf_job_payload.py \
  --git-ref YOUR_COMMITTED_SHA \
  --mode smoke \
  --train-file examples/social_instructions/train.jsonl \
  --eval-file examples/social_instructions/validation.jsonl \
  --hub-model-id micic-mihajlo/gemma-4-12b-social-post-lora \
  --detach \
  > hf-smoke-job.json
```

Generate the HF Jobs payload from the committed SHA:

```bash
python scripts/build_hf_job_payload.py \
  --git-ref YOUR_COMMITTED_SHA \
  --mode smoke \
  --hub-model-id micic-mihajlo/gemma-4-12b-social-post-lora \
  --detach \
  > hf-smoke-job.json
```

The generated job command will:

1. Clone this repository at a committed SHA.
2. Install `git` if the CUDA image does not include it.
3. Install `pip install -e ".[dev]"`.
4. Run the `accelerate launch` command above with the smoke settings.
5. Pass `HF_TOKEN` as a secret so the job can read gated Gemma weights and push
   the adapter.
6. Generate held-out posts from `configs/social_eval_briefs.jsonl`.
7. Run deterministic style metrics against the validation completions.
8. Compare base-model generations against adapter generations.
9. Write a model-card draft and `eval_summary.json` into
   `/workspace/adapter_report`.

Do not start with a long run. The first paid run is a systems test, not a model
quality run.

## Post-Job Evaluation

After the adapter is pushed, generate held-out posts:

```bash
python scripts/generate_social_posts.py \
  --model-id google/gemma-4-12B \
  --briefs-file configs/social_eval_briefs.jsonl \
  --output-file outputs/base_posts.jsonl

python scripts/generate_social_posts.py \
  --model-id google/gemma-4-12B \
  --adapter-id micic-mihajlo/gemma-4-12b-social-post-lora \
  --briefs-file configs/social_eval_briefs.jsonl \
  --output-file outputs/adapter_posts.jsonl
```

Then compare style metrics against validation completions:

```bash
python scripts/evaluate_social_style.py \
  outputs/adapter_posts.jsonl \
  --candidate-field completion \
  --target-file data/social-instructions/validation.jsonl \
  --target-field completion \
  > outputs/style_report.json

python scripts/compare_social_outputs.py \
  --adapter-file outputs/adapter_posts.jsonl \
  --base-file outputs/base_posts.jsonl \
  --target-file data/social-instructions/validation.jsonl \
  > outputs/comparison_report.json
```

Write the adapter report artifacts:

```bash
python scripts/write_adapter_report.py \
  --adapter-id micic-mihajlo/gemma-4-12b-social-post-lora \
  --base-model google/gemma-4-12B \
  --train-file data/social-instructions/train.jsonl \
  --eval-file data/social-instructions/validation.jsonl \
  --generated-file outputs/adapter_posts.jsonl \
  --style-report outputs/style_report.json \
  --comparison-report outputs/comparison_report.json \
  --output-dir outputs/adapter_report
```
