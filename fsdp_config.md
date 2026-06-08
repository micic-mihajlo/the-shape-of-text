# FSDP Configuration for Gemma 12B QLoRA Style Alignment

This repository uses QLoRA for trainable memory efficiency and FSDP-compatible
wrapping for the 12B-class backbone. The committed Accelerate config is locked
to one process for low-cost `l40sx1` smoke jobs; raise `num_processes` only when
the selected host actually has multiple GPUs. The concrete launcher files are:

- `configs/accelerate_fsdp_qlora_gemma4_12b.yaml`
- `configs/trainer_fsdp_qlora_gemma4_12b.json`

## Locked Defaults

- Sharding strategy: `FULL_SHARD`
- Precision: `bf16`
- Quantization: 4-bit NF4, double quantization, bf16 compute, bf16 quant storage
- LoRA target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, `down_proj`
- FSDP auto-wrap policy: transformer-layer wrapping
- FSDP state dict: sharded state dict
- PEFT compatibility: `use_orig_params=true`
- Default process count: `num_processes=1` for one-GPU smoke validation
- Gradient checkpointing: enabled
- Forward prefetch: disabled
- Backward prefetch: `BACKWARD_PRE`
- CPU parameter offload: disabled by default

## Transformer Layer Class

The default layer class is:

```text
Gemma4TextDecoderLayer
```

If the selected Hugging Face Gemma 4 checkpoint exposes a different decoder
class, update both config files before launch. Common adjacent names in Gemma
families are `GemmaDecoderLayer`, `Gemma2DecoderLayer`, and
`Gemma3DecoderLayer`.

## Low-Memory Run Order

For a new host, use this sequence:

1. Run a 10-step smoke test with `--max-length 512`, LoRA rank 8, and no target
   model.
2. Increase to `--max-length 1024`.
3. Increase LoRA rank to 16.
4. Increase MMD/JMQ sample sizes only after the base run is stable.
5. For multi-GPU hosts, raise `num_processes` in the Accelerate config to match
   the GPU count before relying on FSDP sharding behavior.
6. Add a frozen target model only if the host has enough spare memory, or use
   cached target logits instead.

## OOM Triage

Reduce memory pressure in this order:

1. Lower `--max-length`.
2. Lower `--mmd-vocab-sample-size` and `--jmq-vocab-sample-size`.
3. Lower `--mmd-max-positions` and `--jmq-max-positions`.
4. Lower `--lora-r`.
5. Increase `--gradient-accumulation-steps` and keep per-device batch size at 1.
6. Disable the optional frozen target model.
7. As a last resort, enable CPU offload in the Accelerate config.

## Launch Command

```bash
accelerate launch \
  --config_file configs/accelerate_fsdp_qlora_gemma4_12b.yaml \
  -m shape_of_text.train \
  --model-id google/gemma-4-12B \
  --train-file data/train.txt \
  --eval-file data/validation.txt \
  --fsdp "full_shard auto_wrap" \
  --fsdp-config configs/trainer_fsdp_qlora_gemma4_12b.json \
  --max-length 1024 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8
```
