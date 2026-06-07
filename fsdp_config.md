# FSDP Configuration for 12b Text Model Alignment

Efficiently fine-tuning a 12b parameter model for text tasks requires careful memory orchestration. This configuration utilizes Fully Sharded Data Parallel (FSDP) to enable training on standard data-center GPUs.

## Configuration Guidelines
- Sharding Strategy: `FULL_SHARD` is recommended to maximize memory savings across the cluster.
- Mixed Precision: Use `bf16` (BFloat16) to ensure stability during the alignment of large language models.
- Transformer Layer Wrapping: Ensure `transformer_layer_cls` is correctly specified for Gemma 4 to enable effective sharding of the attention and feed-forward blocks.
- Activation Checkpointing: Enable this to allow for larger batch sizes and longer sequence lengths, which are critical for capturing long-form writing styles.
- Optimizer States: Use sharded optimizers to prevent any single GPU from becoming a memory bottleneck.
