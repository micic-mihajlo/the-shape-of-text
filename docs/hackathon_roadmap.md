# Hackathon Roadmap

This is the next useful layer beyond the initial losses and FSDP path.

## Data

- Build a clean human-reference corpus split by genre, length, and task type.
- Add source metadata for evaluation slicing without binding the model to a
  brand, person, or private identity.
- Add deduplication and near-duplicate checks before fine-tuning.

## Training

- Cache target logits for the human corpus to avoid loading a frozen reference
  model during every training step.
- Add a schedule that warms in MMD and JMQ after the cross-entropy loss has
  stabilized.
- Track semantic preservation with a held-out instruction set and response
  similarity checks.
- Add generation-time entropy constraints so sampling does not become erratic.

## Evaluation

- Add a distribution dashboard for KL, entropy, MMD, JMQ, repetition rate,
  distinct-n, and syntax validity.
- Add coherence evals: grammar acceptability, contradiction checks, and task
  answer preservation.
- Add detector-agnostic scorecards as external measurements only; the internal
  optimization target should remain human-reference distribution alignment and
  coherence.

## Shipping

- Publish LoRA adapters separately from the base model.
- Add a small Gradio app that compares base, SFT-only, and distribution-aligned
  generations.
- Add a field report documenting what improved, what broke, and which metrics
  were actually predictive.
