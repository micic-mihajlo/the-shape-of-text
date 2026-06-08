from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from shape_of_text.losses import (
    _EPS,
    _gather_vocab,
    _prepare_alignment_inputs,
    _sample_vocab_indices,
    _target_probabilities_from_labels,
)


def kl_from_logits(
    student_logits: Tensor,
    *,
    labels: Tensor | None = None,
    target_logits: Tensor | None = None,
    attention_mask: Tensor | None = None,
    ignore_index: int = -100,
    shift_labels: bool = True,
    smoothing: float = 0.02,
    temperature: float = 1.0,
    max_positions: int | None = None,
    vocab_sample_size: int | None = None,
) -> dict[str, Tensor]:
    """Estimate KL(target || student) for aligned causal-LM token positions."""

    if target_logits is None and labels is None:
        raise ValueError("kl_from_logits requires target_logits or labels")

    student, target, flat_labels = _prepare_alignment_inputs(
        student_logits,
        labels=labels,
        target_logits=target_logits,
        attention_mask=attention_mask,
        ignore_index=ignore_index,
        shift_labels=shift_labels,
        max_positions=max_positions,
    )
    if student.numel() == 0:
        zero = student_logits.new_zeros(())
        return {
            "kl_divergence": zero,
            "student_entropy": zero,
            "target_entropy": zero,
            "num_tokens": zero,
        }

    vocab_ids = _sample_vocab_indices(
        labels=flat_labels,
        vocab_size=student_logits.size(-1),
        sample_size=vocab_sample_size,
        device=student.device,
    )
    student = _gather_vocab(student, vocab_ids).float() / temperature
    student_log_probs = F.log_softmax(student, dim=-1)
    student_probs = student_log_probs.exp()

    if target is not None:
        target = _gather_vocab(target.detach(), vocab_ids).float() / temperature
        target_log_probs = F.log_softmax(target, dim=-1)
        target_probs = target_log_probs.exp()
    else:
        if flat_labels is None:
            raise ValueError("labels are required when target_logits is not provided")
        target_probs = _target_probabilities_from_labels(
            labels=flat_labels,
            vocab_ids=vocab_ids,
            vocab_size=student_logits.size(-1),
            smoothing=smoothing,
            dtype=torch.float32,
            device=student.device,
        )
        target_log_probs = target_probs.clamp_min(_EPS).log()

    per_token_kl = (target_probs * (target_log_probs - student_log_probs)).sum(dim=-1)
    student_entropy = -(student_probs * student_log_probs).sum(dim=-1)
    target_entropy = -(target_probs * target_log_probs).sum(dim=-1)

    return {
        "kl_divergence": per_token_kl.mean(),
        "student_entropy": student_entropy.mean(),
        "target_entropy": target_entropy.mean(),
        "num_tokens": torch.tensor(float(per_token_kl.numel()), device=student_logits.device),
    }


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, Tensor) else value
    return moved


def _model_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"input_ids", "attention_mask", "position_ids"}
    return {key: value for key, value in batch.items() if key in allowed}


def _extract_logits(outputs: Any) -> Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, Mapping) and "logits" in outputs:
        return outputs["logits"]
    raise TypeError("model outputs must expose a logits tensor")


@torch.no_grad()
def evaluate_kl(
    model: torch.nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    *,
    target_model: torch.nn.Module | None = None,
    max_batches: int | None = None,
    smoothing: float = 0.02,
    temperature: float = 1.0,
    vocab_sample_size: int | None = None,
) -> dict[str, float]:
    """Run a KL evaluation loop over a dataloader.

    When target_model is omitted, the target is the smoothed empirical next-token
    distribution from the labels in each batch.
    """

    was_training = model.training
    target_was_training = target_model.training if target_model is not None else False
    model.eval()
    if target_model is not None:
        target_model.eval()

    device = next(model.parameters()).device
    totals = {"kl_divergence": 0.0, "student_entropy": 0.0, "target_entropy": 0.0}
    total_tokens = 0.0

    for batch_idx, raw_batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = _move_batch_to_device(raw_batch, device)
        labels = batch.get("labels", batch.get("input_ids"))
        outputs = model(**_model_inputs(batch))
        logits = _extract_logits(outputs)

        target_logits = None
        if target_model is not None:
            target_outputs = target_model(**_model_inputs(batch))
            target_logits = _extract_logits(target_outputs)

        metrics = kl_from_logits(
            logits,
            labels=labels,
            target_logits=target_logits,
            attention_mask=batch.get("attention_mask"),
            smoothing=smoothing,
            temperature=temperature,
            vocab_sample_size=vocab_sample_size,
        )
        token_count = float(metrics["num_tokens"].item())
        if token_count == 0:
            continue
        total_tokens += token_count
        for key in totals:
            totals[key] += float(metrics[key].item()) * token_count

    if was_training:
        model.train()
    if target_model is not None and target_was_training:
        target_model.train()

    if total_tokens == 0:
        return {key: 0.0 for key in totals} | {"num_tokens": 0.0}
    return {key: value / total_tokens for key, value in totals.items()} | {
        "num_tokens": total_tokens
    }
