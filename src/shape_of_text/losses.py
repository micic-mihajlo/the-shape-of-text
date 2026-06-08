from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_EPS = 1e-8


@dataclass(frozen=True)
class AlignmentLossOutput:
    """Structured loss return for logging and backward passes."""

    loss: Tensor
    lm_loss: Tensor
    mmd_loss: Tensor
    jmq_loss: Tensor
    components: Mapping[str, Tensor]


def _shift_for_causal_lm(
    logits: Tensor,
    labels: Tensor | None,
    target_logits: Tensor | None,
    attention_mask: Tensor | None,
    shift_labels: bool,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    if not shift_labels:
        return logits, labels, target_logits, attention_mask

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous() if labels is not None else None
    shifted_target = target_logits[:, :-1, :].contiguous() if target_logits is not None else None
    shifted_mask = attention_mask[:, 1:].contiguous() if attention_mask is not None else None
    return shifted_logits, shifted_labels, shifted_target, shifted_mask


def _prepare_alignment_inputs(
    student_logits: Tensor,
    *,
    labels: Tensor | None,
    target_logits: Tensor | None,
    attention_mask: Tensor | None,
    ignore_index: int,
    shift_labels: bool,
    max_positions: int | None,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    student_logits, labels, target_logits, attention_mask = _shift_for_causal_lm(
        student_logits, labels, target_logits, attention_mask, shift_labels
    )

    vocab_size = student_logits.size(-1)
    student_flat = student_logits.reshape(-1, vocab_size)
    target_flat = target_logits.reshape(-1, vocab_size) if target_logits is not None else None

    if labels is None:
        label_flat = None
        keep = torch.ones(student_flat.size(0), dtype=torch.bool, device=student_flat.device)
    else:
        label_flat = labels.reshape(-1)
        keep = label_flat.ne(ignore_index)

    if attention_mask is not None:
        keep = keep & attention_mask.reshape(-1).bool()

    student_flat = student_flat[keep]
    target_flat = target_flat[keep] if target_flat is not None else None
    label_flat = label_flat[keep] if label_flat is not None else None

    if max_positions is not None and student_flat.size(0) > max_positions:
        sample_ids = torch.randperm(
            student_flat.size(0), device=student_flat.device
        )[:max_positions]
        student_flat = student_flat.index_select(0, sample_ids)
        target_flat = target_flat.index_select(0, sample_ids) if target_flat is not None else None
        label_flat = label_flat.index_select(0, sample_ids) if label_flat is not None else None

    return student_flat, target_flat, label_flat


def _sample_vocab_indices(
    *,
    labels: Tensor | None,
    vocab_size: int,
    sample_size: int | None,
    device: torch.device,
) -> Tensor | None:
    if sample_size is None or sample_size <= 0 or sample_size >= vocab_size:
        return None

    if labels is None or labels.numel() == 0:
        return torch.randperm(vocab_size, device=device)[:sample_size].sort().values

    label_ids = labels[labels.ge(0)].unique()
    if label_ids.numel() > sample_size:
        label_ids = label_ids[
            torch.randperm(label_ids.numel(), device=device)[:sample_size]
        ].unique()

    remaining = sample_size - label_ids.numel()
    if remaining <= 0:
        return label_ids[:sample_size].sort().values

    random_ids = torch.empty(0, dtype=torch.long, device=device)
    while random_ids.numel() < remaining:
        draw_count = max(remaining * 2, 32)
        drawn = torch.randint(0, vocab_size, (draw_count,), dtype=torch.long, device=device)
        drawn = drawn[~torch.isin(drawn, label_ids)]
        random_ids = torch.cat([random_ids, drawn]).unique()

    vocab_ids = torch.cat([label_ids, random_ids[:remaining]]).unique()
    if vocab_ids.numel() > sample_size:
        vocab_ids = vocab_ids[:sample_size]
    return vocab_ids.sort().values


def _gather_vocab(logits: Tensor, vocab_ids: Tensor | None) -> Tensor:
    if vocab_ids is None:
        return logits
    return logits.index_select(-1, vocab_ids)


def _target_probabilities_from_labels(
    *,
    labels: Tensor,
    vocab_ids: Tensor | None,
    vocab_size: int,
    smoothing: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if vocab_ids is None:
        vocab_ids = torch.arange(vocab_size, dtype=torch.long, device=device)

    width = vocab_ids.numel()
    smoothing = float(max(0.0, min(1.0, smoothing)))
    target = torch.full(
        (labels.numel(), width),
        smoothing / width,
        dtype=torch.float32,
        device=device,
    )
    matches = labels[:, None].eq(vocab_ids[None, :])
    target = target + matches.to(torch.float32) * (1.0 - smoothing)
    row_sums = target.sum(dim=-1, keepdim=True)
    missing_rows = row_sums.le(_EPS).squeeze(-1)
    if missing_rows.any():
        target[missing_rows] = 1.0 / width
        row_sums = target.sum(dim=-1, keepdim=True)
    target = target / row_sums.clamp_min(_EPS)
    return target.to(dtype=dtype)


def _normalize_pair(student: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    combined = torch.cat([student.detach(), target.detach()], dim=0)
    mean = combined.mean(dim=0, keepdim=True)
    scale = combined.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
    return (student - mean) / scale, (target - mean) / scale


class MMDLoss(nn.Module):
    """Maximum mean discrepancy loss over sampled per-token logit distributions.

    The loss accepts either frozen target logits or target labels. Target logits
    are preferred when cached reference distributions are available. Label-only
    mode builds a smoothed empirical target distribution over a sampled vocab.
    """

    def __init__(
        self,
        *,
        kernel_bandwidths: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
        max_positions: int | None = 128,
        vocab_sample_size: int | None = 2048,
        target_smoothing: float = 0.02,
        temperature: float = 1.0,
        use_probabilities: bool = False,
        normalize_features: bool = True,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if not kernel_bandwidths:
            raise ValueError("kernel_bandwidths must contain at least one value")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.kernel_bandwidths = tuple(float(value) for value in kernel_bandwidths)
        self.max_positions = max_positions
        self.vocab_sample_size = vocab_sample_size
        self.target_smoothing = target_smoothing
        self.temperature = temperature
        self.use_probabilities = use_probabilities
        self.normalize_features = normalize_features
        self.ignore_index = ignore_index

    def forward(
        self,
        student_logits: Tensor,
        *,
        labels: Tensor | None = None,
        target_logits: Tensor | None = None,
        attention_mask: Tensor | None = None,
        shift_labels: bool = False,
    ) -> Tensor:
        if target_logits is None and labels is None:
            raise ValueError("MMDLoss requires target_logits or labels")

        student, target, flat_labels = _prepare_alignment_inputs(
            student_logits,
            labels=labels,
            target_logits=target_logits,
            attention_mask=attention_mask,
            ignore_index=self.ignore_index,
            shift_labels=shift_labels,
            max_positions=self.max_positions,
        )
        if student.numel() == 0:
            return student_logits.new_zeros(())

        vocab_ids = _sample_vocab_indices(
            labels=flat_labels,
            vocab_size=student_logits.size(-1),
            sample_size=self.vocab_sample_size,
            device=student.device,
        )
        student = _gather_vocab(student, vocab_ids).float() / self.temperature

        if target is not None:
            target = _gather_vocab(target.detach(), vocab_ids).float() / self.temperature
            if self.use_probabilities:
                target_features = F.softmax(target, dim=-1)
            else:
                target_features = target
        else:
            if flat_labels is None:
                raise ValueError("labels are required when target_logits is not provided")
            target_probs = _target_probabilities_from_labels(
                labels=flat_labels,
                vocab_ids=vocab_ids,
                vocab_size=student_logits.size(-1),
                smoothing=self.target_smoothing,
                dtype=torch.float32,
                device=student.device,
            )
            target_features = (
                target_probs if self.use_probabilities else target_probs.clamp_min(_EPS).log()
            )

        student_features = F.softmax(student, dim=-1) if self.use_probabilities else student
        target_features = target_features.detach()

        if self.normalize_features:
            student_features, target_features = _normalize_pair(student_features, target_features)

        return self._rbf_mmd(student_features, target_features)

    def _rbf_mmd(self, student: Tensor, target: Tensor) -> Tensor:
        loss = student.new_zeros(())
        for bandwidth in self.kernel_bandwidths:
            gamma = 1.0 / (2.0 * bandwidth * bandwidth)
            xx = torch.exp(-_pairwise_squared_distance(student, student) * gamma).mean()
            yy = torch.exp(-_pairwise_squared_distance(target, target) * gamma).mean()
            xy = torch.exp(-_pairwise_squared_distance(student, target) * gamma).mean()
            loss = loss + xx + yy - 2.0 * xy
        return loss / len(self.kernel_bandwidths)


def _pairwise_squared_distance(left: Tensor, right: Tensor) -> Tensor:
    left_norm = left.pow(2).sum(dim=-1, keepdim=True)
    right_norm = right.pow(2).sum(dim=-1, keepdim=True).T
    distances = left_norm + right_norm - 2.0 * (left @ right.T)
    return distances.clamp_min(0.0)


class JMQLoss(nn.Module):
    """Joint moment quantile loss over distributional features.

    JMQ compares entropy, confidence, logit moments, covariance, and quantiles.
    It is intentionally feature-based so it can regularize higher-order shape
    without materializing pairwise kernels across every vocab dimension.
    """

    def __init__(
        self,
        *,
        moment_orders: Sequence[int] = (1, 2, 3, 4),
        quantile_levels: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
        max_positions: int | None = 256,
        vocab_sample_size: int | None = 4096,
        target_smoothing: float = 0.02,
        temperature: float = 1.0,
        moment_weight: float = 1.0,
        quantile_weight: float = 1.0,
        covariance_weight: float = 0.25,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if not moment_orders:
            raise ValueError("moment_orders must contain at least one value")
        if not quantile_levels:
            raise ValueError("quantile_levels must contain at least one value")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.moment_orders = tuple(int(value) for value in moment_orders)
        self.quantile_levels = tuple(float(value) for value in quantile_levels)
        self.max_positions = max_positions
        self.vocab_sample_size = vocab_sample_size
        self.target_smoothing = target_smoothing
        self.temperature = temperature
        self.moment_weight = moment_weight
        self.quantile_weight = quantile_weight
        self.covariance_weight = covariance_weight
        self.ignore_index = ignore_index

    def forward(
        self,
        student_logits: Tensor,
        *,
        labels: Tensor | None = None,
        target_logits: Tensor | None = None,
        attention_mask: Tensor | None = None,
        shift_labels: bool = False,
    ) -> Tensor:
        if target_logits is None and labels is None:
            raise ValueError("JMQLoss requires target_logits or labels")

        student, target, flat_labels = _prepare_alignment_inputs(
            student_logits,
            labels=labels,
            target_logits=target_logits,
            attention_mask=attention_mask,
            ignore_index=self.ignore_index,
            shift_labels=shift_labels,
            max_positions=self.max_positions,
        )
        if student.numel() == 0:
            return student_logits.new_zeros(())

        vocab_ids = _sample_vocab_indices(
            labels=flat_labels,
            vocab_size=student_logits.size(-1),
            sample_size=self.vocab_sample_size,
            device=student.device,
        )
        student = _gather_vocab(student, vocab_ids).float() / self.temperature

        if target is not None:
            target = _gather_vocab(target.detach(), vocab_ids).float() / self.temperature
        else:
            if flat_labels is None:
                raise ValueError("labels are required when target_logits is not provided")
            target_probs = _target_probabilities_from_labels(
                labels=flat_labels,
                vocab_ids=vocab_ids,
                vocab_size=student_logits.size(-1),
                smoothing=self.target_smoothing,
                dtype=torch.float32,
                device=student.device,
            )
            target = target_probs.clamp_min(_EPS).log()

        student_features = self._distribution_features(student)
        target_features = self._distribution_features(target).detach()

        loss = student.new_zeros(())
        if self.moment_weight:
            loss = loss + self.moment_weight * self._moment_loss(student_features, target_features)
        if self.quantile_weight:
            loss = loss + self.quantile_weight * self._quantile_loss(
                student_features, target_features
            )
        if self.covariance_weight:
            loss = loss + self.covariance_weight * self._covariance_loss(
                student_features, target_features
            )
        return loss

    @staticmethod
    def _distribution_features(logits: Tensor) -> Tensor:
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        normalized_entropy = entropy / torch.log(
            torch.tensor(float(logits.size(-1)), dtype=entropy.dtype, device=entropy.device)
        ).clamp_min(_EPS)

        top_k = min(2, logits.size(-1))
        top_values = probs.topk(k=top_k, dim=-1).values
        max_prob = top_values[:, 0]
        margin = top_values[:, 0] - top_values[:, 1] if top_k == 2 else torch.zeros_like(max_prob)

        logit_mean = logits.mean(dim=-1)
        centered = logits - logit_mean[:, None]
        logit_std = centered.pow(2).mean(dim=-1).sqrt().clamp_min(1e-4)
        skewness = centered.pow(3).mean(dim=-1) / logit_std.pow(3)
        kurtosis = centered.pow(4).mean(dim=-1) / logit_std.pow(4)
        effective_support = entropy.exp() / logits.size(-1)

        return torch.stack(
            [
                normalized_entropy,
                max_prob,
                margin,
                logit_mean,
                logit_std,
                skewness,
                kurtosis,
                effective_support,
            ],
            dim=-1,
        )

    def _moment_loss(self, student: Tensor, target: Tensor) -> Tensor:
        student_mean = student.mean(dim=0)
        target_mean = target.mean(dim=0)
        student_centered = student - student_mean
        target_centered = target - target_mean

        loss = student.new_zeros(())
        for order in self.moment_orders:
            if order == 1:
                student_moment = student_mean
                target_moment = target_mean
            else:
                student_moment = student_centered.pow(order).mean(dim=0)
                target_moment = target_centered.pow(order).mean(dim=0)
            loss = loss + F.mse_loss(student_moment, target_moment)
        return loss / len(self.moment_orders)

    def _quantile_loss(self, student: Tensor, target: Tensor) -> Tensor:
        quantiles = torch.tensor(self.quantile_levels, dtype=torch.float32, device=student.device)
        student_quantiles = torch.quantile(student.float(), quantiles, dim=0)
        target_quantiles = torch.quantile(target.float(), quantiles, dim=0)
        return F.mse_loss(student_quantiles, target_quantiles)

    @staticmethod
    def _covariance_loss(student: Tensor, target: Tensor) -> Tensor:
        if student.size(0) < 2 or target.size(0) < 2:
            return student.new_zeros(())
        student_centered = student - student.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)
        student_cov = student_centered.T @ student_centered / (student.size(0) - 1)
        target_cov = target_centered.T @ target_centered / (target.size(0) - 1)
        return F.mse_loss(student_cov, target_cov)


class CausalLMAlignmentLoss(nn.Module):
    """Composable causal-LM loss: SFT cross entropy plus MMD and JMQ."""

    def __init__(
        self,
        *,
        lm_weight: float = 1.0,
        mmd_weight: float = 0.05,
        jmq_weight: float = 0.05,
        mmd_loss: MMDLoss | None = None,
        jmq_loss: JMQLoss | None = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.lm_weight = lm_weight
        self.mmd_weight = mmd_weight
        self.jmq_weight = jmq_weight
        self.mmd_loss = mmd_loss or MMDLoss(ignore_index=ignore_index)
        self.jmq_loss = jmq_loss or JMQLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        *,
        target_logits: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> AlignmentLossOutput:
        if labels is None:
            raise ValueError("labels are required for causal LM alignment loss")

        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        if shifted_labels.ne(self.ignore_index).any():
            lm_loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
                ignore_index=self.ignore_index,
            )
        else:
            lm_loss = logits.new_zeros(())

        if self.mmd_weight:
            mmd_loss = self.mmd_loss(
                logits,
                labels=labels,
                target_logits=target_logits,
                attention_mask=attention_mask,
                shift_labels=True,
            )
        else:
            mmd_loss = logits.new_zeros(())

        if self.jmq_weight:
            jmq_loss = self.jmq_loss(
                logits,
                labels=labels,
                target_logits=target_logits,
                attention_mask=attention_mask,
                shift_labels=True,
            )
        else:
            jmq_loss = logits.new_zeros(())

        total = self.lm_weight * lm_loss + self.mmd_weight * mmd_loss + self.jmq_weight * jmq_loss
        return AlignmentLossOutput(
            loss=total,
            lm_loss=lm_loss.detach(),
            mmd_loss=mmd_loss.detach(),
            jmq_loss=jmq_loss.detach(),
            components={
                "loss": total.detach(),
                "lm_loss": lm_loss.detach(),
                "mmd_loss": mmd_loss.detach(),
                "jmq_loss": jmq_loss.detach(),
            },
        )
