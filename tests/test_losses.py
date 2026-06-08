import pytest

torch = pytest.importorskip("torch")

from shape_of_text.losses import CausalLMAlignmentLoss, JMQLoss, MMDLoss


def test_mmd_zero_for_matching_target_logits():
    logits = torch.randn(2, 4, 17)
    labels = torch.randint(0, 17, (2, 4))
    loss = MMDLoss(max_positions=8, vocab_sample_size=12)(
        logits,
        labels=labels,
        target_logits=logits.clone(),
        shift_labels=True,
    )
    assert loss.item() < 1e-6


def test_jmq_zero_for_matching_target_logits():
    logits = torch.randn(2, 5, 19)
    labels = torch.randint(0, 19, (2, 5))
    loss = JMQLoss(max_positions=8, vocab_sample_size=12)(
        logits,
        labels=labels,
        target_logits=logits.clone(),
        shift_labels=True,
    )
    assert loss.item() < 1e-6


def test_causal_alignment_loss_backpropagates():
    logits = torch.randn(2, 6, 23, requires_grad=True)
    labels = torch.randint(0, 23, (2, 6))
    labels[:, 0] = -100
    loss_fn = CausalLMAlignmentLoss(
        mmd_weight=0.1,
        jmq_weight=0.1,
        mmd_loss=MMDLoss(max_positions=8, vocab_sample_size=16),
        jmq_loss=JMQLoss(max_positions=8, vocab_sample_size=16),
    )
    output = loss_fn(logits, labels)
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
