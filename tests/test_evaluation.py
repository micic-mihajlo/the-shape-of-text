import pytest

torch = pytest.importorskip("torch")

from shape_of_text.evaluation import kl_from_logits


def test_kl_is_zero_for_identical_target_logits():
    logits = torch.randn(2, 5, 13)
    labels = torch.randint(0, 13, (2, 5))
    metrics = kl_from_logits(logits, labels=labels, target_logits=logits.clone())
    assert metrics["kl_divergence"].item() < 1e-6
    assert metrics["num_tokens"].item() > 0
