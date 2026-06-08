from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("datasets")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from shape_of_text.losses import CausalLMAlignmentLoss
from shape_of_text.train import (
    AlignmentTrainer,
    CausalLMCollator,
    fsdp_uses_activation_checkpointing,
    trainer_gradient_checkpointing_enabled,
)


class DummyTokenizer:
    pad_token_id = 0

    def __init__(self, padding_side: str) -> None:
        self.padding_side = padding_side

    def pad(self, features, padding=True, return_tensors="pt"):
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            padding_ids = [self.pad_token_id] * pad_len
            padding_mask = [0] * pad_len
            if self.padding_side == "left":
                input_ids.append(padding_ids + feature["input_ids"])
                attention_mask.append(padding_mask + feature["attention_mask"])
            else:
                input_ids.append(feature["input_ids"] + padding_ids)
                attention_mask.append(feature["attention_mask"] + padding_mask)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def test_alignment_weight_warmup_schedule():
    trainer = object.__new__(AlignmentTrainer)
    trainer.state = SimpleNamespace(global_step=4)
    trainer.alignment_loss = CausalLMAlignmentLoss(mmd_weight=0.2, jmq_weight=0.4)
    trainer.mmd_weight = 0.2
    trainer.jmq_weight = 0.4
    trainer.mmd_warmup_steps = 10
    trainer.jmq_warmup_steps = 5

    trainer._apply_alignment_schedule()

    assert trainer.alignment_loss.mmd_weight == 0.1
    assert trainer.alignment_loss.jmq_weight == 0.4


def test_causal_lm_collator_pads_labels_on_right():
    collator = CausalLMCollator(DummyTokenizer("right"))
    batch = collator(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [3], "attention_mask": [1], "labels": [3]},
        ]
    )
    assert batch["labels"].tolist() == [[-100, 2], [3, -100]]


def test_causal_lm_collator_pads_labels_on_left():
    collator = CausalLMCollator(DummyTokenizer("left"))
    batch = collator(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [3], "attention_mask": [1], "labels": [3]},
        ]
    )
    assert batch["labels"].tolist() == [[-100, 2], [-100, 3]]


def test_trainer_gradient_checkpointing_yields_to_fsdp_activation_checkpointing():
    args = SimpleNamespace(no_gradient_checkpointing=False, fsdp="full_shard auto_wrap")

    assert trainer_gradient_checkpointing_enabled(args, {"activation_checkpointing": True}) is False


def test_trainer_gradient_checkpointing_stays_enabled_without_fsdp_activation_checkpointing():
    args = SimpleNamespace(no_gradient_checkpointing=False, fsdp="full_shard auto_wrap")

    assert trainer_gradient_checkpointing_enabled(args, {"activation_checkpointing": False}) is True


def test_trainer_gradient_checkpointing_respects_explicit_disable():
    args = SimpleNamespace(no_gradient_checkpointing=True, fsdp="")

    assert trainer_gradient_checkpointing_enabled(args, None) is False


def test_fsdp_activation_checkpointing_supports_accelerate_key():
    assert fsdp_uses_activation_checkpointing({"fsdp_activation_checkpointing": True}) is True
