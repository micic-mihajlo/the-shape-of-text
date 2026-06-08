from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("datasets")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from shape_of_text.losses import CausalLMAlignmentLoss
from shape_of_text.train import (
    AlignmentTrainer,
    CausalLMCollator,
    fsdp_uses_activation_checkpointing,
    recast_non_quantized_params_for_fsdp,
    tokenize_chat_instruction,
    trainer_gradient_checkpointing_enabled,
    trainer_optimizer_name,
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


class DummyChatTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = ""
        for message in messages:
            text += f"<{message['role']}>" + message["content"] + f"</{message['role']}>"
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) for char in text]


class DummyStringChatTokenizer(DummyChatTokenizer):
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        return "".join(
            f"<{message['role']}>" + message["content"] + f"</{message['role']}>"
            for message in messages
        ) + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}


class DummyDictStringChatTokenizer(DummyStringChatTokenizer):
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        return {
            "input_ids": super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
        }


class DummyBatchEncodingChatTokenizer(DummyChatTokenizer):
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        input_ids = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        return transformers.BatchEncoding(
            {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
        )


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


def test_tokenize_chat_instruction_masks_prompt_prefix():
    input_ids, attention_mask, labels = tokenize_chat_instruction(
        DummyChatTokenizer(),
        prompt="Write a post.",
        completion="We shipped the small fix today.",
        max_length=512,
    )

    first_label = next(index for index, value in enumerate(labels) if value != -100)
    assert labels[:first_label] == [-100] * first_label
    assert labels[first_label:] == input_ids[first_label:]
    assert attention_mask == [1] * len(input_ids)


def test_tokenize_chat_instruction_handles_string_chat_template_return():
    input_ids, _, labels = tokenize_chat_instruction(
        DummyStringChatTokenizer(),
        prompt="Write a post.",
        completion="We shipped the small fix today.",
        max_length=512,
    )

    assert all(isinstance(token_id, int) for token_id in input_ids)
    assert any(label != -100 for label in labels)


def test_tokenize_chat_instruction_handles_dict_string_chat_template_return():
    input_ids, _, labels = tokenize_chat_instruction(
        DummyDictStringChatTokenizer(),
        prompt="Write a post.",
        completion="We shipped the small fix today.",
        max_length=512,
    )

    assert all(isinstance(token_id, int) for token_id in input_ids)
    assert any(label != -100 for label in labels)


def test_tokenize_chat_instruction_handles_batch_encoding_return():
    input_ids, _, labels = tokenize_chat_instruction(
        DummyBatchEncodingChatTokenizer(),
        prompt="Write a post.",
        completion="We shipped the small fix today.",
        max_length=512,
    )

    assert all(isinstance(token_id, int) for token_id in input_ids)
    assert any(label != -100 for label in labels)


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


def test_causal_lm_collator_rejects_string_token_ids():
    collator = CausalLMCollator(DummyTokenizer("right"))

    with pytest.raises(ValueError, match="input_ids must be token ids"):
        collator([{"input_ids": "input_ids", "attention_mask": [1], "labels": [1]}])


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


def test_recast_non_quantized_params_for_fsdp_makes_float_params_uniform():
    model = torch.nn.Module()
    model.fp32 = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    model.bf16 = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))

    recast_non_quantized_params_for_fsdp(model, torch.bfloat16)

    assert model.fp32.dtype is torch.bfloat16
    assert model.bf16.dtype is torch.bfloat16


def test_trainer_optimizer_uses_torch_adam_for_fsdp_qlora():
    args = SimpleNamespace(fsdp="full_shard auto_wrap", no_4bit=False)

    assert trainer_optimizer_name(args) == "adamw_torch"


def test_trainer_optimizer_uses_bitsandbytes_without_fsdp_qlora():
    args = SimpleNamespace(fsdp="", no_4bit=False)

    assert trainer_optimizer_name(args) == "paged_adamw_8bit"


def test_trainer_optimizer_uses_torch_adam_without_4bit():
    args = SimpleNamespace(fsdp="", no_4bit=True)

    assert trainer_optimizer_name(args) == "adamw_torch"
