from __future__ import annotations

import argparse
import inspect
import json
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from shape_of_text.evaluation import evaluate_kl
from shape_of_text.losses import CausalLMAlignmentLoss, JMQLoss, MMDLoss

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - depends on installed Transformers version
    AutoModelForImageTextToText = None


class AlignmentTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        alignment_loss: CausalLMAlignmentLoss,
        target_model: torch.nn.Module | None = None,
        kl_eval_batches: int | None = 8,
        kl_vocab_sample_size: int | None = 4096,
        kl_smoothing: float = 0.02,
        mmd_warmup_steps: int = 0,
        jmq_warmup_steps: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.alignment_loss = alignment_loss
        self.target_model = target_model
        self.kl_eval_batches = kl_eval_batches
        self.kl_vocab_sample_size = kl_vocab_sample_size
        self.kl_smoothing = kl_smoothing
        self.mmd_weight = alignment_loss.mmd_weight
        self.jmq_weight = alignment_loss.jmq_weight
        self.mmd_warmup_steps = max(0, mmd_warmup_steps)
        self.jmq_warmup_steps = max(0, jmq_warmup_steps)
        if self.target_model is not None:
            self.target_model.requires_grad_(False)
            self.target_model.eval()
            try:
                self.target_model.to(self.args.device)
            except ValueError:
                # Quantized models may already be pinned to their load device.
                pass

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs.get("labels")
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }
        outputs = model(**model_inputs)

        target_logits = None
        if self.target_model is not None:
            with torch.no_grad():
                target_outputs = self.target_model(**model_inputs)
                target_logits = target_outputs.logits

        self._apply_alignment_schedule()
        loss_output = self.alignment_loss(
            outputs.logits,
            labels,
            target_logits=target_logits,
            attention_mask=inputs.get("attention_mask"),
        )
        return (loss_output.loss, outputs) if return_outputs else loss_output.loss

    def _scheduled_weight(self, base_weight: float, warmup_steps: int) -> float:
        if base_weight == 0 or warmup_steps <= 0:
            return base_weight
        progress = min(1.0, (self.state.global_step + 1) / warmup_steps)
        return base_weight * progress

    def _apply_alignment_schedule(self) -> None:
        self.alignment_loss.mmd_weight = self._scheduled_weight(
            self.mmd_weight, self.mmd_warmup_steps
        )
        self.alignment_loss.jmq_weight = self._scheduled_weight(
            self.jmq_weight, self.jmq_warmup_steps
        )

    def evaluate(
        self,
        eval_dataset: Any | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            return metrics

        dataloader = self.get_eval_dataloader(dataset)
        kl_metrics = evaluate_kl(
            self.model,
            dataloader,
            target_model=self.target_model,
            max_batches=self.kl_eval_batches,
            smoothing=self.kl_smoothing,
            vocab_sample_size=self.kl_vocab_sample_size,
        )
        prefixed = {f"{metric_key_prefix}_{key}": value for key, value in kl_metrics.items()}
        metrics.update(prefixed)
        self.log(prefixed)
        return metrics


class CausalLMCollator:
    def __init__(self, tokenizer: AutoTokenizer, *, label_pad_token_id: int = -100) -> None:
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [
            {
                "input_ids": example["input_ids"],
                "attention_mask": example.get("attention_mask", [1] * len(example["input_ids"])),
            }
            for example in examples
        ]
        batch = self.tokenizer.pad(input_features, padding=True, return_tensors="pt")

        max_length = batch["input_ids"].size(1)
        labels = []
        for example in examples:
            label_values = list(example["labels"])
            pad_length = max_length - len(label_values)
            label_padding = [self.label_pad_token_id] * pad_length
            if self.tokenizer.padding_side == "left":
                labels.append(label_padding + label_values)
            else:
                labels.append(label_values + label_padding)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma style distribution fine-tuning")
    parser.add_argument("--model-id", default="google/gemma-4-12B")
    parser.add_argument(
        "--model-class",
        choices=("image-text-to-text", "causal-lm"),
        default="image-text-to-text",
    )
    parser.add_argument("--target-model-id", default=None)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument(
        "--dataset-format",
        choices=("text", "instruction-jsonl"),
        default="text",
    )
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--completion-field", default="completion")
    parser.add_argument("--mask-prompt-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="runs/gemma4-12b-style-alignment")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--mmd-weight", type=float, default=0.05)
    parser.add_argument("--jmq-weight", type=float, default=0.05)
    parser.add_argument("--mmd-warmup-steps", type=int, default=0)
    parser.add_argument("--jmq-warmup-steps", type=int, default=0)
    parser.add_argument("--target-smoothing", type=float, default=0.02)
    parser.add_argument("--mmd-max-positions", type=int, default=128)
    parser.add_argument("--jmq-max-positions", type=int, default=256)
    parser.add_argument("--mmd-vocab-sample-size", type=int, default=2048)
    parser.add_argument("--jmq-vocab-sample-size", type=int, default=4096)
    parser.add_argument("--kl-eval-batches", type=int, default=8)
    parser.add_argument("--kl-vocab-sample-size", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--fsdp", default="")
    parser.add_argument("--fsdp-config", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--hub-private-repo", action="store_true")
    parser.add_argument("--hub-token", default=None)
    parser.add_argument("--hub-commit-message", default="Train social-post LoRA adapter")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    return parser.parse_args()


def load_tokenized_dataset(tokenizer: AutoTokenizer, args: argparse.Namespace):
    if args.dataset_format == "instruction-jsonl":
        return load_instruction_dataset(tokenizer, args)
    return load_text_dataset(tokenizer, args)


def load_text_dataset(tokenizer: AutoTokenizer, args: argparse.Namespace):
    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file
    raw = load_dataset("text", data_files=data_files)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], add_special_tokens=True)

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names)

    def group_texts(examples: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated = {key: list(chain(*examples[key])) for key in examples}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // args.max_length) * args.max_length
        result = {
            key: [
                tokens[index : index + args.max_length]
                for index in range(0, total_length, args.max_length)
            ]
            for key, tokens in concatenated.items()
        }
        result["labels"] = [tokens.copy() for tokens in result["input_ids"]]
        return result

    return tokenized.map(group_texts, batched=True)


def load_instruction_dataset(tokenizer: AutoTokenizer, args: argparse.Namespace):
    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file
    raw = load_dataset("json", data_files=data_files)

    def tokenize(example: dict[str, Any]) -> dict[str, list[int]]:
        prompt = str(example[args.prompt_field]).strip()
        completion = str(example[args.completion_field]).strip()
        if not prompt or not completion:
            raise ValueError("instruction examples require non-empty prompt and completion")

        prompt_text = prompt.rstrip() + "\n\n"
        completion_text = completion.strip() + tokenizer.eos_token
        prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
        completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
        completion_budget = min(len(completion_ids), args.max_length)
        prompt_budget = args.max_length - completion_budget
        prompt_ids = prompt_ids[:prompt_budget]
        completion_ids = completion_ids[:completion_budget]
        input_ids = prompt_ids + completion_ids
        prompt_len = len(prompt_ids)

        if args.mask_prompt_labels:
            labels = [-100] * prompt_len + input_ids[prompt_len:]
        else:
            labels = input_ids.copy()

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels[: len(input_ids)],
        }

    columns = raw["train"].column_names
    return raw.map(tokenize, remove_columns=columns)


def quantization_config(enabled: bool) -> BitsAndBytesConfig | None:
    if not enabled:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )


def torch_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.torch_dtype == "bfloat16":
        return torch.bfloat16
    if args.torch_dtype == "float16":
        return torch.float16
    if args.torch_dtype == "float32":
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16 if args.bf16 else torch.float16
    return torch.float32


def auto_model_class(args: argparse.Namespace):
    if args.model_class == "causal-lm":
        return AutoModelForCausalLM
    if AutoModelForImageTextToText is None:
        raise ImportError(
            "AutoModelForImageTextToText is unavailable. Upgrade transformers "
            "or pass --model-class causal-lm for a causal-LM checkpoint."
        )
    return AutoModelForImageTextToText


def load_trainable_model(args: argparse.Namespace) -> torch.nn.Module:
    q_config = quantization_config(not args.no_4bit)
    model_cls = auto_model_class(args)
    model = model_cls.from_pretrained(
        args.model_id,
        quantization_config=q_config,
        dtype=torch_dtype(args),
        attn_implementation=args.attn_implementation,
        device_map=None,
    )
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=not args.no_gradient_checkpointing
        )
    elif not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules,
    )
    return get_peft_model(model, lora_config)


def load_target_model(args: argparse.Namespace) -> torch.nn.Module | None:
    if not args.target_model_id:
        return None
    model_cls = auto_model_class(args)
    model = model_cls.from_pretrained(
        args.target_model_id,
        quantization_config=quantization_config(not args.no_4bit),
        dtype=torch_dtype(args),
        attn_implementation=args.attn_implementation,
        device_map=None,
    )
    model.requires_grad_(False)
    model.eval()
    return model


def load_fsdp_config(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_training_arguments(args: argparse.Namespace, has_validation: bool) -> TrainingArguments:
    eval_value = "steps" if has_validation else "no"
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "bf16": args.bf16 and torch.cuda.is_available(),
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "eval_strategy": eval_value,
        "evaluation_strategy": eval_value,
        "remove_unused_columns": False,
        "optim": "paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        "fsdp": args.fsdp,
        "fsdp_config": load_fsdp_config(args.fsdp_config),
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id,
        "hub_private_repo": args.hub_private_repo,
        "hub_token": args.hub_token,
        "hub_strategy": "end",
    }
    supported = inspect.signature(TrainingArguments).parameters
    return TrainingArguments(**{key: value for key, value in kwargs.items() if key in supported})


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_tokenized_dataset(tokenizer, args)
    model = load_trainable_model(args)
    target_model = load_target_model(args)

    mmd_loss = MMDLoss(
        max_positions=args.mmd_max_positions,
        vocab_sample_size=args.mmd_vocab_sample_size,
        target_smoothing=args.target_smoothing,
    )
    jmq_loss = JMQLoss(
        max_positions=args.jmq_max_positions,
        vocab_sample_size=args.jmq_vocab_sample_size,
        target_smoothing=args.target_smoothing,
    )
    alignment_loss = CausalLMAlignmentLoss(
        mmd_weight=args.mmd_weight,
        jmq_weight=args.jmq_weight,
        mmd_loss=mmd_loss,
        jmq_loss=jmq_loss,
    )

    training_args = build_training_arguments(args, has_validation="validation" in dataset)

    trainer = AlignmentTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        data_collator=CausalLMCollator(tokenizer),
        alignment_loss=alignment_loss,
        target_model=target_model,
        kl_eval_batches=args.kl_eval_batches,
        kl_vocab_sample_size=args.kl_vocab_sample_size,
        kl_smoothing=args.target_smoothing,
        mmd_warmup_steps=args.mmd_warmup_steps,
        jmq_warmup_steps=args.jmq_warmup_steps,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub(commit_message=args.hub_commit_message)


if __name__ == "__main__":
    main()
