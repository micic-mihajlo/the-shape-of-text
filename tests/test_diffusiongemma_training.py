import argparse

from scripts.build_diffusiongemma_inference_hf_job_payload import build_payload as build_inference_payload
from scripts.build_diffusiongemma_training_hf_job_payload import build_payload, hf_cli_command
from scripts.run_hf_diffusiongemma_training import (
    adapter_card,
    chat_messages,
    clean_generated_text,
    eval_hard_case_rows,
    natural_augmented_rows,
    natural_prompt_from_row,
    record_from_row,
    repeated_hard_case_rows,
)
from shape_of_text.quality import founder_rewrite_quality_report


def test_diffusiongemma_training_record_builds_chat_messages():
    record = record_from_row(
        {
            "id": "sample",
            "prompt": "Rewrite this.",
            "completion": "Final post.",
        },
        fallback_id="row-0",
    )

    assert record.record_id == "sample"
    assert chat_messages(record) == [
        {"role": "user", "content": "Rewrite this."},
        {"role": "assistant", "content": "Final post."},
    ]


def test_diffusiongemma_training_record_rejects_empty_completion():
    try:
        record_from_row({"prompt": "Rewrite this.", "completion": " "}, fallback_id="bad")
    except ValueError as error:
        assert "prompt and completion" in str(error)
    else:
        raise AssertionError("Expected empty completion to fail")


def test_clean_generated_text_removes_prompt_and_channels():
    prompt = "Rewrite the rough draft."
    raw = (
        "Rewrite the rough draft.\n"
        "<|channel>final\n"
        "Prompt: ignored\n"
        "This is the post.\n"
        "<turn|>"
    )

    assert clean_generated_text(raw, prompt) == "This is the post."


def test_clean_generated_text_accepts_decoded_list():
    assert clean_generated_text(["First line.", "Second line."], "prompt") == "First line.\nSecond line."


def test_clean_generated_text_strips_leading_channel_name():
    assert clean_generated_text("thought\nThis is the post.", "prompt") == "This is the post."


def test_adapter_card_has_hub_yaml_metadata():
    card = adapter_card(
        base_model="unsloth/diffusiongemma-26B-A4B-it",
        hub_model_id="micic-mihajlo/diffusiongemma-social-writer-lora",
        train_examples=100,
        skipped_examples=2,
        max_steps=160,
        lora_rank=32,
    )

    assert card.startswith("---\nlicense: gemma\n")
    assert "base_model: unsloth/diffusiongemma-26B-A4B-it" in card
    assert "block-diffusion" in card
    assert "Train examples used: 100" in card


def test_natural_prompt_from_row_removes_scaffolding_but_keeps_source_draft():
    prompt = natural_prompt_from_row(
        {
            "platform": "LinkedIn",
            "source_draft": "The model loads in LM Studio but needs five retries.",
        }
    )

    assert prompt is not None
    assert "LM Studio" in prompt
    assert "Must include these exact strings" not in prompt
    assert "Avoid these phrases" not in prompt
    assert "Return only one finished post" in prompt


def test_natural_augmented_rows_adds_realistic_prompt_variant():
    rows = [
        {
            "id": "x",
            "prompt": "Original scaffolded prompt",
            "completion": "Final post.",
            "source_draft": "Rough source.",
        }
    ]

    augmented = natural_augmented_rows(rows)

    assert len(augmented) == 2
    assert augmented[0]["prompt"] == "Original scaffolded prompt"
    assert augmented[1]["id"] == "x__natural_prompt"
    assert "Rough source." in augmented[1]["prompt"]
    assert augmented[1]["completion"] == "Final post."


def test_eval_hard_case_rows_pass_founder_rewrite_quality():
    rows = [
        {
            "id": "lm_studio_first_try",
            "prompt": "Rewrite this rough post.",
            "required_terms": ["LM Studio", "Gemma", "first try"],
            "avoid_terms": ["small product update", "not flashy"],
        },
        {
            "id": "unknown",
            "prompt": "No hard case.",
            "required_terms": [],
        },
    ]

    hard_cases = eval_hard_case_rows(rows)
    report = founder_rewrite_quality_report(hard_cases)

    assert [row["id"] for row in hard_cases] == ["lm_studio_first_try__hard_case"]
    assert report["ok"] is True


def test_repeated_hard_case_rows_weights_failures_without_changing_completion():
    rows = [{"id": "refund_macro", "prompt": "Rewrite.", "required_terms": ["refund policy"]}]

    hard_cases = repeated_hard_case_rows(rows, repeat=3)

    assert len(hard_cases) == 3
    assert [row["id"] for row in hard_cases] == [
        "refund_macro__hard_case__repeat_1",
        "refund_macro__hard_case__repeat_2",
        "refund_macro__hard_case__repeat_3",
    ]
    assert len({row["completion"] for row in hard_cases}) == 1


def test_diffusiongemma_training_payload_runs_remote_a100_job():
    args = argparse.Namespace(
        repo_url="https://github.com/micic-mihajlo/the-shape-of-text.git",
        git_ref="abc123",
        image="nvidia/cuda:12.8.0-devel-ubuntu22.04",
        flavor="a100-large",
        timeout="4h",
        detach=True,
        volume=[],
        hub_model_id="micic-mihajlo/diffusiongemma-social-writer-lora",
        max_steps=160,
        grad_accum=4,
        learning_rate=1e-4,
        lora_r=32,
        lora_alpha=64,
        eval_limit=10,
        max_denoising_steps=32,
        include_eval_hard_cases=True,
        hard_case_repeat=4,
        min_free_gb=50.0,
    )

    payload = build_payload(args)
    command = "\n".join(payload["args"]["command"])

    assert payload["operation"] == "run"
    assert payload["args"]["flavor"] == "a100-large"
    assert payload["args"]["secrets"] == {"HF_TOKEN": "$HF_TOKEN"}
    assert "UNSLOTH_RETURN_LOGITS=1" in command
    assert "pip install unsloth" in command
    assert "git checkout abc123" in command
    assert "scripts/run_hf_diffusiongemma_training.py" in command
    assert "--hub-model-id micic-mihajlo/diffusiongemma-social-writer-lora" in command
    assert "--max-steps 160" in command
    assert "--learning-rate 0.0001" in command
    assert "--max-denoising-steps 32" in command
    assert "--include-eval-hard-cases" in command
    assert "--hard-case-repeat 4" in command
    assert "--min-free-gb 50.0" in command


def test_diffusiongemma_training_cli_command_uses_secret_flag():
    args = argparse.Namespace(
        repo_url="https://github.com/micic-mihajlo/the-shape-of-text.git",
        git_ref="abc123",
        image="nvidia/cuda:12.8.0-devel-ubuntu22.04",
        flavor="a100-large",
        timeout="4h",
        detach=True,
        volume=[],
        hub_model_id="micic-mihajlo/diffusiongemma-social-writer-lora",
        max_steps=160,
        grad_accum=4,
        learning_rate=1e-4,
        lora_r=32,
        lora_alpha=64,
        eval_limit=10,
        max_denoising_steps=32,
        include_eval_hard_cases=False,
        hard_case_repeat=1,
        min_free_gb=50.0,
        cli=True,
    )

    command = hf_cli_command(args)

    assert command.startswith("hf jobs run --detach --flavor a100-large")
    assert "--secrets HF_TOKEN" in command
    assert "nvidia/cuda:12.8.0-devel-ubuntu22.04 -- /bin/bash -lc" in command


def test_diffusiongemma_inference_payload_loads_existing_adapter_without_training():
    args = argparse.Namespace(
        repo_url="https://github.com/micic-mihajlo/the-shape-of-text.git",
        git_ref="abc123",
        image="nvidia/cuda:12.8.0-devel-ubuntu22.04",
        flavor="a100-large",
        timeout="90m",
        detach=True,
        volume=[],
        adapter_id="micic-mihajlo/diffusiongemma-social-writer-lora",
        eval_limit=3,
        max_denoising_steps=32,
        max_new_tokens=256,
        min_free_gb=50.0,
        artifact_repo="micic-mihajlo/diffusiongemma-social-writer-lora",
        artifact_repo_type="model",
        artifact_path_prefix="inference-runs",
        run_id="smoke-1",
        prompt="Rewrite this post.",
    )

    payload = build_inference_payload(args)
    command = "\n".join(payload["args"]["command"])

    assert payload["operation"] == "run"
    assert payload["args"]["flavor"] == "a100-large"
    assert payload["args"]["secrets"] == {"HF_TOKEN": "$HF_TOKEN"}
    assert "git checkout abc123" in command
    assert "scripts/run_hf_diffusiongemma_inference.py" in command
    assert "scripts/run_hf_diffusiongemma_training.py" not in command
    assert "--adapter-id micic-mihajlo/diffusiongemma-social-writer-lora" in command
    assert "--artifact-path-prefix inference-runs" in command
    assert "--run-id smoke-1" in command
    assert "--prompt 'Rewrite this post.'" in command
