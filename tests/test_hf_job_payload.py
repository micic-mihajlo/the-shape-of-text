from pathlib import Path

from scripts.build_hf_job_payload import build_payload, parse_args


def test_hf_job_payload_contains_smoke_training_and_eval_command(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_hf_job_payload.py",
            "--git-ref",
            "abc123",
            "--hub-model-id",
            "micic-mihajlo/adapter",
            "--train-file",
            "data/train.jsonl",
            "--eval-file",
            "data/eval.jsonl",
        ],
    )
    payload = build_payload(parse_args())
    command = "\n".join(payload["args"]["command"])
    assert payload["operation"] == "run"
    assert payload["args"]["flavor"] == "l40sx1"
    assert "command -v git" in command
    assert command.index("command -v git") < command.index("git clone")
    assert "--max-steps 10" in command
    assert "--max-length 512" in command
    assert "--eval-steps 5" in command
    assert "--save-steps 10" in command
    assert "--logging-steps 1" in command
    assert "--lora-r 8" in command
    assert "--lora-alpha 16" in command
    assert "--push-to-hub" in command
    assert "--hub-token ${HF_TOKEN}" in command
    assert "scripts/generate_social_posts.py" in command
    assert "base_posts.jsonl" in command
    assert "adapter_posts.jsonl" in command
    assert "scripts/compare_social_outputs.py" in command
    assert payload["args"]["secrets"] == {"HF_TOKEN": "$HF_TOKEN"}


def test_full_payload_uses_requested_steps(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_hf_job_payload.py",
            "--git-ref",
            "abc123",
            "--mode",
            "full",
            "--hub-model-id",
            "micic-mihajlo/adapter",
            "--max-steps",
            "123",
            "--max-length",
            "768",
        ],
    )
    payload = build_payload(parse_args())
    command = "\n".join(payload["args"]["command"])
    assert "--max-steps 123" in command
    assert "--max-length 768" in command
    assert "--eval-steps 100" in command
    assert "--save-steps 123" in command
