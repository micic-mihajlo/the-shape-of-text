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
    assert payload["args"]["image"] == "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"
    assert payload["args"]["flavor"] == "l40sx1"
    assert "command -v git" in command
    assert "command -v gcc" in command
    assert "build-essential" in command
    assert command.index("command -v git") < command.index("git clone")
    assert "--max-steps 10" in command
    assert "--max-length 512" in command
    assert "--eval-steps 5" in command
    assert "--save-steps 10" in command
    assert "--logging-steps 1" in command
    assert "--lora-r 8" in command
    assert "--lora-alpha 16" in command
    assert "--push-to-hub" not in command
    assert "--adapter-id /workspace/runs/gemma-4-12b-social-post-lora-smoke" in command
    assert "scripts/generate_social_posts.py" in command
    assert "base_posts.jsonl" in command
    assert "adapter_posts.jsonl" in command
    assert "scripts/compare_social_outputs.py" in command
    assert "cp /workspace/adapter_report/README.md" in command
    assert "cp /workspace/comparison_report.json" in command
    assert "scripts/upload_hf_adapter.py" in command
    assert "--token ${HF_TOKEN}" in command
    assert "--create-pr" in command
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


def test_preflight_payload_uses_cpu_and_skips_hub_secret(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_hf_job_payload.py",
            "--git-ref",
            "abc123",
            "--mode",
            "preflight",
            "--hub-model-id",
            "micic-mihajlo/adapter",
            "--train-file",
            "examples/social_instructions/train.jsonl",
            "--eval-file",
            "examples/social_instructions/validation.jsonl",
        ],
    )
    payload = build_payload(parse_args())
    command = "\n".join(payload["args"]["command"])
    assert payload["args"]["flavor"] == "cpu-upgrade"
    assert payload["args"]["timeout"] == "45m"
    assert "secrets" not in payload["args"]
    assert "gcc --version" in command
    assert "python -m shape_of_text.train --help" in command
    assert "scripts/validate_preflight.py" in command
    assert "scripts/build_hf_job_payload.py" in command
    assert "python -m pytest -q" in command
    assert "REMOTE_CPU_PREFLIGHT_OK" in command
