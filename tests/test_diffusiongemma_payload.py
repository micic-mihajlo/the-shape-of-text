import argparse
import json
from pathlib import Path

from scripts.build_diffusiongemma_hf_job_payload import build_payload
from scripts.generate_diffusiongemma_llamacpp_posts import chat_payload, completion_from_response
import scripts.run_colab_diffusiongemma_smoke as smoke


def test_diffusiongemma_hf_payload_runs_remote_smoke():
    args = argparse.Namespace(
        repo_url="https://github.com/micic-mihajlo/the-shape-of-text.git",
        git_ref="abc123",
        image="nvidia/cuda:12.5.1-devel-ubuntu22.04",
        flavor="l40sx1",
        timeout="2h",
        detach=True,
        volume=[],
        gguf_repo="unsloth/diffusiongemma-26B-A4B-it-GGUF",
        gguf_quant="Q4_K_M",
    )

    payload = build_payload(args)
    command = "\n".join(payload["args"]["command"])

    assert payload["operation"] == "run"
    assert payload["args"]["image"] == "nvidia/cuda:12.5.1-devel-ubuntu22.04"
    assert payload["args"]["flavor"] == "l40sx1"
    assert payload["args"]["detach"] is True
    assert payload["args"]["secrets"] == {"HF_TOKEN": "$HF_TOKEN"}
    assert "git checkout abc123" in command
    assert "libssl-dev" in command
    assert "REQUIRE_CUDA=1" in command
    assert "unsloth/diffusiongemma-26B-A4B-it-GGUF" in command
    assert "python3 scripts/run_colab_diffusiongemma_smoke.py" in command


def test_llamacpp_chat_payload_is_founder_rewrite_only():
    args = argparse.Namespace(model="diffusiongemma", max_tokens=220, temperature=0.4, top_p=0.9)

    payload = chat_payload("Rewrite this founder post.", args)

    assert payload["model"] == "diffusiongemma"
    assert payload["messages"][0]["role"] == "system"
    assert "Return only the final post" in payload["messages"][0]["content"]
    assert payload["messages"][1]["content"] == "Rewrite this founder post."
    assert payload["temperature"] == 0.4


def test_completion_from_openai_compatible_response():
    response = {"choices": [{"message": {"content": "  Done.  "}}]}

    assert completion_from_response(response) == "Done."


def test_llamacpp_build_targets_a100_cuda_arch_by_default(monkeypatch, tmp_path):
    commands = []
    llama_dir = tmp_path / "llama.cpp"
    server = llama_dir / "build" / "bin" / "llama-server"
    server.parent.mkdir(parents=True)
    server.touch()

    def fake_run(command, *, cwd=None):
        commands.append(command)

    monkeypatch.setattr(smoke.shutil, "which", lambda name: "/usr/local/cuda/bin/nvcc")
    monkeypatch.setattr(smoke, "run", fake_run)

    assert smoke.ensure_llama_cpp(Path(llama_dir)) == server

    cmake_configure = commands[0]
    assert "-DCMAKE_CUDA_ARCHITECTURES=80" in cmake_configure
    assert "-DGGML_CUDA_ARCHITECTURES=native" not in cmake_configure
