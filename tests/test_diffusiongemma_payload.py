import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.build_diffusiongemma_hf_job_payload import build_payload
from scripts.generate_diffusiongemma_llamacpp_cli_posts import clean_completion, cli_prompt
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
    cli = llama_dir / "build" / "bin" / "llama-diffusion-cli"
    cli.parent.mkdir(parents=True)
    cli.touch()

    def fake_run(command, *, cwd=None):
        commands.append(command)

    monkeypatch.setattr(smoke.shutil, "which", lambda name: "/usr/local/cuda/bin/nvcc")
    monkeypatch.setattr(smoke, "run", fake_run)

    assert smoke.ensure_llama_cpp(Path(llama_dir)) == cli

    assert ["git", "fetch", "origin", "pull/24423/head:diffusiongemma"] in commands
    assert ["git", "checkout", "diffusiongemma"] in commands
    cmake_configure = next(command for command in commands if command[0] == "cmake")
    assert "-DCMAKE_CUDA_ARCHITECTURES=80" in cmake_configure
    assert "-DGGML_CUDA_ARCHITECTURES=native" not in cmake_configure
    assert ["cmake", "--build", "build", "--target", "llama-diffusion-cli", "-j"] in commands


def test_diffusiongemma_default_gguf_filename_matches_unsloth_repo():
    assert smoke.default_gguf_filename("Q4_K_M") == "diffusiongemma-26B-A4B-it-Q4_K_M.gguf"


def test_diffusiongemma_cli_prompt_requests_final_post_only():
    prompt = cli_prompt({"id": "x", "prompt": "Rewrite this.", "platform": "LinkedIn"})

    assert "Return only the final post" in prompt
    assert "Final post:" in prompt


def test_diffusiongemma_cli_completion_removes_echoed_prompt():
    prompt = "Prompt body\n\nFinal post:\n"
    raw = f"logs\n{prompt}This is the post."

    assert clean_completion(raw, prompt) == "This is the post."


def test_diffusiongemma_generator_file_entrypoint_imports_from_any_cwd(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/generate_diffusiongemma_llamacpp_posts.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Generate founder/social eval posts" in result.stdout
