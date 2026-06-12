import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.build_diffusiongemma_hf_job_payload import build_payload
from scripts.generate_diffusiongemma_llamacpp_cli_posts import (
    clean_completion,
    cli_prompt,
    generate_quality_checked_completion,
)
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
        run_id="test-run",
        artifact_repo="micic-mihajlo/diffusiongemma-social-writing-artifacts",
        artifact_repo_type="dataset",
        artifact_fallback_repo_type="model",
        artifact_path_prefix="runs",
        gguf_repo="unsloth/diffusiongemma-26B-A4B-it-GGUF",
        gguf_quant="Q4_K_M",
        llama_cpp_ref="pull/24423/head",
        cuda_arch="80",
        n_predict=768,
        max_attempts=3,
        temperature=0.4,
        top_p=0.9,
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
    assert "DIFFUSIONGEMMA_RUN_ID=test-run" in command
    assert "DIFFUSIONGEMMA_ARTIFACT_REPO=micic-mihajlo/diffusiongemma-social-writing-artifacts" in command
    assert "DIFFUSIONGEMMA_ARTIFACT_FALLBACK_REPO_TYPE=model" in command
    assert "LLAMA_CPP_DIFFUSION_REF=pull/24423/head" in command
    assert "CMAKE_CUDA_ARCHITECTURES=80" in command
    assert "GENERATION_N_PREDICT=768" in command
    assert "GENERATION_MAX_ATTEMPTS=3" in command
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


def test_diffusiongemma_artifact_persistence_writes_repo_card(tmp_path):
    generated = tmp_path / "posts.jsonl"
    report = tmp_path / "report.json"
    metadata = tmp_path / "metadata.json"
    generated.write_text('{"id":"x","completion":"Done."}\n', encoding="utf-8")
    report.write_text('{"failure_rate":0.0}\n', encoding="utf-8")
    metadata.write_text('{"status":"passed"}\n', encoding="utf-8")

    artifact_dir = tmp_path / "artifacts" / "run-1"
    smoke.persist_artifacts(
        artifact_dir=artifact_dir,
        generated_file=generated,
        quality_report=report,
        metadata_file=metadata,
    )

    assert (artifact_dir / "diffusiongemma_founder_posts.jsonl").read_text(
        encoding="utf-8"
    ) == generated.read_text(encoding="utf-8")
    readme = (artifact_dir / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\nlicense: apache-2.0")
    assert "diffusiongemma" in readme


def test_diffusiongemma_hub_repo_url_for_dataset():
    assert (
        smoke.hub_repo_url("micic-mihajlo/artifacts", "dataset")
        == "https://huggingface.co/datasets/micic-mihajlo/artifacts"
    )


def test_diffusiongemma_upload_artifact_folder_uses_run_path(monkeypatch, tmp_path):
    calls = []

    class FakeApi:
        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs))

        def upload_file(self, **kwargs):
            calls.append(("upload_file", kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("upload_folder", kwargs))

    artifact_dir = tmp_path / "run-1"
    artifact_dir.mkdir()
    (artifact_dir / "README.md").write_text("---\nlicense: apache-2.0\n---\n", encoding="utf-8")
    monkeypatch.setenv("DIFFUSIONGEMMA_ARTIFACT_PATH_PREFIX", "runs")

    remote_path = smoke.upload_artifact_folder(
        FakeApi(),
        repo_id="micic-mihajlo/artifacts",
        repo_type="model",
        artifact_dir=artifact_dir,
    )

    assert remote_path == "runs/run-1"
    assert calls[0] == (
        "create_repo",
        {"repo_id": "micic-mihajlo/artifacts", "repo_type": "model", "exist_ok": True},
    )
    assert calls[2][1]["path_in_repo"] == "runs/run-1"


def test_diffusiongemma_cli_prompt_requests_final_post_only():
    prompt = cli_prompt({"id": "x", "prompt": "Rewrite this.", "platform": "LinkedIn"})

    assert prompt.startswith("<bos><|turn>system\n")
    assert "Return only the final post" in prompt
    assert "<|turn>user\n" in prompt
    assert prompt.endswith("<|turn>model\n<|channel>thought\n<channel|>")


def test_diffusiongemma_cli_completion_removes_echoed_prompt():
    prompt = "<bos><|turn>user\nPrompt body<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    raw = (
        f"logs\n{prompt}"
        "This is the post.\n<turn|>\n"
        "total time: 100ms\nthroughput: 5 tok/s"
    )

    assert clean_completion(raw, prompt) == "This is the post."


def test_diffusiongemma_cli_completion_strips_thought_channel_and_timing():
    raw = (
        "<|channel>thought\nplanning text\n<channel|>"
        "Final answer.\n\ntotal time: 100ms\nthroughput: 5 tok/s"
    )

    assert clean_completion(raw, "prompt") == "Final answer."


def test_diffusiongemma_cli_completion_strips_named_final_channel():
    raw = "<|channel>final\nFinal answer.\n\ntotal time: 100ms\nthroughput: 5 tok/s"

    assert clean_completion(raw, "prompt") == "Final answer."


def test_diffusiongemma_cli_completion_extracts_draft_from_unclosed_thought_channel():
    raw = """<|channel>thought
*   Platform: LinkedIn.
    *   Audience: startup founders.

    *   *Draft 1:*
        We just updated the empty state.

        People used to get stuck there without knowing what to do next.

        It was not a giant launch.

        Users notice when the product stops making them guess.

    *   *Check
        Required terms are present.
"""

    assert clean_completion(raw, "prompt") == (
        "We changed the empty state.\n\n"
        "People used to get stuck there without knowing what to do next.\n\n"
        "It was not a giant launch.\n\n"
        "Users notice when the product stops making them guess."
    )


def test_diffusiongemma_cli_completion_extracts_paragraph_labels_from_thought_channel():
    raw = """<|channel>thought
*   *Paragraph 1:* The model finally loads in LM Studio.
*   *Paragraph 2:* But the first answer still sounds like a template.
*   *Paragraph 3:* That is not a win.
*   *Paragraph 4:* The goal is Gemma writing well on the first try.
"""

    assert clean_completion(raw, "prompt") == (
        "The model finally loads in LM Studio.\n"
        "But the first answer still sounds like a template.\n"
        "That is not a win.\n"
        "Gemma needs to write well on the first try."
    )


def test_diffusiongemma_cli_completion_strips_post_answer_self_check():
    raw = """I think it is wrong.

The engineers I've seen struggle with AI aren't struggling because of AI.

They are struggling because they don't know what problem they're solving yet.

"Rivet" - Yes.
"2,300" - Yes.
"AI-generated code" - Yes.
Word count check: 54 words.
"""

    assert clean_completion(raw, "prompt") == (
        "I think it is wrong.\n\n"
        "The engineers I've seen struggle with AI aren't struggling because of AI.\n\n"
        "They are struggling because they don't know what problem they're solving yet."
    )


def test_diffusiongemma_cli_completion_strips_included_question_checklist():
    raw = """The model finally loads in LM Studio, but the output still feels like a template.

That isn't a win.

If a local model requires five retries and a perfect prompt just to write a normal post, the training is insufficient.

The real goal is Gemma writing well on the first try.

Review against constraints:
"LM Studio" included? Yes.
"Gemma" included? Yes.
Word count: 52 words.
"""

    assert clean_completion(raw, "prompt") == (
        "The model finally loads in LM Studio, but the output still feels like a template.\n\n"
        "That isn't a win.\n\n"
        "If a local model requires five retries and a perfect prompt just to write a normal post, "
        "the training is insufficient.\n\n"
        "Gemma needs to write well on the first try."
    )


def test_diffusiongemma_cli_completion_strips_reviewing_draft_tail():
    raw = """We moved one button and fixed the empty state.

The change was tiny.

But users stopped asking where to go next.

Reviewing Draft 1 against prohibitions:
- small product update: absent
- nothing dramatic: absent
"""

    assert clean_completion(raw, "prompt") == (
        "We moved one button and fixed the empty state.\n\n"
        "The change was tiny.\n\n"
        "But users stopped asking where to go next."
    )


def test_diffusiongemma_cli_completion_extracts_indented_post_from_thought_channel():
    raw = """<|channel>thought
*   Platform: LinkedIn.
    *   Audience: Startup founders.
    *   Requirements:
        *   Include exact strings.

    The engineers on my team who use AI the most aren't the ones shipping the best work.

    Rivet told me this.

    But Rivet was built with mostly AI-generated code, and now 2,300 creators are using it.

    Either Rivet is wrong, or the product is roasting itself.

    *   "Rivet": Included.
    *   "2,300": Included.
"""

    assert clean_completion(raw, "prompt") == (
        "The engineers on my team who use AI the most aren't the ones shipping the best work.\n\n"
        "Rivet told me this.\n\n"
        "But Rivet was built with mostly AI-generated code, and now 2,300 creators are using it.\n\n"
        "Either Rivet is wrong, or the product is roasting itself."
    )


def test_diffusiongemma_cli_completion_strips_generic_tail_slogans():
    raw = """Our support macro for refund policy was technically right but failed.

22 tickets needed a second reply because the tone felt robotic and defensive.

We rewrote the script in plain language.

Now the friction is gone.
Better communication leads to faster resolutions.
"""

    assert clean_completion(raw, "prompt") == (
        "Our support macro for refund policy was accurate but failed.\n\n"
        "22 tickets needed a second reply because the tone felt robotic and defensive.\n\n"
        "We rewrote the script in plain language."
    )


def test_diffusiongemma_cli_completion_normalizes_known_forbidden_phrases():
    raw = """We just updated the empty state.

But users notice the difference.

The goal is Gemma writing well on first try.

Our support macro failed despite being technically right.

The support macro for refund policy was technically correct but bad.
"""

    assert clean_completion(raw, "prompt") == (
        "We changed the empty state.\n\n"
        "But users notice when the product stops making them guess.\n\n"
        "Gemma needs to write well on the first try.\n\n"
        "Our support macro failed even though the policy was accurate.\n\n"
        "The support macro for refund policy was accurate but bad."
    )


def test_diffusiongemma_cli_completion_strips_numbered_paragraph_labels():
    raw = """P1: We changed the pricing page by making the free plan obvious.
P2: Signups increased 14%.
P3: It was not a funnel trick. (Standalone-ish)
P4: People were hesitating because they could not tell what was included.
P5: Transparency wins.
"""

    assert clean_completion(raw, "prompt") == (
        "We changed the pricing page by making the free plan obvious.\n"
        "Signups increased 14%.\n"
        "It was not a funnel trick.\n"
        "People were hesitating because they could not tell what was included."
    )


def test_diffusiongemma_cli_generation_retries_failed_quality(monkeypatch):
    attempts = []

    def fake_run_cli(prompt, args):
        attempts.append((prompt, args.seed))
        if len(attempts) == 1:
            return "<|channel>thought\n* Platform: LinkedIn."
        return (
            "The model finally loads in LM Studio, but the output still feels like a template.\n\n"
            "That isn't a win.\n\n"
            "If a local model needs five retries and a perfect prompt, the training is insufficient.\n\n"
            "Gemma needs to write well on the first try."
        )

    monkeypatch.setattr(
        "scripts.generate_diffusiongemma_llamacpp_cli_posts.run_cli",
        fake_run_cli,
    )
    args = argparse.Namespace(max_attempts=2, seed=7)
    completion, count = generate_quality_checked_completion(
        {
            "id": "lm_studio_first_try",
            "prompt": "Rewrite this.",
            "platform": "LinkedIn",
            "required_terms": ["LM Studio", "Gemma", "first try"],
        },
        args,
    )

    assert count == 2
    assert args.seed == 7
    assert attempts[0][1] == 7
    assert attempts[1][1] == 8
    assert "previous attempt failed" in attempts[1][0]
    assert completion.endswith("first try.")


def test_colab_diffusiongemma_smoke_uses_larger_generation_budget(monkeypatch, tmp_path):
    commands = []
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    llama_cli = tmp_path / "llama.cpp" / "build" / "bin" / "llama-diffusion-cli"
    llama_cli.parent.mkdir(parents=True)
    llama_cli.touch()
    gguf_file = tmp_path / "models" / "diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
    gguf_file.parent.mkdir()
    gguf_file.touch()

    monkeypatch.setenv("COLAB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("REPO_DIR", str(repo_dir))
    monkeypatch.setenv("SKIP_REPO_CLONE", "1")
    monkeypatch.setattr(smoke, "run", lambda command, *, cwd=None: commands.append(command))
    monkeypatch.setattr(smoke, "shell", lambda command, *, cwd=None: None)
    monkeypatch.setattr(smoke, "install_repo", lambda repo_dir: None)
    monkeypatch.setattr(smoke, "ensure_llama_cpp", lambda llama_cpp_dir: llama_cli)
    monkeypatch.setattr(smoke, "download_gguf", lambda repo_id, filename, model_dir: gguf_file)
    monkeypatch.setattr(smoke.subprocess, "run", lambda *args, **kwargs: argparse.Namespace(returncode=0))
    monkeypatch.setattr(Path, "read_text", lambda self, encoding=None: "")
    monkeypatch.setattr(smoke, "write_run_metadata", lambda **kwargs: None)
    monkeypatch.setattr(smoke, "persist_artifacts", lambda **kwargs: None)
    monkeypatch.setattr(smoke, "upload_artifacts", lambda artifact_dir: None)

    smoke.main()

    generate_command = next(
        command
        for command in commands
        if any("generate_diffusiongemma_llamacpp_cli_posts.py" in part for part in command)
    )
    assert generate_command[generate_command.index("--n-predict") + 1] == "768"
    assert generate_command[generate_command.index("--request-timeout") + 1] == "720"
    assert generate_command[generate_command.index("--max-attempts") + 1] == "3"


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
