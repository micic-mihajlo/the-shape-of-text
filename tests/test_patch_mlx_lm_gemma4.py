from scripts.patch_mlx_lm_gemma4 import (
    patch_gemma4_sanitizer,
    patch_no_thinking_dataset,
)


def test_patch_gemma4_sanitizer_adds_vision_embedder(tmp_path):
    path = tmp_path / "gemma4.py"
    path.write_text(
        'if k.startswith((\n                    "embed_vision",\n                )):\n',
        encoding="utf-8",
    )

    changed = patch_gemma4_sanitizer(path, dry_run=False)

    assert changed is True
    assert '"vision_embedder",' in path.read_text(encoding="utf-8")


def test_patch_no_thinking_dataset_adds_template_kwarg(tmp_path):
    path = tmp_path / "datasets.py"
    path.write_text(
        "tokens = self.tokenizer.apply_chat_template(\n"
        "    messages,\n"
        "    return_dict=False,\n"
        ")\n",
        encoding="utf-8",
    )

    changed = patch_no_thinking_dataset(path, dry_run=False)

    assert changed is True
    text = path.read_text(encoding="utf-8")
    assert "enable_thinking=False," in text


def test_patch_no_thinking_dataset_handles_single_line_call(tmp_path):
    path = tmp_path / "datasets.py"
    path.write_text(
        "tokens = self.tokenizer.apply_chat_template(messages, return_dict=False)\n",
        encoding="utf-8",
    )

    changed = patch_no_thinking_dataset(path, dry_run=False)

    assert changed is True
    assert "return_dict=False, enable_thinking=False)" in path.read_text(
        encoding="utf-8"
    )
