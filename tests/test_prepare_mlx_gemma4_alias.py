import json

from scripts.prepare_mlx_gemma4_alias import prepare_alias


def test_prepare_alias_patches_config_and_symlinks_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "gemma4_unified", "architectures": ["Old"]}),
        encoding="utf-8",
    )
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")

    output = tmp_path / "alias"
    linked = prepare_alias(
        source_dir=source,
        output_dir=output,
        link_mode="symlink",
        force=False,
    )

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "gemma4"
    assert config["architectures"] == ["Gemma4ForConditionalGeneration"]
    assert (output / "tokenizer.json").is_symlink()
    assert linked == ["config.json", "tokenizer.json"]
