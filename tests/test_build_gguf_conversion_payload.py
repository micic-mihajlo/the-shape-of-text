from scripts.build_gguf_conversion_payload import build_payload, parse_args


def test_gguf_conversion_payload_merges_quantizes_and_uploads(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_gguf_conversion_payload.py",
            "--adapter-model",
            "micic-mihajlo/adapter",
            "--adapter-revision",
            "abc123",
            "--hub-model-id",
            "micic-mihajlo/adapter-GGUF",
            "--detach",
        ],
    )

    payload = build_payload(parse_args())
    command = "\n".join(payload["args"]["command"])

    assert payload["operation"] == "run"
    assert payload["args"]["secrets"] == {"HF_TOKEN": "$HF_TOKEN"}
    assert "PeftModel.from_pretrained" in command
    assert "revision=adapter_revision" in command
    assert '"adapter_revision": adapter_revision' in command
    assert "abc123" in command
    assert "merge_and_unload" in command
    assert "language_model" in command
    assert 'text_model._tied_weights_keys = {"lm_head.weight": "embed_tokens.weight"}' in command
    assert "/workspace/llama.cpp/requirements.txt" not in command
    assert "requirements-convert_legacy_llama.txt" in command
    assert "convert_hf_to_gguf.py" in command
    assert "llama-quantize" in command
    assert "Q4_K_M" in command
    assert "GGUF_UPLOAD_OK" in command
    assert payload["args"]["detach"] is True
