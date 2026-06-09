from scripts.generate_social_posts import (
    control_token_bad_words,
    generation_record,
    generation_stop_token_ids,
    parse_args,
    prompt_text,
)


class FakeGemmaTokenizer:
    eos_token_id = 1
    pad_token_id = 0
    eot_token = "<turn|>"
    image_token = "<|image|>"
    tool_response_token = "<tool_response|>"

    def encode(self, token, add_special_tokens=False):
        return {
            "<turn|>": [106],
            "<|image|>": [258881],
            "<tool_response|>": [51],
            "<|tool_response>": [500, 501],
            "_": [236779],
            "[": [123],
            "]": [124],
            "[Link]": [900],
        }.get(token, [])


def test_prompt_text_includes_platform_audience_and_prompt():
    prompt = prompt_text(
        {
            "platform": "LinkedIn",
            "audience": "technical founders",
            "prompt": "Write about a launch.",
        }
    )
    assert "Platform: LinkedIn" in prompt
    assert "Audience: technical founders" in prompt
    assert "Write about a launch." in prompt
    assert prompt.endswith("\n\n")


def test_prompt_text_appends_required_exact_strings():
    prompt = prompt_text(
        {
            "prompt": "Rewrite this launch note.",
            "required_terms": ["Rivet", "2,300", "AI-generated code"],
        }
    )

    assert "Must include these exact strings: Rivet, 2,300, AI-generated code." in prompt


def test_generation_defaults_are_bounded_for_social_posts(monkeypatch):
    monkeypatch.setattr("sys.argv", ["generate_social_posts.py"])

    args = parse_args()

    assert args.max_new_tokens == 220
    assert args.temperature == 0.0
    assert args.top_p == 0.85
    assert args.repetition_penalty == 1.0
    assert args.no_repeat_ngram_size == 5


def test_generation_allows_gemma_turn_token_as_stop_token():
    tokenizer = FakeGemmaTokenizer()
    stop_ids = generation_stop_token_ids(tokenizer)

    bad_words = control_token_bad_words(tokenizer, allowed_token_ids=set(stop_ids))

    assert stop_ids == [1, 106]
    assert [106] not in bad_words
    assert [258881] in bad_words
    assert [51] in bad_words
    assert [236779] in bad_words
    assert [123] in bad_words
    assert [124] in bad_words
    assert [900] in bad_words


def test_generation_record_carries_rewrite_quality_metadata():
    record = generation_record(
        {
            "id": "rewrite",
            "platform": "LinkedIn",
            "audience": "founders",
            "prompt": "Rewrite this.",
            "required_terms": ["Rivet", "2,300"],
            "avoid_terms": ["small product update"],
        },
        "Finished post.",
    )

    assert record["required_terms"] == ["Rivet", "2,300"]
    assert record["avoid_terms"] == ["small product update"]
