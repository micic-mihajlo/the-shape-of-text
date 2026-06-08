from scripts.generate_social_posts import parse_args, prompt_text


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


def test_generation_defaults_are_bounded_for_social_posts(monkeypatch):
    monkeypatch.setattr("sys.argv", ["generate_social_posts.py"])

    args = parse_args()

    assert args.max_new_tokens == 140
    assert args.repetition_penalty == 1.12
    assert args.no_repeat_ngram_size == 5
