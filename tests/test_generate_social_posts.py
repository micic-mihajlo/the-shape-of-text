from scripts.generate_social_posts import prompt_text


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
