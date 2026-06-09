from scripts.generate_mlx_social_posts import render_generation_prompt


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<user>{messages[0]['content']}</user><assistant>"


def test_render_generation_prompt_uses_chat_template():
    rendered = render_generation_prompt(
        FakeTokenizer(),
        "Rewrite this.",
        use_chat_template=True,
    )

    assert rendered == "<user>Rewrite this.</user><assistant>"


def test_render_generation_prompt_can_use_raw_prompt():
    rendered = render_generation_prompt(
        FakeTokenizer(),
        "Rewrite this.\n\n",
        use_chat_template=False,
    )

    assert rendered == "Rewrite this.\n\n"
