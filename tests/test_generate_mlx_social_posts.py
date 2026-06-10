import pytest

mx = pytest.importorskip("mlx.core")

from scripts.generate_mlx_social_posts import (
    DEFAULT_BAD_PHRASES,
    bad_phrase_token_sequences,
    make_bad_phrase_processor,
    make_no_repeat_ngram_processor,
    render_generation_prompt,
)


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return f"<user>{messages[0]['content']}</user><assistant>"


def test_render_generation_prompt_uses_chat_template():
    rendered = render_generation_prompt(
        FakeTokenizer(),
        "Rewrite this.",
        use_chat_template=True,
        enable_thinking=False,
    )

    assert rendered == "<user>Rewrite this.</user><assistant>"


def test_render_generation_prompt_can_use_raw_prompt():
    rendered = render_generation_prompt(
        FakeTokenizer(),
        "Rewrite this.\n\n",
        use_chat_template=False,
        enable_thinking=False,
    )

    assert rendered == "Rewrite this.\n\n"


class FakeEncodingTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        vocab = {
            "bad": 1,
            " phrase": 2,
            "bad phrase": 3,
            " Bad": 4,
            " phrase": 2,
        }
        return [vocab.get(text, 99)]


def test_bad_phrase_token_sequences_include_boundary_variants():
    sequences = bad_phrase_token_sequences(FakeEncodingTokenizer(), ["bad phrase"])

    assert (3,) in sequences
    assert (99,) in sequences or (4,) in sequences


def test_default_bad_phrases_cover_observed_template_failures():
    assert "the lesson was simple" in DEFAULT_BAD_PHRASES
    assert "technically perfect" in DEFAULT_BAD_PHRASES
    assert "hiding in plain sight" in DEFAULT_BAD_PHRASES
    assert "small change, but it matters" in DEFAULT_BAD_PHRASES
    assert "massive achievement" in DEFAULT_BAD_PHRASES


def test_bad_phrase_processor_blocks_single_token_phrase():
    processor = make_bad_phrase_processor([(3,)])
    logits = mx.zeros((1, 5))

    processed = processor(mx.array([7, 8]), logits)

    assert processed[0, 3].item() == -float("inf")


def test_bad_phrase_processor_blocks_final_token_after_prefix():
    processor = make_bad_phrase_processor([(10, 11, 12)])
    logits = mx.zeros((1, 20))

    processed = processor(mx.array([4, 10, 11]), logits)

    assert processed[0, 12].item() == -float("inf")


def test_no_repeat_ngram_processor_blocks_repeated_ngram():
    processor = make_no_repeat_ngram_processor(3)
    logits = mx.zeros((1, 20))

    processed = processor(mx.array([4, 10, 11, 12, 10, 11]), logits)

    assert processed[0, 12].item() == -float("inf")
