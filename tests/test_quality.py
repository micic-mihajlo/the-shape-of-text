from shape_of_text.quality import evaluate_completion_quality, quality_report


def test_quality_rejects_special_token_template_output():
    result = evaluate_completion_quality(
        {
            "prompt": "Write a concise LinkedIn post about a product improvement.",
            "completion": (
                "<image>|# The Problem\n\n"
                "The system is experiencing [describe the specific performance issue].\n"
                "```python\ndef process_data(data):\n    pass\n```"
            ),
        }
    )

    codes = {issue.code for issue in result.issues}
    assert "special_token_leak" in codes
    assert "template_or_code" in codes


def test_quality_rejects_prompt_echo():
    result = evaluate_completion_quality(
        {
            "prompt": "Write a sharp launch post for technical founders about a tiny workflow fix.",
            "completion": (
                "Write a sharp launch post for technical founders about a tiny workflow fix. "
                "The post should be written in English."
            ),
        }
    )

    assert "prompt_echo" in {issue.code for issue in result.issues}


def test_quality_allows_requested_changelog_terms():
    result = evaluate_completion_quality(
        {
            "prompt": (
                "Write a product changelog post about adding prompt-to-output training "
                "examples, KL tracking, and adapter publishing. Keep it plain and specific."
            ),
            "completion": (
                "Small changelog today: the training flow now supports prompt-to-output "
                "examples, KL tracking, and adapter publishing in one path. It makes the "
                "run easier to inspect and gives us a cleaner handoff from experiment to "
                "usable adapter."
            ),
        }
    )

    assert result.ok


def test_quality_rejects_unicode_token_soup():
    result = evaluate_completion_quality(
        {
            "prompt": "Write a concise LinkedIn post about a small product improvement.",
            "completion": (
                "makeSoundлизи williams 獲 setEmail Mayıs 🧸 📎 d اfter "
                "Josie Paston MSBuildTools immunoassay laravel vertx isother "
                "Burkina ISNI Monika pursuance getBlue ⚜ 鮭 micronaut 氨 accidens "
                "FileInputStream Nucleaire InnoDB KEYUP MetaMask ActionMode "
                "wikimedia FullHD CallOptions hypothalam Serrurier Klinefelter "
                "𒉡 रिक्वायरमेंट கட்டமை ಚೇಂಜೆಸ್ કાર્યવાહી መሳሳይ脱毛"
            ),
        }
    )

    codes = {issue.code for issue in result.issues}
    assert "low_english_signal" in codes
    assert "non_latin_noise" in codes


def test_quality_accepts_concrete_social_post():
    result = evaluate_completion_quality(
        {
            "prompt": "Write a LinkedIn post about a small product improvement.",
            "completion": (
                "Small product update: the export flow now remembers your last settings. "
                "It is not flashy, but it removes three repeated clicks from a task people "
                "run every week. That is the kind of cleanup users actually feel."
            ),
        }
    )

    assert result.ok


def test_quality_accepts_small_amount_of_social_punctuation():
    result = evaluate_completion_quality(
        {
            "prompt": "Write a short post about improving onboarding.",
            "completion": (
                "We cleaned up onboarding today. Nothing dramatic: one fewer decision, "
                "clearer defaults, and a setup path that gets out of the way faster. "
                "Small fix, better first run."
            ),
        }
    )

    assert result.ok


def test_quality_report_summarizes_failures():
    report = quality_report(
        [
            {
                "id": "bad",
                "prompt": "Write a post.",
                "completion": "<image>| Platform: LinkedIn",
            },
            {
                "id": "good",
                "prompt": "Write a post.",
                "completion": (
                    "We shipped a quieter settings page today. The important change is simple: "
                    "the defaults now match how people actually use the product."
                ),
            },
        ]
    )

    assert report["ok"] is False
    assert report["failed"] == 1
    assert report["failures"][0]["id"] == "bad"
