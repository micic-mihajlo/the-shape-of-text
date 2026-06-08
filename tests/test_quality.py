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


def test_quality_rejects_option_menu_output():
    result = evaluate_completion_quality(
        {
            "prompt": "Write one LinkedIn post about a product update.",
            "completion": (
                "Here are three options depending on the tone you want.\n\n"
                "### Option 1: Best for engagement\n"
                "We shipped a smaller settings flow today. It removes a repeated step "
                "and gives teams one less thing to think about before they publish."
            ),
        }
    )

    assert "template_or_code" in {issue.code for issue in result.issues}


def test_quality_rejects_repeated_phrase_loop():
    result = evaluate_completion_quality(
        {
            "prompt": "Write one LinkedIn post about a product update.",
            "completion": (
                "We cleaned up the first-run checklist today. The setup path is shorter, "
                "the defaults are clearer, and new teams get to the useful moment faster. "
                "That is a better default state. That is a better default state. "
                "That is a better default state."
            ),
        }
    )

    assert "repeated_phrase" in {issue.code for issue in result.issues}


def test_quality_rejects_placeholder_and_suffix_artifacts():
    result = evaluate_completion_quality(
        {
            "prompt": "Write one LinkedIn launch post.",
            "completion": (
                "We built [Framework Name] to make fine-tuning runs easier to inspect. "
                "The useful bit is that every adapter gets a small eval report before it "
                "is shared. [Link] #buildinpublic #shipit_with_me_today_tool_launch_post"
            ),
        }
    )

    codes = {issue.code for issue in result.issues}
    assert "placeholder_text" in codes
    assert "artifact_suffix" in codes
    assert "hashtag_artifact" in codes


def test_quality_rejects_plain_underscore_artifacts():
    result = evaluate_completion_quality(
        {
            "prompt": "Write one LinkedIn post about a model training tool.",
            "completion": (
                "We shipped a small fine_tuning workflow update today. It makes the "
                "evaluation handoff easier to inspect before anyone publishes a new "
                "adapter to the hub."
            ),
        }
    )

    assert "artifact_suffix" in {issue.code for issue in result.issues}


def test_quality_rejects_too_many_hashtags():
    result = evaluate_completion_quality(
        {
            "prompt": "Write one concise X post about a lesson learned.",
            "completion": (
                "Lesson learned: run the tiny smoke test before the expensive GPU job. "
                "It catches broken paths while the fix is still cheap. "
                "#MLOps #Engineering #AI #DevOps #LLM #Automation"
            ),
        }
    )

    assert "too_many_hashtags" in {issue.code for issue in result.issues}


def test_quality_rejects_self_correction_restart():
    result = evaluate_completion_quality(
        {
            "prompt": "Write one recap post.",
            "completion": (
                "That is a wrap on the weekend sprint. We got the live sync working and "
                "the next step is cleaning up onboarding. Wait, that was too long. "
                "Let's try again. One more time: shipped the sync, polish is next."
            ),
        }
    )

    assert "self_correction_restart" in {issue.code for issue in result.issues}


def test_quality_rejects_unfinished_tail():
    result = evaluate_completion_quality(
        {
            "prompt": "Write one plain LinkedIn post.",
            "completion": (
                "The useful part of the update is not the implementation detail. It is "
                "that the next action is easier to trust, and the team no longer has to "
                "guess which result came from the"
            ),
        }
    )

    assert "unfinished_tail" in {issue.code for issue in result.issues}


def test_quality_rejects_long_post_without_terminal_punctuation():
    result = evaluate_completion_quality(
        {
            "prompt": "Write one LinkedIn post.",
            "completion": (
                "We shipped a small update to the evaluation flow today. It gives every "
                "adapter a visible quality report before upload, which makes broken "
                "generations easier to catch before anyone tries the model locally. "
                "The main win is that review happens while the context is still fresh "
                "and the fix is still cheap"
            ),
        }
    )

    assert "missing_terminal_punctuation" in {issue.code for issue in result.issues}


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
