from shape_of_text.quality import (
    evaluate_founder_rewrite_quality,
    founder_rewrite_quality_report,
)


def test_founder_rewrite_rejects_generic_social_slop():
    result = evaluate_founder_rewrite_quality(
        {
            "prompt": "Rewrite this founder post.",
            "completion": (
                "Small product update: we changed the settings page today. It is not "
                "flashy, but it removes friction from the workflow and gives users the "
                "kind of cleanup users actually feel."
            ),
        }
    )

    assert "generic_social_slop" in {issue.code for issue in result.issues}


def test_founder_rewrite_rejects_missing_required_anchor():
    result = evaluate_founder_rewrite_quality(
        {
            "prompt": "Rewrite this post and keep Rivet.",
            "required_terms": ["Rivet", "2,300"],
            "completion": (
                "The engineers using AI the most are not automatically shipping the best work.\n\n"
                "That is the uncomfortable part.\n\n"
                "The tool can make unclear thinking louder. It can also make clear thinking "
                "move faster. The difference is not the model. The difference is whether "
                "the person knew what problem they were solving before the model answered."
            ),
        }
    )

    assert "missing_required_anchor" in {issue.code for issue in result.issues}


def test_founder_rewrite_matches_numeric_anchors_without_false_substrings():
    comma_result = evaluate_founder_rewrite_quality(
        {
            "prompt": "Rewrite this post.",
            "required_terms": ["2,300 creators", "14%"],
            "completion": (
                "The creator tool now has 2300 creators using it.\n\n"
                "The pricing page changed too, and signups increased 14% after the "
                "free plan became obvious."
            ),
        }
    )
    false_substring_result = evaluate_founder_rewrite_quality(
        {
            "prompt": "Rewrite this post.",
            "required_terms": ["2,300 creators", "14%"],
            "completion": (
                "The creator tool now has 2310 creators using it.\n\n"
                "The pricing page changed too, and signups increased 114% after the "
                "free plan became obvious."
            ),
        }
    )

    assert "missing_required_anchor" not in {issue.code for issue in comma_result.issues}
    assert "missing_required_anchor" in {
        issue.code for issue in false_substring_result.issues
    }


def test_founder_rewrite_rejects_forbidden_terms_with_avoid_terms():
    result = evaluate_founder_rewrite_quality(
        {
            "prompt": "Rewrite this post.",
            "avoid_terms": ["field report"],
            "forbidden_terms": ["AI the least"],
            "completion": (
                "The engineers using AI the least are shipping the best work.\n\n"
                "That sounds useful, but it reverses the original point and should not "
                "survive the founder rewrite quality gate."
            ),
        }
    )

    assert "forbidden_term" in {issue.code for issue in result.issues}


def test_founder_rewrite_rejects_non_ascii_alpha_leakage():
    result = evaluate_founder_rewrite_quality(
        {
            "prompt": "Rewrite this post.",
            "completion": (
                "The model looked useful until one странный token slipped into the post.\n\n"
                "That is enough to fail the first-try writing bar because the output "
                "still needs manual cleanup."
            ),
        }
    )

    assert "non_ascii_alpha" in {issue.code for issue in result.issues}


def test_founder_rewrite_accepts_specific_rewrite_with_rhythm():
    result = evaluate_founder_rewrite_quality(
        {
            "prompt": "Rewrite this founder post.",
            "required_terms": ["Rivet", "2,300", "AI-generated code"],
            "completion": (
                "The engineers using AI the most aren't automatically shipping the best work.\n\n"
                "Rivet is a funny counterexample.\n\n"
                "It was built with a lot of AI-generated code. Now 2,300 people use it, "
                "which makes the take a little awkward.\n\n"
                "AI does not replace thinking.\n"
                "It exposes the missing parts faster."
            ),
        }
    )

    assert result.ok


def test_founder_rewrite_report_summarizes_profile():
    report = founder_rewrite_quality_report(
        [
            {
                "id": "good",
                "prompt": "Rewrite this.",
                "required_terms": ["Gemma"],
                "completion": (
                    "A Gemma checkpoint loading is not a finish line.\n\n"
                    "We learned that the annoying way.\n\n"
                    "The model loaded, the adapter attached, and the first answer still "
                    "sounded like a template. For a writing model, that means the core "
                    "job is not done.\n\n"
                    "First try is the product."
                ),
            }
        ]
    )

    assert report["ok"] is True
    assert report["settings"]["profile"] == "founder_rewrite"
