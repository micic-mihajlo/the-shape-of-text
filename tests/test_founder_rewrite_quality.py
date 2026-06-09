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
