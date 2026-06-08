from shape_of_text.style_metrics import aggregate_style_metrics, style_delta, top_repeated_terms


def test_style_metrics_aggregate_and_delta():
    target = [
        "We shipped a focused update. Try it today.",
        "A small release landed with clearer defaults.",
    ]
    candidate = [
        "We shipped a focused update. Try it today. Share feedback.",
        "A small release landed with clearer defaults.",
    ]

    metrics = aggregate_style_metrics(candidate)
    assert metrics["word_count"] > 0
    assert metrics["cta_marker_count"] >= 1

    delta = style_delta(candidate, target)
    assert delta["word_count"] > 0

    repeated = top_repeated_terms(candidate, limit=2)
    assert repeated
