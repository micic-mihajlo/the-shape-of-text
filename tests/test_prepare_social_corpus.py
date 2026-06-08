import argparse
import json

from scripts.prepare_social_corpus import prepare_posts, write_split


def test_prepare_posts_normalizes_dedupes_and_splits(tmp_path):
    source = tmp_path / "posts.jsonl"
    rows = [
        {"text": "  A useful launch note with a concrete detail and clean next step.  "},
        {"text": "A useful launch note with a concrete detail and clean next step."},
        {"text": "too short"},
        {"text": "Read this update https://example.com because it has enough useful detail."},
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    args = argparse.Namespace(
        inputs=[source],
        field="text",
        keep_urls=False,
        min_chars=20,
        max_chars=200,
        seed=1,
    )
    posts = prepare_posts(args)
    assert len(posts) == 2
    assert all("https://" not in post for post in posts)

    train_path, validation_path = write_split(posts, tmp_path / "out", 0.5)
    assert train_path.read_text(encoding="utf-8").strip()
    assert validation_path.read_text(encoding="utf-8").strip()
