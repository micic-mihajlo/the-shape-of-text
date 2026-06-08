from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from statistics import mean

WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_RE = re.compile(r"[.!?]+")
HASHTAG_RE = re.compile(r"#\w+")
MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
CTA_WORDS = {
    "apply",
    "comment",
    "join",
    "learn",
    "read",
    "reply",
    "share",
    "subscribe",
    "try",
    "watch",
}


@dataclass(frozen=True)
class PostStyleMetrics:
    char_count: float
    word_count: float
    sentence_count: float
    avg_word_length: float
    avg_sentence_length: float
    type_token_ratio: float
    punctuation_rate: float
    hashtag_count: float
    mention_count: float
    url_count: float
    cta_marker_count: float


def post_style_metrics(text: str) -> PostStyleMetrics:
    words = WORD_RE.findall(text)
    lowered_words = [word.casefold() for word in words]
    sentence_count = max(1, len(SENTENCE_RE.findall(text)))
    char_count = len(text)
    word_count = len(words)
    punctuation_count = sum(1 for char in text if char in ".,!?;:")
    cta_count = sum(1 for word in lowered_words if word in CTA_WORDS)

    return PostStyleMetrics(
        char_count=float(char_count),
        word_count=float(word_count),
        sentence_count=float(sentence_count),
        avg_word_length=float(mean(len(word) for word in words)) if words else 0.0,
        avg_sentence_length=float(word_count / sentence_count) if sentence_count else 0.0,
        type_token_ratio=float(len(set(lowered_words)) / word_count) if word_count else 0.0,
        punctuation_rate=float(punctuation_count / max(1, char_count)),
        hashtag_count=float(len(HASHTAG_RE.findall(text))),
        mention_count=float(len(MENTION_RE.findall(text))),
        url_count=float(len(URL_RE.findall(text))),
        cta_marker_count=float(cta_count),
    )


def aggregate_style_metrics(posts: list[str]) -> dict[str, float]:
    if not posts:
        return {field: 0.0 for field in PostStyleMetrics.__dataclass_fields__}

    metrics = [post_style_metrics(post) for post in posts]
    return {
        field: float(mean(getattr(metric, field) for metric in metrics))
        for field in PostStyleMetrics.__dataclass_fields__
    }


def top_repeated_terms(posts: list[str], *, limit: int = 20) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for post in posts:
        counter.update(word.casefold() for word in WORD_RE.findall(post) if len(word) > 3)
    return counter.most_common(limit)


def style_delta(candidate_posts: list[str], target_posts: list[str]) -> dict[str, float]:
    candidate = aggregate_style_metrics(candidate_posts)
    target = aggregate_style_metrics(target_posts)
    return {key: candidate[key] - target[key] for key in candidate}
