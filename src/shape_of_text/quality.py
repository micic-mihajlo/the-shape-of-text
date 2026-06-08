from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable


_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_/-]*")
_SPACE_RE = re.compile(r"\s+")

SPECIAL_TOKEN_PATTERNS = (
    r"<\|?image\|?>",
    r"<image\|>",
    r"<\|?audio\|?>",
    r"<audio\|>",
    r"<\|?video\|?>",
    r"<video\|>",
    r"<\|?tool",
    r"<tool\|>",
    r"<\|?turn",
    r"<turn\|>",
    r"<\|?channel",
    r"<channel\|>",
    r"<\|think\|>",
    r"<bos>",
    r"<eos>",
    r"<pad>",
)

TEMPLATE_PATTERNS = (
    r"```",
    r"\bdef\s+[A-Za-z_][A-Za-z0-9_]*\s*\(",
    r"\bclass\s+[A-Za-z_][A-Za-z0-9_]*\s*[:(]",
    r"^\s*#\s*the problem\b",
    r"\bcurrent state\s*\(before fix\)",
    r"\bproposed solution\b",
    r"\bthe system is experiencing\b",
    r"\[describe\b",
    r"\[explain\b",
    r"\bexisting code that causes issues\b",
    r"\bi recommend implementing\b",
    r"\bthe post should be written in english\b",
    r"^\s*prompt\s*:",
    r"^\s*platform\s*:",
    r"^\s*audience\s*:",
    r"^\s*write a\b",
)


@dataclass(frozen=True)
class QualityIssue:
    code: str
    message: str


@dataclass(frozen=True)
class QualityResult:
    ok: bool
    issues: tuple[QualityIssue, ...]
    word_count: int
    prompt_echo_score: float


def normalize_for_overlap(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return _SPACE_RE.sub(" ", text).strip()


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def prompt_echo_score(completion: str, prompt: str) -> float:
    prompt_norm = normalize_for_overlap(prompt)
    completion_norm = normalize_for_overlap(completion)
    if not prompt_norm or not completion_norm:
        return 0.0

    matcher = SequenceMatcher(None, prompt_norm, completion_norm, autojunk=False)
    longest = matcher.find_longest_match(0, len(prompt_norm), 0, len(completion_norm)).size
    return longest / max(1, min(len(prompt_norm), len(completion_norm)))


def _matches_any(patterns: Iterable[str], text: str) -> list[str]:
    return [
        pattern
        for pattern in patterns
        if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    ]


def evaluate_completion_quality(
    record: dict[str, Any],
    *,
    completion_field: str = "completion",
    prompt_field: str = "prompt",
    min_words: int = 18,
    max_words: int = 220,
    max_prompt_echo_score: float = 0.35,
) -> QualityResult:
    completion = str(record.get(completion_field) or "").strip()
    prompt = str(record.get(prompt_field) or "").strip()
    issues: list[QualityIssue] = []

    count = word_count(completion)
    if not completion:
        issues.append(QualityIssue("empty", "completion is empty"))
    elif count < min_words:
        issues.append(
            QualityIssue("too_short", f"completion has {count} words; minimum is {min_words}")
        )
    if count > max_words:
        issues.append(
            QualityIssue("too_long", f"completion has {count} words; maximum is {max_words}")
        )

    special_matches = _matches_any(SPECIAL_TOKEN_PATTERNS, completion)
    if special_matches:
        issues.append(
            QualityIssue(
                "special_token_leak",
                "completion contains model/control special tokens",
            )
        )

    template_matches = _matches_any(TEMPLATE_PATTERNS, completion)
    if template_matches:
        issues.append(
            QualityIssue("template_or_code", "completion looks like a template or code block")
        )

    echo = prompt_echo_score(completion, prompt)
    if echo > max_prompt_echo_score:
        issues.append(
            QualityIssue(
                "prompt_echo",
                f"completion copies too much of the prompt; echo score {echo:.3f}",
            )
        )

    return QualityResult(
        ok=not issues,
        issues=tuple(issues),
        word_count=count,
        prompt_echo_score=echo,
    )


def quality_report(
    records: list[dict[str, Any]],
    *,
    completion_field: str = "completion",
    prompt_field: str = "prompt",
    min_words: int = 18,
    max_words: int = 220,
    max_prompt_echo_score: float = 0.35,
) -> dict[str, Any]:
    results = [
        evaluate_completion_quality(
            record,
            completion_field=completion_field,
            prompt_field=prompt_field,
            min_words=min_words,
            max_words=max_words,
            max_prompt_echo_score=max_prompt_echo_score,
        )
        for record in records
    ]
    failures = []
    for index, (record, result) in enumerate(zip(records, results, strict=True)):
        if result.ok:
            continue
        failures.append(
            {
                "id": record.get("id", index),
                "index": index,
                "issues": [
                    {"code": issue.code, "message": issue.message}
                    for issue in result.issues
                ],
                "word_count": result.word_count,
                "prompt_echo_score": result.prompt_echo_score,
            }
        )

    total = len(records)
    failed = len(failures)
    return {
        "ok": failed == 0,
        "total": total,
        "passed": total - failed,
        "failed": failed,
        "failure_rate": failed / total if total else 1.0,
        "failures": failures,
        "settings": {
            "completion_field": completion_field,
            "prompt_field": prompt_field,
            "min_words": min_words,
            "max_words": max_words,
            "max_prompt_echo_score": max_prompt_echo_score,
        },
    }
