from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable


_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_/-]*")
_SPACE_RE = re.compile(r"\s+")
_SENTENCE_PUNCT_RE = re.compile(r"[.!?]")
_HASHTAG_RE = re.compile(r"#[A-Za-z0-9_]+")
_TERMINAL_PUNCT_RE = re.compile(r"[.!?]\s*$")

COMMON_ENGLISH_WORDS = frozenset(
    {
        "a",
        "about",
        "and",
        "are",
        "as",
        "at",
        "but",
        "for",
        "from",
        "have",
        "in",
        "is",
        "it",
        "just",
        "not",
        "of",
        "on",
        "our",
        "that",
        "the",
        "this",
        "to",
        "today",
        "was",
        "we",
        "with",
        "you",
    }
)

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
    r"\bhere are (three|[0-9]+) options\b",
    r"\boption\s*[0-9]+\s*:",
    r"\bbest for\b",
    r"\bdepending on the (specific )?(tone|vibe)\b",
    r"\bthe post should be written in english\b",
    r"^\s*prompt\s*:",
    r"^\s*platform\s*:",
    r"^\s*audience\s*:",
    r"^\s*write a\b",
)

PLACEHOLDER_PATTERNS = (
    r"\[(?:link|framework name|company name|product name|tool name|insert [^\]]+)\]",
    r"\[[^\]]*(?:here|name|link|describe|explain|placeholder)[^\]]*\]",
)

ARTIFACT_PATTERNS = (
    r"\b[A-Za-z0-9]+_[A-Za-z0-9_]+\b",
    r"\b[A-Za-z0-9]+_[A-Za-z0-9_]*_[A-Za-z0-9_]+\b",
    r"\b[A-Za-z0-9]+_[A-Za-z0-9_]{8,}\b",
    r"_\s*$",
)

FOUNDER_REWRITE_SLOP_PATTERNS = (
    r"\bsmall product update\b",
    r"\btiny product update\b",
    r"\bnot (?:flashy|dramatic)\b",
    r"\bnothing dramatic\b",
    r"\bthe useful (?:part|change|bit)\b",
    r"\bthe (?:main|practical|technical) win\b",
    r"\bit removes friction\b",
    r"\bmoments of friction\b",
    r"\bthe kind of cleanup\b",
    r"\busers actually feel\b",
    r"\bworkflow (?:honest|people repeat)\b",
    r"\bthe next obvious step\b",
    r"\beasier to trust\b",
    r"\bwithout adding another step\b",
    r"\bpolite praise\b",
    r"\bquick field note\b",
    r"\bfield report\b",
    r"\bgrounded feedback ask\b",
    r"\bconcrete detail\b",
    r"\bsmall but useful\b",
    r"\bcommon path\b",
    r"\bdefault path\b",
    r"\breal workflow\b",
    r"\bsurface area\b",
    r"\bbar for this update\b",
    r"\bin better shape\b",
)

FOUNDER_REWRITE_PREFACE_PATTERNS = (
    r"^\s*sure[,!. ]",
    r"^\s*absolutely[,!. ]",
    r"^\s*here(?:'s| is)\b",
    r"^\s*i'?d write it like this\b",
    r"^\s*one (?:way|version)\b",
)

SELF_CORRECTION_PATTERNS = (
    r"\blet'?s try again\b",
    r"\bone more time\b",
    r"\bwait,?\s+that was\b",
    r"\bmaybe not\b",
)

UNFINISHED_TAIL_RE = re.compile(
    r"(?:\b(?:the|a|an|to|for|of|on|in|with|and|but|or|because|that|this|"
    r"those|these|from|by|about|into|instead|haven't|hasn't|isn't|aren't|"
    r"doesn't|don't)\b|for user)\s*$",
    flags=re.IGNORECASE,
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


def ascii_words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_RE.finditer(text)]


def prompt_echo_score(completion: str, prompt: str) -> float:
    prompt_norm = normalize_for_overlap(prompt)
    completion_norm = normalize_for_overlap(completion)
    if not prompt_norm or not completion_norm:
        return 0.0

    matcher = SequenceMatcher(None, prompt_norm, completion_norm, autojunk=False)
    longest = matcher.find_longest_match(0, len(prompt_norm), 0, len(completion_norm)).size
    return longest / max(1, min(len(prompt_norm), len(completion_norm)))


def copies_prompt_instruction(completion: str, prompt: str) -> bool:
    prompt_norm = normalize_for_overlap(prompt)
    completion_norm = normalize_for_overlap(completion)
    if not prompt_norm or not completion_norm:
        return False

    prompt_words = prompt_norm.split()
    starts = [0]
    starts.extend(
        index
        for index, word in enumerate(prompt_words[:12])
        if word in {"write", "explain"}
    )
    for start in starts:
        prefix = " ".join(prompt_words[start : start + 8])
        if len(prefix) < 24:
            continue
        if completion_norm.startswith(prefix) or prefix in completion_norm[:180]:
            return True

    instruction_phrases = (
        "the post should",
        "do not claim",
        "do not mention",
        "keep it plain",
        "keep the tone",
        "make the ask",
    )
    return any(phrase in prompt_norm and phrase in completion_norm for phrase in instruction_phrases)


def _matches_any(patterns: Iterable[str], text: str) -> list[str]:
    return [
        pattern
        for pattern in patterns
        if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    ]


def _character_noise(completion: str) -> dict[str, float]:
    visible = [char for char in completion if not char.isspace()]
    total = len(visible)
    if total == 0:
        return {
            "control_count": 0.0,
            "non_ascii_alpha_ratio": 0.0,
            "symbol_ratio": 0.0,
            "unusual_ratio": 0.0,
        }

    alpha_count = 0
    non_ascii_alpha_count = 0
    symbol_count = 0
    control_count = 0
    unusual_count = 0
    for char in visible:
        category = unicodedata.category(char)
        if char.isalpha():
            alpha_count += 1
            if not char.isascii():
                non_ascii_alpha_count += 1
        if category.startswith("S"):
            symbol_count += 1
        if category.startswith("C"):
            control_count += 1
        if not char.isascii() or category.startswith(("C", "S")):
            unusual_count += 1

    return {
        "control_count": float(control_count),
        "non_ascii_alpha_ratio": non_ascii_alpha_count / max(1, alpha_count),
        "symbol_ratio": symbol_count / total,
        "unusual_ratio": unusual_count / total,
    }


def repeated_phrase(completion: str, *, phrase_size: int = 5, max_count: int = 2) -> str | None:
    words = ascii_words(completion)
    if len(words) < phrase_size * (max_count + 1):
        return None
    counts: dict[tuple[str, ...], int] = {}
    for index in range(len(words) - phrase_size + 1):
        phrase = tuple(words[index : index + phrase_size])
        counts[phrase] = counts.get(phrase, 0) + 1
        if counts[phrase] > max_count:
            return " ".join(phrase)
    return None


def hashtag_issues(completion: str, *, max_hashtags: int = 4) -> list[QualityIssue]:
    hashtags = _HASHTAG_RE.findall(completion)
    issues: list[QualityIssue] = []
    if len(hashtags) > max_hashtags:
        issues.append(
            QualityIssue(
                "too_many_hashtags",
                f"completion has {len(hashtags)} hashtags; maximum is {max_hashtags}",
            )
        )
    noisy = [
        hashtag
        for hashtag in hashtags
        if len(hashtag) > 32 or hashtag.count("_") > 1
    ]
    if noisy:
        issues.append(
            QualityIssue("hashtag_artifact", "completion contains malformed hashtag artifacts")
        )
    return issues


def has_unfinished_tail(completion: str) -> bool:
    stripped = completion.strip()
    if not stripped:
        return False
    if _TERMINAL_PUNCT_RE.search(stripped):
        return False
    if _HASHTAG_RE.search(stripped.rsplit(maxsplit=1)[-1]):
        return False
    return bool(UNFINISHED_TAIL_RE.search(stripped))


def lacks_terminal_punctuation(completion: str, *, min_words: int = 35) -> bool:
    stripped = completion.strip()
    if not stripped or word_count(stripped) < min_words:
        return False
    return _TERMINAL_PUNCT_RE.search(stripped) is None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _compact_normalized(text: str) -> str:
    return re.sub(r"\s+", "", normalize_for_overlap(text))


def _normalized_term_variants(term: str) -> list[str]:
    variants = [term]
    numeric_commas_removed = re.sub(r"(?<=\d),(?=\d)", "", term)
    if numeric_commas_removed != term:
        variants.append(numeric_commas_removed)
    normalized = []
    for variant in variants:
        value = normalize_for_overlap(variant)
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _contains_normalized_phrase(text_norm: str, phrase_norm: str) -> bool:
    return re.search(
        rf"(?<![a-z0-9]){re.escape(phrase_norm)}(?![a-z0-9])",
        text_norm,
    ) is not None


def _term_present(text: str, term: str) -> bool:
    variants = _normalized_term_variants(term)
    if not variants:
        return True
    text_norm = normalize_for_overlap(text)
    return any(_contains_normalized_phrase(text_norm, variant) for variant in variants)


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _paragraph_word_counts(text: str) -> list[int]:
    return [
        word_count(paragraph)
        for paragraph in re.split(r"\n\s*\n", text.strip())
        if paragraph.strip()
    ]


def founder_rewrite_extra_issues(record: dict[str, Any], completion: str) -> list[QualityIssue]:
    issues: list[QualityIssue] = []

    preface_matches = _matches_any(FOUNDER_REWRITE_PREFACE_PATTERNS, completion)
    if preface_matches:
        issues.append(
            QualityIssue("assistant_preface", "completion starts with assistant framing")
        )

    slop_matches = _matches_any(FOUNDER_REWRITE_SLOP_PATTERNS, completion)
    if slop_matches:
        issues.append(
            QualityIssue(
                "generic_social_slop",
                "completion uses generic product-update or LinkedIn filler phrasing",
            )
        )

    forbidden_terms = _string_list(record.get("avoid_terms"))
    forbidden_terms.extend(_string_list(record.get("forbidden_terms")))
    forbidden = [
        term
        for term in forbidden_terms
        if _term_present(completion, term)
    ]
    if forbidden:
        issues.append(
            QualityIssue(
                "forbidden_term",
                f"completion includes forbidden terms: {', '.join(forbidden[:4])}",
            )
        )

    non_ascii_alpha = sorted({char for char in completion if char.isalpha() and not char.isascii()})
    if non_ascii_alpha:
        preview = "".join(non_ascii_alpha[:6])
        issues.append(
            QualityIssue(
                "non_ascii_alpha",
                f"completion contains non-ASCII alphabetic characters: {preview}",
            )
        )

    missing = [
        term
        for term in _string_list(record.get("required_terms") or record.get("anchors"))
        if not _term_present(completion, term)
    ]
    if missing:
        issues.append(
            QualityIssue(
                "missing_required_anchor",
                f"completion dropped required anchors: {', '.join(missing[:4])}",
            )
        )

    count = word_count(completion)
    lines = _nonempty_lines(completion)
    if count >= 50 and len(lines) < 3:
        issues.append(
            QualityIssue(
                "flat_single_block",
                "long founder rewrite should not be one dense paragraph",
            )
        )
    if count >= 65 and not any(word_count(line) <= 8 for line in lines):
        issues.append(
            QualityIssue(
                "missing_human_rhythm",
                "long founder rewrite needs at least one short emphasis line",
            )
        )

    long_paragraphs = [count for count in _paragraph_word_counts(completion) if count > 90]
    if long_paragraphs:
        issues.append(
            QualityIssue("overlong_paragraph", "completion contains an overlong paragraph")
        )

    return issues


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

    words = ascii_words(completion)
    common_word_count = sum(1 for word in words if word in COMMON_ENGLISH_WORDS)
    min_common_words = 2 if count < 45 else 3
    if count >= min_words and common_word_count < min_common_words:
        issues.append(
            QualityIssue(
                "low_english_signal",
                f"completion has only {common_word_count} common English connector words",
            )
        )

    noise = _character_noise(completion)
    if noise["control_count"] > 0:
        issues.append(
            QualityIssue("control_character_noise", "completion contains control characters")
        )
    if noise["non_ascii_alpha_ratio"] > 0.08:
        issues.append(
            QualityIssue(
                "non_latin_noise",
                f"completion has non-Latin alphabetic ratio {noise['non_ascii_alpha_ratio']:.3f}",
            )
        )
    if noise["symbol_ratio"] > 0.12:
        issues.append(
            QualityIssue(
                "symbol_noise",
                f"completion has symbol ratio {noise['symbol_ratio']:.3f}",
            )
        )
    if noise["unusual_ratio"] > 0.18:
        issues.append(
            QualityIssue(
                "unreadable_character_mix",
                f"completion has unusual character ratio {noise['unusual_ratio']:.3f}",
            )
        )
    if count > 35 and not _SENTENCE_PUNCT_RE.search(completion):
        issues.append(
            QualityIssue(
                "no_sentence_punctuation",
                "long completion has no sentence-ending punctuation",
            )
        )
    phrase = repeated_phrase(completion)
    if phrase is not None:
        issues.append(
            QualityIssue("repeated_phrase", f"completion repeats phrase: {phrase!r}")
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

    placeholder_matches = _matches_any(PLACEHOLDER_PATTERNS, completion)
    if placeholder_matches:
        issues.append(
            QualityIssue("placeholder_text", "completion contains placeholder text")
        )

    artifact_matches = _matches_any(ARTIFACT_PATTERNS, completion)
    if artifact_matches:
        issues.append(
            QualityIssue("artifact_suffix", "completion contains underscore or suffix artifacts")
        )

    self_correction_matches = _matches_any(SELF_CORRECTION_PATTERNS, completion)
    if self_correction_matches:
        issues.append(
            QualityIssue("self_correction_restart", "completion restarts or critiques itself")
        )

    issues.extend(hashtag_issues(completion))

    if has_unfinished_tail(completion):
        issues.append(
            QualityIssue("unfinished_tail", "completion appears to stop mid-thought")
        )
    elif lacks_terminal_punctuation(completion):
        issues.append(
            QualityIssue("missing_terminal_punctuation", "completion does not end cleanly")
        )

    echo = prompt_echo_score(completion, prompt)
    if echo > max_prompt_echo_score and copies_prompt_instruction(completion, prompt):
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


def evaluate_founder_rewrite_quality(
    record: dict[str, Any],
    *,
    completion_field: str = "completion",
    prompt_field: str = "prompt",
    min_words: int = 35,
    max_words: int = 260,
    max_prompt_echo_score: float = 0.35,
) -> QualityResult:
    base = evaluate_completion_quality(
        record,
        completion_field=completion_field,
        prompt_field=prompt_field,
        min_words=min_words,
        max_words=max_words,
        max_prompt_echo_score=max_prompt_echo_score,
    )
    completion = str(record.get(completion_field) or "").strip()
    issues = list(base.issues)
    issues.extend(founder_rewrite_extra_issues(record, completion))
    return QualityResult(
        ok=not issues,
        issues=tuple(issues),
        word_count=base.word_count,
        prompt_echo_score=base.prompt_echo_score,
    )


def founder_rewrite_quality_report(
    records: list[dict[str, Any]],
    *,
    completion_field: str = "completion",
    prompt_field: str = "prompt",
    min_words: int = 35,
    max_words: int = 260,
    max_prompt_echo_score: float = 0.35,
) -> dict[str, Any]:
    results = [
        evaluate_founder_rewrite_quality(
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
            "profile": "founder_rewrite",
        },
    }
