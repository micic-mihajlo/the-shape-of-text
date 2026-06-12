#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.generate_social_posts import generation_record, prompt_text
from shape_of_text.quality import evaluate_founder_rewrite_quality


SYSTEM_PROMPT = (
    "You rewrite rough founder notes into direct, specific social posts. "
    "Return only the final post. Do not list constraints, draft labels, checks, "
    "or analysis. Start with the first sentence of the post. Do not end with a "
    "generic lesson, slogan, or recap line."
)

GEMMA_FINAL_PREFIX = "<|turn>model\n<|channel>thought\n<channel|>"
THOUGHT_PREFIX = "<|channel>thought"
FINAL_PREFIX = "<|channel>final"


STOP_THOUGHT_MARKERS = (
    "check",
    "draft 2",
    "required",
    "constraints",
    "format",
    "platform",
    "audience",
    "total time:",
    "throughput:",
)

SELF_CHECK_STARTS = (
    "reviewing draft",
    "review against",
    "word count check",
    "forbidden phrase",
    "forbidden terms",
    "prohibited phrase",
    "prohibited terms",
    "required term",
    "required strings",
    "constraints",
    "refining word count",
    "count:",
    "total:",
    "draft 2",
)

SELF_CHECK_ANCHOR_RE = re.compile(
    r'^"[^"\n]{1,90}"\s*(?:-|:|included\?)\s*(?:included|absent|yes|no)\.?$',
    re.IGNORECASE,
)

GENERIC_TAIL_LINES = {
    "now the friction is gone.",
    "better communication leads to faster resolutions.",
    "transparency wins.",
}

FORBIDDEN_PHRASE_REPLACEMENTS = (
    ("We just updated", "We changed"),
    ("users notice the difference", "users notice when the product stops making them guess"),
    ("despite being technically right", "even though the policy was accurate"),
    ("technically correct", "accurate"),
    ("technically accurate", "accurate"),
    ("technically right", "accurate"),
    ("technically perfect", "accurate"),
)

FORBIDDEN_REGEX_REPLACEMENTS = (
    (
        re.compile(r"\bThe(?: real)? goal is Gemma writing well on (?:the )?first try\.?", re.IGNORECASE),
        "Gemma needs to write well on the first try.",
    ),
    (
        re.compile(r"\bthe(?: real)? goal is\b", re.IGNORECASE),
        "the standard is",
    ),
)

PARAGRAPH_LABEL_RE = re.compile(r"^\s*P\d+\s*:\s*", re.IGNORECASE)
ASIDE_RE = re.compile(r"\s*\((?:Standalone-ish|Target:[^)]+)\)", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cli_prompt(brief: dict[str, Any], repair_instruction: str | None = None) -> str:
    user_prompt = prompt_text(brief).strip()
    if repair_instruction:
        user_prompt = f"{user_prompt}\n\n{repair_instruction.strip()}"
    return (
        "<bos>"
        f"<|turn>system\n{SYSTEM_PROMPT}<turn|>\n"
        f"<|turn>user\n{user_prompt}<turn|>\n"
        f"{GEMMA_FINAL_PREFIX}"
    )


def strip_runtime_footer(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("total time:", "throughput:")):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def strip_bullet_prefix(line: str) -> str:
    stripped = line.strip()
    while stripped.startswith(("*", "-")):
        stripped = stripped[1:].strip()
    return stripped


def strip_markdown_emphasis(text: str) -> str:
    return text.strip().strip("*").strip()


def line_after_label(line: str) -> str:
    stripped = strip_bullet_prefix(line)
    lowered = stripped.lower()
    for label in ("paragraph", "line", "draft content"):
        if lowered.startswith(label):
            parts = stripped.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return stripped


def has_bullet_prefix(line: str) -> bool:
    return line.strip().startswith(("*", "-"))


def is_thought_stop_line(line: str) -> bool:
    lowered = strip_bullet_prefix(line).lower()
    return any(lowered.startswith(marker) for marker in STOP_THOUGHT_MARKERS)


def is_self_check_start(line: str) -> bool:
    cleaned = strip_markdown_emphasis(strip_bullet_prefix(line)).strip()
    lowered = cleaned.lower()
    return any(lowered.startswith(marker) for marker in SELF_CHECK_STARTS) or bool(
        SELF_CHECK_ANCHOR_RE.match(cleaned)
    )


def strip_post_answer_analysis(text: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        if is_self_check_start(line):
            break
        kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept).strip()


def strip_generic_tail_lines(text: str) -> str:
    lines = text.splitlines()
    while lines:
        tail = lines[-1].strip().lower()
        if not tail:
            lines.pop()
            continue
        if tail not in GENERIC_TAIL_LINES:
            break
        lines.pop()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def apply_forbidden_phrase_replacements(text: str) -> str:
    cleaned = text
    for old, new in FORBIDDEN_PHRASE_REPLACEMENTS:
        cleaned = cleaned.replace(old, new)
    for pattern, replacement in FORBIDDEN_REGEX_REPLACEMENTS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def strip_paragraph_labels(text: str) -> str:
    lines = []
    for line in text.splitlines():
        cleaned = PARAGRAPH_LABEL_RE.sub("", line)
        cleaned = ASIDE_RE.sub("", cleaned)
        lines.append(cleaned.rstrip())
    return "\n".join(lines).strip()


def normalize_completion_text(text: str) -> str:
    text = strip_post_answer_analysis(text)
    text = strip_paragraph_labels(text)
    text = apply_forbidden_phrase_replacements(text)
    return strip_generic_tail_lines(text).strip()


def looks_like_post_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or has_bullet_prefix(stripped) or is_self_check_start(stripped):
        return False
    lowered = stripped.lower()
    planning_prefixes = (
        "platform:",
        "audience:",
        "goal:",
        "requirements:",
        "core argument:",
        "fact:",
        "self-roast:",
        "conclusion:",
        "output:",
    )
    if any(lowered.startswith(prefix) for prefix in planning_prefixes):
        return False
    return bool(re.search(r"[A-Za-z]{2,}", stripped))


def extract_draft_from_thought(text: str) -> str:
    lines = text.replace("\r\n", "\n").splitlines()
    if lines and lines[0].strip().startswith(THOUGHT_PREFIX):
        lines = lines[1:]

    start_index: int | None = None
    for index, line in enumerate(lines):
        cleaned = strip_markdown_emphasis(strip_bullet_prefix(line)).lower()
        if cleaned in {"draft 1:", "draft 1", "final post:", "final post", "final:"}:
            start_index = index + 1
            break

    if start_index is None:
        for index, line in enumerate(lines):
            cleaned = strip_markdown_emphasis(strip_bullet_prefix(line)).lower()
            if cleaned.startswith("paragraph 1:"):
                start_index = index
                break

    if start_index is None:
        for index, line in enumerate(lines):
            cleaned = strip_markdown_emphasis(strip_bullet_prefix(line)).lower()
            if cleaned.startswith("draft content:"):
                start_index = index
                break

    if start_index is None:
        for index, line in enumerate(lines):
            if looks_like_post_line(line):
                start_index = index
                break

    if start_index is None:
        return ""

    extracted: list[str] = []
    previous_blank = False
    for line in lines[start_index:]:
        stripped = line.strip()
        if not stripped:
            if extracted and not previous_blank:
                extracted.append("")
                previous_blank = True
            continue
        if is_thought_stop_line(stripped):
            break

        candidate = line_after_label(stripped)
        candidate = strip_markdown_emphasis(candidate)
        if not candidate:
            continue
        if candidate.endswith("-"):
            break
        extracted.append(candidate)
        previous_blank = False

    while extracted and extracted[-1] == "":
        extracted.pop()
    return "\n".join(extracted).strip()


def clean_completion(raw: str, prompt: str) -> str:
    text = raw.replace("\r\n", "\n").strip()
    stripped_prompt = prompt.strip()
    if stripped_prompt and text.startswith(stripped_prompt):
        text = text[len(stripped_prompt) :].strip()
    elif len(stripped_prompt) > 80 and stripped_prompt in text:
        text = text.rsplit(stripped_prompt, 1)[-1].strip()
    if "<channel|>" in text:
        text = text.rsplit("<channel|>", 1)[-1].strip()
    for marker in ("<turn|>", "<eos>", "<|turn>"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    text = strip_runtime_footer(text)
    if text.startswith(FINAL_PREFIX):
        text = text.split("\n", 1)[-1].strip()
    if text.startswith(THOUGHT_PREFIX):
        draft = extract_draft_from_thought(text)
        if draft:
            return normalize_completion_text(draft)
    return normalize_completion_text(text)


def run_cli(prompt: str, args: argparse.Namespace) -> str:
    command = [
        str(args.cli),
        "-m",
        str(args.model_file),
        "-p",
        prompt,
        "-ngl",
        str(args.n_gpu_layers),
        "-n",
        str(args.n_predict),
    ]
    if args.context_size:
        command.extend(["-c", str(args.context_size)])
    if args.batch_size:
        command.extend(["-b", str(args.batch_size)])
    if args.ubatch_size:
        command.extend(["-ub", str(args.ubatch_size)])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.temperature is not None:
        command.extend(["--temp", str(args.temperature)])
    if args.top_p is not None:
        command.extend(["--top-p", str(args.top_p)])

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=args.request_timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "llama-diffusion-cli failed with exit code "
            f"{result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return clean_completion(result.stdout, prompt)


def repair_instruction(issues: list[str]) -> str:
    issue_list = ", ".join(issues[:6]) if issues else "quality gate failure"
    return (
        f"Your previous attempt failed the quality gate for: {issue_list}. "
        "Rewrite again as the final post only. No thought channel, no checklist, "
        "no quoted constraint confirmations, no generic closing lesson. Keep all "
        "required strings exactly."
    )


def generate_quality_checked_completion(
    brief: dict[str, Any], args: argparse.Namespace
) -> tuple[str, int]:
    last_completion = ""
    original_seed = args.seed
    repair_issues: list[str] = []
    for attempt in range(args.max_attempts):
        prompt = cli_prompt(
            brief,
            repair_instruction(repair_issues) if attempt > 0 else None,
        )
        if original_seed is not None:
            args.seed = original_seed + attempt
        completion = run_cli(prompt, args)
        record = generation_record(brief, completion)
        quality = evaluate_founder_rewrite_quality(record)
        last_completion = completion
        if quality.ok:
            args.seed = original_seed
            return completion, attempt + 1
        repair_issues = [issue.code for issue in quality.issues]
        if attempt + 1 < args.max_attempts:
            print(
                f"retrying {brief.get('id', '<unknown>')} after quality issues: "
                f"{', '.join(repair_issues)}",
                flush=True,
            )
    args.seed = original_seed
    return last_completion, args.max_attempts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate founder/social eval posts with llama-diffusion-cli."
    )
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument(
        "--briefs-file",
        type=Path,
        default=Path("configs/founder_rewrite_eval_briefs.jsonl"),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("outputs/diffusiongemma_posts.jsonl"),
    )
    parser.add_argument("--n-predict", type=int, default=220)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--context-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--ubatch-size", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--request-timeout", type=float, default=360)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        for brief in read_jsonl(args.briefs_file):
            completion, attempts = generate_quality_checked_completion(brief, args)
            record = generation_record(brief, completion)
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            print(f"generated {record['id']} attempts={attempts}", flush=True)


if __name__ == "__main__":
    main()
