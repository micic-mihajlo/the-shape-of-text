#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


TOPICS = [
    {
        "topic": "a settings page cleanup",
        "detail": "the defaults now match the way people actually use the tool",
        "friction": "three repeated clicks from a weekly workflow",
        "ask": "which setting still feels too hidden",
        "audience": "existing users",
    },
    {
        "topic": "a prompt-to-output eval harness",
        "detail": "every run now writes the prompt, completion, and quality report together",
        "friction": "guesswork from model comparison",
        "ask": "what failure case we should add next",
        "audience": "AI builders",
    },
    {
        "topic": "a tiny local smoke test",
        "detail": "the smallest run catches broken data shapes before GPU time starts",
        "friction": "expensive surprises during remote training",
        "ask": "which preflight check saved them recently",
        "audience": "engineers",
    },
    {
        "topic": "a cleaner export flow",
        "detail": "the filename, format, and destination now stay visible until export finishes",
        "friction": "second-guessing where a file went",
        "ask": "what export format still needs attention",
        "audience": "operators",
    },
    {
        "topic": "a social post drafting adapter",
        "detail": "the model is judged on held-out briefs before it gets published",
        "friction": "shipping a model that only works in the demo prompt",
        "ask": "what kind of brief is hardest to write from",
        "audience": "technical founders",
    },
    {
        "topic": "a lightweight quality gate",
        "detail": "bad generations fail the job before any artifact reaches the hub",
        "friction": "manual review after a broken upload",
        "ask": "which output pattern should be blocked first",
        "audience": "ML engineers",
    },
    {
        "topic": "a changelog parser",
        "detail": "short updates can now become structured release notes without rewriting them",
        "friction": "turning small fixes into readable updates",
        "ask": "what release-note style feels most useful",
        "audience": "product teams",
    },
    {
        "topic": "a dashboard latency fix",
        "detail": "the slowest view now loads after the first result instead of waiting for every panel",
        "friction": "blank screens during routine checks",
        "ask": "which dashboard moment still feels slow",
        "audience": "busy teams",
    },
    {
        "topic": "a better onboarding checklist",
        "detail": "new users now see the next useful action instead of a wall of setup tasks",
        "friction": "dropping out before the first successful run",
        "ask": "which first-run step should be shorter",
        "audience": "startup founders",
    },
    {
        "topic": "a model-card refresh",
        "detail": "the card now shows data counts, eval metrics, and known limitations",
        "friction": "unclear handoffs between training and testing",
        "ask": "what evidence makes a model easier to trust",
        "audience": "open-source contributors",
    },
    {
        "topic": "a weekly planning view",
        "detail": "tasks now group by outcome instead of the order they were added",
        "friction": "busy lists that hide the important work",
        "ask": "which planning view actually survives Monday",
        "audience": "indie hackers",
    },
    {
        "topic": "a note-to-plan workflow",
        "detail": "messy research notes now turn into a short plan with risks and next actions",
        "friction": "starting from the same scratch document every time",
        "ask": "what kind of note is hardest to organize",
        "audience": "weekend builders",
    },
    {
        "topic": "a regression test for generated text",
        "detail": "the test catches prompt echo, control tokens, placeholders, and code blocks",
        "friction": "polishing a model after the bad artifact is already public",
        "ask": "which text regression is easiest to miss",
        "audience": "AI engineers",
    },
    {
        "topic": "a billing page simplification",
        "detail": "the plan, renewal date, and cancel action now live in one predictable place",
        "friction": "support tickets about basic account state",
        "ask": "which billing detail should never be hidden",
        "audience": "SaaS operators",
    },
    {
        "topic": "a feedback inbox triage pass",
        "detail": "similar requests now cluster before anyone writes a roadmap note",
        "friction": "mistaking the loudest request for the most common one",
        "ask": "how teams separate signal from noise",
        "audience": "product managers",
    },
    {
        "topic": "a documentation search fix",
        "detail": "exact error messages now rank above broad conceptual pages",
        "friction": "reading three docs pages to find one command",
        "ask": "which docs query still wastes time",
        "audience": "developers",
    },
]


ARCHETYPES = [
    "launch",
    "changelog",
    "lesson",
    "field_report",
    "contrarian",
    "ask",
    "technical_plain",
    "x_short",
]

DIRECT_POST_INSTRUCTION = (
    " Return only one finished post. Do not give options, labels, headings, "
    "explanations, or analysis."
)


def direct_prompt(text: str) -> str:
    return text + DIRECT_POST_INSTRUCTION


def prompt_for(topic: dict[str, str], archetype: str, platform: str) -> str:
    audience = topic["audience"]
    base = f"Write a {platform} post about {topic['topic']}."
    if archetype == "launch":
        return direct_prompt(
            f"{base} Audience: {audience}. Make it concrete, modest, and useful. "
            "Mention what changed and end with a grounded feedback ask."
        )
    if archetype == "changelog":
        return direct_prompt(
            f"{base} Audience: {audience}. Write it like a plain product changelog, "
            "not a campaign. Include the user-visible improvement."
        )
    if archetype == "lesson":
        return direct_prompt(
            f"{base} Audience: {audience}. Frame it as a lesson learned from shipping. "
            "Keep it concise and specific."
        )
    if archetype == "field_report":
        return direct_prompt(
            f"{base} Audience: {audience}. Write a field report with what worked, "
            "what changed, and what happens next."
        )
    if archetype == "contrarian":
        return direct_prompt(
            f"{base} Audience: {audience}. Start from a restrained contrarian take "
            "and make the practical point clear."
        )
    if archetype == "ask":
        return direct_prompt(
            f"{base} Audience: {audience}. Ask for useful feedback without sounding "
            "like a marketing campaign."
        )
    if archetype == "technical_plain":
        return direct_prompt(
            f"{base} Audience: {audience}. Explain the technical value in plain "
            "language without equations or jargon."
        )
    return direct_prompt(
        f"{base} Audience: {audience}. Make it short enough for X, with one concrete "
        "detail and no hype."
    )


def completion_for(topic: dict[str, str], archetype: str, platform: str) -> str:
    title = topic["topic"]
    detail = topic["detail"]
    friction = topic["friction"]
    ask = topic["ask"]

    if archetype == "launch":
        return (
            f"Small launch today: {title}.\n\n"
            f"The useful change is simple: {detail}. It removes {friction}, which is "
            "exactly the kind of small drag that compounds when people use a product "
            f"every week.\n\nIf you try it, I want the unpolished feedback: {ask}?"
        )
    if archetype == "changelog":
        return (
            f"Product note: we cleaned up {title}.\n\n"
            f"What changed: {detail}.\n"
            f"Why it matters: it removes {friction}.\n"
            "The goal was not to make the workflow bigger. It was to make the next "
            "obvious step easier to trust."
        )
    if archetype == "lesson":
        return (
            f"Lesson from shipping {title}: small checks beat late cleanup.\n\n"
            f"We found that {detail}. That mattered because the old path created "
            f"{friction}.\n\n"
            "The fix was not dramatic. It just made the workflow honest earlier, "
            "before a small miss turned into a public artifact."
        )
    if archetype == "field_report":
        return (
            f"Field report: {title} is in better shape.\n\n"
            f"What worked: {detail}.\n"
            f"What changed: the default path no longer leaves people dealing with {friction}.\n"
            f"Next step: watch real usage and tighten the part people still question: {ask}."
        )
    if archetype == "contrarian":
        return (
            "A take I keep coming back to: the best product updates usually feel "
            f"smaller than the work behind them.\n\nWith {title}, the point was not "
            f"to add more surface area. The point was that {detail}, so users avoid "
            f"{friction}.\n\nUseful beats impressive here."
        )
    if archetype == "ask":
        return (
            f"We are testing {title} with a sharper constraint: feedback has to point "
            "to a real workflow, not a vague preference.\n\n"
            f"The current improvement is that {detail}. The risk is that we only solved "
            f"the visible part of the problem: {friction}.\n\n"
            f"If you have seen this break down, I would like to know: {ask}?"
        )
    if archetype == "technical_plain":
        return (
            f"The technical value of {title} is not the machinery around it. It is the "
            f"behavior change: {detail}.\n\nThat matters because users should not have "
            f"to think about {friction}. A good system makes the correct path feel "
            "boring, repeatable, and easy to inspect."
        )
    return (
        f"Shipped: {title}. {detail}. The win is not flashy, but it cuts {friction}. "
        f"That is the kind of update people notice after the third time they use it."
    )


def build_examples(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    for topic in TOPICS:
        for archetype in ARCHETYPES:
            platform = "X" if archetype == "x_short" else "LinkedIn"
            examples.append(
                {
                    "prompt": prompt_for(topic, archetype, platform),
                    "completion": completion_for(topic, archetype, platform),
                    "platform": platform,
                    "topic": topic["topic"],
                    "audience": topic["audience"],
                    "archetype": archetype,
                }
            )
    rng.shuffle(examples)
    return examples


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build original social-writing SFT data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/hackathon_social_instructions"),
    )
    parser.add_argument("--validation-ratio", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = build_examples(args.seed)
    validation_count = max(1, int(len(examples) * args.validation_ratio))
    validation = examples[:validation_count]
    train = examples[validation_count:]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    print(f"examples={len(examples)}")
    print(f"train={len(train)}")
    print(f"validation={len(validation)}")


if __name__ == "__main__":
    main()
