#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


GLOBAL_AVOID_TERMS = [
    "small product update",
    "not flashy",
    "nothing dramatic",
    "the useful part",
    "it removes friction",
    "users actually feel",
    "field report",
    "quick field note",
    "main win",
    "concrete detail",
    "grounded feedback ask",
]

PROMPT_VARIANTS = [
    (
        "Rewrite this rough draft into one finished LinkedIn post. Keep the argument, "
        "keep the named facts, and make it sound like a founder thinking in public."
    ),
    (
        "Edit this draft for a sharp social post. Preserve the actual claim and anchors. "
        "Cut generic launch language and tidy corporate phrasing."
    ),
    (
        "Turn the draft into a human founder post with a clear turn, specific facts, "
        "and uneven but readable rhythm."
    ),
    (
        "Rewrite for LinkedIn without making it sound like an announcement. Keep the "
        "voice direct, slightly opinionated, and concrete."
    ),
    (
        "Make this post publishable on the first try. Remove filler, keep the stakes, "
        "and return only the final post."
    ),
]

DIRECT_REWRITE_INSTRUCTION = (
    "Return only one finished post. Do not give options, labels, headings, analysis, "
    "hashtags, placeholders, or bracketed links. Do not invent metrics. End cleanly."
)

SEED_REWRITES: list[dict[str, Any]] = [
    {
        "id": "ai_usage_rivet",
        "audience": "startup founders",
        "required_terms": ["Rivet", "1,800", "AI-generated code"],
        "source_draft": (
            "Some people on the team use AI all day but they are not always the best. "
            "Rivet was built with AI-generated code and has 1,800 teams. Maybe AI is "
            "not the issue, maybe thinking is."
        ),
        "completion": (
            "The engineers using AI the most aren't automatically shipping the best work.\n\n"
            "Rivet proved the opposite problem to me.\n\n"
            "It was built with a lot of AI-generated code. Now 1,800 teams use it every "
            "week. So when someone says AI makes engineers worse, I want to know what "
            "they were aiming at before the model got involved.\n\n"
            "AI does not replace taste.\n"
            "It accelerates whatever taste was already there."
        ),
    },
    {
        "id": "northstar_onboarding",
        "audience": "technical founders",
        "required_terms": ["Northstar", "three setup steps", "four minutes"],
        "source_draft": (
            "We improved Northstar onboarding. We removed three setup steps. New users "
            "can get to the first run in four minutes. This is a better first experience."
        ),
        "completion": (
            "We removed three setup steps from Northstar this week.\n\n"
            "That sounds too small to post about until you watch a new user hit them in "
            "a row.\n\n"
            "The bug was not the checklist. The bug was making people prove they "
            "understood the product before the product had done anything useful for them.\n\n"
            "Now the first run starts in under four minutes.\n"
            "Much better signal."
        ),
    },
    {
        "id": "mason_refactor",
        "audience": "engineering leads",
        "required_terms": ["Mason", "six-week refactor", "18-line patch"],
        "source_draft": (
            "We thought Mason needed a six-week refactor. It turned out the issue could "
            "be fixed with an 18-line patch. Lesson about scoping."
        ),
        "completion": (
            "Mason did not need the six-week refactor we had planned.\n\n"
            "It needed an 18-line patch and a little honesty about the real problem.\n\n"
            "That was uncomfortable, because a big refactor feels like serious work. "
            "Deleting the excuse and fixing the boring edge case feels smaller.\n\n"
            "But users do not grade effort.\n"
            "They grade whether the thing works tomorrow."
        ),
    },
    {
        "id": "lane_churn",
        "audience": "SaaS operators",
        "required_terms": ["Lane", "11 customers", "churn dashboard"],
        "source_draft": (
            "The Lane churn dashboard was misleading. 11 customers had the same issue "
            "but the dashboard made it look like separate reasons. We changed the view."
        ),
        "completion": (
            "Our churn dashboard in Lane was telling the truth in the least helpful way.\n\n"
            "11 customers looked like 11 different stories.\n\n"
            "When we read the notes, it was the same reason wearing different clothes: "
            "people did not know what to do after setup.\n\n"
            "The dashboard is cleaner now, but the lesson was sharper than the UI fix.\n"
            "Bad grouping can make a simple problem look strategic."
        ),
    },
    {
        "id": "eval_harness",
        "audience": "AI builders",
        "required_terms": ["eval harness", "prompt", "completion"],
        "source_draft": (
            "We built an eval harness. It stores prompt and completion together. This "
            "makes judging model outputs easier and reduces random vibes in eval."
        ),
        "completion": (
            "I do not trust model evals that separate the prompt from the completion.\n\n"
            "That was the reason for the new eval harness.\n\n"
            "Every run now saves both sides together, then scores the output against the "
            "same brief it answered. It is a boring change, but it kills a lot of fake "
            "confidence.\n\n"
            "Most bad writing models do not fail in the loss curve.\n"
            "They fail when you read the first answer."
        ),
    },
    {
        "id": "pricing_page",
        "audience": "founders",
        "required_terms": ["pricing page", "free plan", "14%"],
        "source_draft": (
            "We changed the pricing page. The free plan was confusing people. Signups "
            "went up 14 percent after we stopped hiding what the free plan includes."
        ),
        "completion": (
            "We made one pricing page change and signups moved 14%.\n\n"
            "The change was embarrassing: we stopped hiding what the free plan included.\n\n"
            "No new funnel. No clever packaging. Just fewer people wondering whether the "
            "button was going to waste their time.\n\n"
            "Sometimes the growth idea is not bigger persuasion.\n"
            "It is fewer reasons to hesitate."
        ),
    },
    {
        "id": "agent_bug",
        "audience": "AI engineers",
        "required_terms": ["agent", "refund email", "sandbox"],
        "source_draft": (
            "The agent drafted a refund email in production when it should only have "
            "done that in sandbox. We caught it. The fix was permissions and better evals."
        ),
        "completion": (
            "The scariest AI bug we hit was not the model being dumb.\n\n"
            "It was the agent being almost right in the wrong place.\n\n"
            "It drafted a refund email in production when that action belonged in the "
            "sandbox. Nobody got charged. Nobody got emailed. But the boundary was too "
            "soft, and that was enough.\n\n"
            "The fix was permissions first, evals second.\n"
            "Trust starts with what the system cannot do."
        ),
    },
    {
        "id": "weekly_digest",
        "audience": "product teams",
        "required_terms": ["weekly digest", "Monday", "12 minutes"],
        "source_draft": (
            "The weekly digest is faster. It took 12 minutes on Monday before and now "
            "the team can read it quickly. It focuses on decisions instead of every update."
        ),
        "completion": (
            "Our weekly digest used to take 12 minutes to read on Monday.\n\n"
            "That sounds harmless until you remember it was supposed to make the week "
            "clearer, not steal the first meeting.\n\n"
            "We changed the format around decisions instead of activity. What changed, "
            "what is blocked, what needs a yes or no.\n\n"
            "Less reporting theater.\n"
            "More useful Monday."
        ),
    },
    {
        "id": "customer_request",
        "audience": "product founders",
        "required_terms": ["37 requests", "one checkbox", "roadmap"],
        "source_draft": (
            "We had 37 requests for one checkbox. It looked like roadmap noise at first "
            "but it was actually a broken assumption in our settings."
        ),
        "completion": (
            "37 requests for one checkbox looked like roadmap noise.\n\n"
            "It was not.\n\n"
            "It was customers pointing at a bad assumption we kept defending because the "
            "settings page made sense to us.\n\n"
            "That is the annoying part about feedback. The wording is rarely perfect. "
            "The pattern is usually louder than the sentence.\n\n"
            "We shipped the checkbox."
        ),
    },
    {
        "id": "security_review",
        "audience": "engineering managers",
        "required_terms": ["security review", "CSV export", "Friday"],
        "source_draft": (
            "Security review found the CSV export could include fields that were not "
            "shown in the UI. We fixed it Friday. This was a good reminder."
        ),
        "completion": (
            "A security review caught something we should have caught sooner.\n\n"
            "The CSV export could include fields the UI did not show.\n\n"
            "That is exactly the sort of bug that feels boring until it is suddenly "
            "expensive. The screen looked right. The export told a different story.\n\n"
            "We fixed it Friday.\n"
            "The reminder: every second path is still the product."
        ),
    },
    {
        "id": "hiring_interview",
        "audience": "founders hiring engineers",
        "required_terms": ["interview", "debugging", "senior engineer"],
        "source_draft": (
            "We changed the interview. We care more about debugging than perfect syntax. "
            "A senior engineer should be able to make uncertainty smaller."
        ),
        "completion": (
            "I care less about perfect syntax in an interview than I used to.\n\n"
            "I care a lot more about debugging.\n\n"
            "A senior engineer earns the title by making uncertainty smaller. They ask "
            "better questions, cut the problem cleanly, and notice when the evidence "
            "does not support the story.\n\n"
            "The code matters.\n"
            "The shape of the thinking matters first."
        ),
    },
    {
        "id": "analytics_filter",
        "audience": "growth teams",
        "required_terms": ["analytics filter", "paid traffic", "Tuesday"],
        "source_draft": (
            "On Tuesday we added an analytics filter for paid traffic. It made the "
            "report less confusing because organic and paid were mixed together."
        ),
        "completion": (
            "On Tuesday we added the analytics filter everyone assumed already existed.\n\n"
            "Paid traffic and organic traffic had been sitting in the same report, "
            "politely ruining each other's numbers.\n\n"
            "The product was not missing a dashboard. It was missing a way to stop lying "
            "to ourselves with averages.\n\n"
            "The analytics filter is live now.\n"
            "The conversations are already shorter."
        ),
    },
    {
        "id": "launch_thread",
        "audience": "indie hackers",
        "required_terms": ["launch thread", "48 hours", "reply rate"],
        "source_draft": (
            "We studied the launch thread. It had a better reply rate after 48 hours "
            "because we answered objections instead of adding more features."
        ),
        "completion": (
            "The launch thread got better after we stopped adding features to it.\n\n"
            "For 48 hours we only answered the objections people were already leaving.\n\n"
            "No new screenshots. No bigger promise. Just clearer replies to the parts "
            "that made people hesitate.\n\n"
            "The reply rate moved because the thread became a conversation.\n"
            "That should have been obvious earlier."
        ),
    },
    {
        "id": "solo_founder",
        "audience": "solo founders",
        "required_terms": ["solo founder", "calendar", "two hours"],
        "source_draft": (
            "A solo founder told me the calendar feature saved two hours. It was because "
            "the system stopped asking them to schedule the same thing twice."
        ),
        "completion": (
            "A solo founder told me the calendar change saved them two hours last week.\n\n"
            "I wanted the clever reason.\n\n"
            "The actual reason was that we stopped asking them to schedule the same "
            "thing twice.\n\n"
            "That is the product lesson I keep relearning: people do not need every "
            "workflow to feel magical.\n\n"
            "They need the dumb repeat work to disappear."
        ),
    },
    {
        "id": "support_macro",
        "audience": "support leads",
        "required_terms": ["support macro", "refund policy", "22 tickets"],
        "source_draft": (
            "The support macro for refund policy was causing problems. 22 tickets got "
            "confusing replies. We rewrote the macro and made it less robotic."
        ),
        "completion": (
            "The support macro was technically correct and still bad.\n\n"
            "That is how we ended up with 22 tickets about the refund policy that needed "
            "a second reply.\n\n"
            "The old answer protected the company. The new answer explains the decision "
            "like a person would say it out loud.\n\n"
            "Same refund policy.\n"
            "Less customer punishment."
        ),
    },
    {
        "id": "model_checkpoint",
        "audience": "ML engineers",
        "required_terms": ["checkpoint", "Gemma", "first try"],
        "source_draft": (
            "The checkpoint loaded in Gemma but the first try generation was bad. This "
            "means loading is not enough. The checkpoint has to write well first try."
        ),
        "completion": (
            "A checkpoint loading is not a finish line.\n\n"
            "We proved that the annoying way with Gemma.\n\n"
            "The model loaded. The adapter attached. The first try output still sounded "
            "like a template wearing a blazer.\n\n"
            "So the task changed: train against the actual failure, not the happy path.\n\n"
            "For writing models, first try is the product."
        ),
    },
    {
        "id": "calendar_sync",
        "audience": "operations teams",
        "required_terms": ["calendar sync", "duplicate meetings", "four customers"],
        "source_draft": (
            "Calendar sync had duplicate meetings for four customers. We fixed the sync "
            "logic and added a guard so the same event cannot be imported twice."
        ),
        "completion": (
            "Four customers had duplicate meetings because our calendar sync trusted the "
            "wrong identifier.\n\n"
            "That is a small sentence for a very annoying bug.\n\n"
            "People plan their day from a calendar. If we put the same meeting there "
            "twice, the product stops being a helper and starts being suspicious.\n\n"
            "The guard is in now.\n"
            "Same event, one import, no guessing."
        ),
    },
    {
        "id": "code_review",
        "audience": "engineering teams",
        "required_terms": ["code review", "17 comments", "root cause"],
        "source_draft": (
            "The code review had 17 comments but most of them were style. We missed the "
            "root cause at first. We need fewer comments and better comments."
        ),
        "completion": (
            "A code review with 17 comments can still miss the root cause.\n\n"
            "We did.\n\n"
            "Most of the thread was style cleanup. Useful, maybe. But the real issue was "
            "that the function owned two different decisions and hid the failure between them.\n\n"
            "The better review would have had three comments.\n"
            "One of them would have mattered."
        ),
    },
    {
        "id": "onboarding_email",
        "audience": "SaaS founders",
        "required_terms": ["onboarding email", "day three", "activation"],
        "source_draft": (
            "The onboarding email on day three was hurting activation. It asked people "
            "to read docs instead of asking them to finish the setup."
        ),
        "completion": (
            "Our day three onboarding email was losing the plot.\n\n"
            "It sent people to docs when the only thing that mattered was finishing setup.\n\n"
            "That is how a helpful email becomes a polite escape hatch. It gives the user "
            "something to read instead of helping them do the thing they came for.\n\n"
            "We rewrote it around activation.\n"
            "One action. No homework."
        ),
    },
    {
        "id": "creator_tool",
        "audience": "creator economy founders",
        "required_terms": ["creator tool", "2,300 creators", "template"],
        "source_draft": (
            "The creator tool has 2,300 creators. They were all using the same template "
            "and posts were sounding the same. We need to make it less generic."
        ),
        "completion": (
            "The creator tool worked well enough to create a new problem.\n\n"
            "2,300 creators were using it, and too many posts started sounding like they "
            "came from the same template.\n\n"
            "That is the quiet danger with writing software. Success can make everyone "
            "sound more optimized and less alive.\n\n"
            "The next version has to protect taste, not just speed."
        ),
    },
    {
        "id": "sales_call",
        "audience": "B2B founders",
        "required_terms": ["sales call", "procurement", "nine minutes"],
        "source_draft": (
            "A sales call went badly because the demo took nine minutes to get to "
            "procurement. The buyer cared about procurement and not the rest."
        ),
        "completion": (
            "The sales call did not go badly because the product was weak.\n\n"
            "It went badly because we spent nine minutes avoiding the thing the buyer "
            "actually cared about: procurement.\n\n"
            "We had the answer. We just buried it under a tour of features they had not "
            "asked for yet.\n\n"
            "That is not a demo problem.\n"
            "That is a listening problem."
        ),
    },
    {
        "id": "docs_search",
        "audience": "developer tool builders",
        "required_terms": ["docs search", "exact error", "12 tabs"],
        "source_draft": (
            "Docs search was bad. One user opened 12 tabs to solve an exact error. We "
            "changed ranking so exact errors show up first."
        ),
        "completion": (
            "A user opened 12 tabs to solve one exact error.\n\n"
            "That is a docs search failure, not a power-user habit.\n\n"
            "The old ranking treated the error message like a vague topic. The new one "
            "treats it like the user handing us the answer key.\n\n"
            "Exact error first.\n"
            "Philosophy later."
        ),
    },
    {
        "id": "refund_flow",
        "audience": "operators",
        "required_terms": ["refund flow", "Stripe", "manual note"],
        "source_draft": (
            "The refund flow in Stripe worked but the manual note was missing. Support "
            "could not tell why refunds happened. We added the note."
        ),
        "completion": (
            "The refund flow worked in Stripe and still left support blind.\n\n"
            "That was the bug.\n\n"
            "Money moved correctly, but the manual note never made it over. So two days "
            "later, nobody could tell why the refund happened without reading the whole thread.\n\n"
            "We added the note.\n"
            "Operational memory is a feature."
        ),
    },
]


def prompt_for(seed: dict[str, Any], variant: str) -> str:
    required = ", ".join(seed["required_terms"])
    avoid = ", ".join(GLOBAL_AVOID_TERMS)
    return (
        f"{variant}\n\n"
        f"Rough draft:\n{seed['source_draft']}\n\n"
        f"Keep these anchors: {required}.\n"
        f"Avoid these phrases and their cadence: {avoid}.\n"
        f"{DIRECT_REWRITE_INSTRUCTION}"
    )


def build_examples(seed: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for rewrite in SEED_REWRITES:
        for variant_index, variant in enumerate(PROMPT_VARIANTS):
            examples.append(
                {
                    "id": f"{rewrite['id']}_{variant_index}",
                    "prompt": prompt_for(rewrite, variant),
                    "completion": rewrite["completion"],
                    "platform": "LinkedIn",
                    "audience": rewrite["audience"],
                    "source_draft": rewrite["source_draft"],
                    "required_terms": rewrite["required_terms"],
                    "avoid_terms": GLOBAL_AVOID_TERMS,
                    "style_family": "founder_rewrite",
                }
            )

    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no-slop founder rewrite SFT data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/founder_rewrite_instructions"),
    )
    parser.add_argument("--validation-ratio", type=float, default=0.14)
    parser.add_argument("--seed", type=int, default=73)
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
