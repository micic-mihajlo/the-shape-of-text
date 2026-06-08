# Social Post Fine-Tune Goal

Build a Gemma 4 12B LoRA adapter that writes strong general-purpose social media
posts from compact briefs.

## Success Criteria

- Produces posts with a clear hook, concrete detail, concise progression, and a
  useful ending or call to action.
- Preserves the user's requested topic, audience, claims, and constraints.
- Avoids brittle brand imitation, private identity mimicry, and dependency on a
  single company corpus.
- Improves held-out social-post quality against the base model and SFT-only
  baseline.
- Tracks distribution metrics during training: KL divergence, entropy, MMD, and
  JMQ.

## Training Plan

1. Prepare a corpus with `scripts/prepare_social_corpus.py`.
2. Run a short tiny-model smoke test locally.
3. Run a short Gemma 4 12B QLoRA smoke job on a CUDA host.
4. Train SFT-only for a baseline adapter.
5. Train distribution-aligned adapters with MMD/JMQ warm-in.
6. Compare base, SFT-only, and distribution-aligned outputs on held-out briefs.
7. Publish the best LoRA adapter and its eval report to Hugging Face.

## Eval Set

Use 50-100 held-out briefs covering:

- Launch announcements
- Founder updates
- Product changelogs
- Lessons learned
- Event recaps
- Hiring or community posts
- Technical explanations for non-specialists
- Short Twitter/X-style posts
- Longer LinkedIn-style posts

Each eval item should include topic, audience, desired platform, factual
constraints, and forbidden claims.

The starter eval file is `configs/social_eval_briefs.jsonl`. Expand it to at
least 50 held-out briefs before judging adapter quality.
