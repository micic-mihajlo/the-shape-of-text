# DiffusionGemma A100 Smoke Evidence - 2026-06-12

Remote job: https://huggingface.co/jobs/micic-mihajlo/6a2c4a9c871c005b5352b733

Commit tested: `7d852ee`

Hardware: Hugging Face Jobs `a100-large`

Runtime path:

- Built `ggml-org/llama.cpp` DiffusionGemma PR `pull/24423/head`.
- Built `bin/llama-diffusion-cli` with `CMAKE_CUDA_ARCHITECTURES=80`.
- Downloaded `unsloth/diffusiongemma-26B-A4B-it-GGUF`
  `diffusiongemma-26B-A4B-it-Q4_K_M.gguf`.
- Generated all 10 founder rewrite eval briefs with `--max-attempts 3`.
- Final quality report passed: `10/10`, `failure_rate=0.0`.

The job exited after generation because the Hugging Face token could run jobs
and download models, but could not create or write Hub repos for artifact
upload.

## Quality Report

```json
{
  "failed": 0,
  "failure_rate": 0.0,
  "failures": [],
  "ok": true,
  "passed": 10,
  "settings": {
    "completion_field": "completion",
    "max_prompt_echo_score": 0.35,
    "max_words": 260,
    "min_words": 35,
    "profile": "founder_rewrite",
    "prompt_field": "prompt"
  },
  "total": 10
}
```

## Generated Completions

### rivet_ai_engineers

```text
The engineers on my team who use AI the most aren't the ones shipping the best work.
This is what Rivet told me.
Rivet was built with mostly AI-generated code, and now 2,300 creators are using it.
Either Rivet is wrong, or the product is roasting itself.
I think it is just wrong.
Many struggle because they don't know what problem they are solving. AI just makes that visible faster.
Rivet is still learning. So is every engineer who thinks AI replaced the need to think.
```

### lm_studio_first_try

```text
The model finally loads in LM Studio, but the output still feels like a generic template.

That is not a win.

If a local model requires five retries and a perfect prompt just to write a normal post, the training isn't finished.

Gemma needs to write well on the first try.

We need to stop engineering prompts and start demanding quality.
```

### noticeable_improvement

```text
We just fixed the empty state.

People were getting stuck because they didn't know where to start.

It wasn't a massive overhaul, but users notice the shift in clarity.

Sometimes the most obvious fix is what actually keeps people moving forward.
```

### agent_boundary

```text
An AI agent almost sent a customer email from the wrong environment.
It was supposed to be sandbox-only.
We caught the error before it hit send.
Safety is hard.
Agent safety is permissions first, evals second.
```

### creator_template_problem

```text
Our tool reached 2,300 creators.
But I noticed a problem.
Every post starts looking like the same template.
Speed alone doesn't win if the output feels hollow.
The next version of the product has to protect taste.
```

### pricing_hesitation

```text
We changed the pricing page by making the free plan obvious.
Signups increased 14%.
It was not a funnel trick.
People were hesitating because they could not tell what was included.
Clarity drives conversion. We stopped guessing and started showing what users get.
```

### churn_grouping

```text
Our churn dashboard said 11 customers left for different reasons.
I looked into the exit notes.
Every single one pointed back to the same issue.
The onboarding was unclear after setup.
The data made one root cause look like many separate problems.
We need to fix the flow.
```

### refund_macro

```text
The support macro for refund policy was accurate but bad.
22 tickets needed a second reply because the answer sounded robotic and defensive.
We rewrote it in plain language.
Tone matters.
Now customers understand the process the first time.
Efficiency improved without adding code.
```

### review_missed_root

```text
A code review had 17 comments and still missed the root cause.
Most of the feedback focused on formatting and style.
We missed the architectural flaw entirely.
The real issue was ownership boundaries within the function.
One deep comment is worth more than a dozen shallow nitpicks.
```

### day_three_email

```text
The onboarding email on day three hurt activation.
We were sending users to documentation when they just needed to finish their setup.
It felt like homework.
We rewrote it around one clear action.
Completion rates spiked.
```
