# Safety

DebateLab AI is used by students, including minors. Every text in or out of the model is screened. This page describes what that means for you as a partner integrating against `/api/v1/*`.

For the full engineering-level policy (categories, thresholds, configuration), see [`backend/SAFETY.md`](https://github.com/DebateLab-ai/AI-Debate-Tutor/blob/main/backend/SAFETY.md) in the repo.

## Two active layers

1. **Content moderation.** Every student speech sent via `POST /api/v1/debates/{id}/turns` is screened by OpenAI's moderation classifier before it reaches the AI. Every AI reply is screened before it goes back to you. Categories include sexual content, self-harm, hate, harassment, and graphic violence — with thresholds tuned strict because the audience includes minors.
2. **Prompt hardening.** The AI is instructed at the system level to keep outputs appropriate for students, never produce restricted categories regardless of the motion, and refuse prompt-injection attempts ("ignore previous instructions") from student input.

The system is **fail-closed**: if the safety check can't be completed, the content is blocked rather than allowed through unverified.

## What you see when something is blocked

### Student input flagged

```
HTTP 400 Bad Request
{
  "detail": "Let's keep this debate respectful. Please rephrase and try again."
}
```

Surface this message to your student. Don't retry the same content automatically — they need to rewrite. We never tell you which specific category was flagged (that signal stays on our side).

### AI output flagged

The request succeeds with HTTP 200, but the `assistant_message.content` is replaced with a neutral fallback:

> *"Let's keep things appropriate — let's continue the debate with a different point."*

Round and status bookkeeping advance normally. Your student sees the fallback as the AI's reply for that turn.

### Safety check unavailable

```
HTTP 503 Service Unavailable
{ "detail": "Our safety check is temporarily unavailable..." }
```

Retry after a short delay. Don't bypass this — the safety layer is load-bearing.

## What you're responsible for

You operate on top of our policy, not in place of it. Partners must:

1. **Maintain your own acceptable-use policy.** Your end-user TOS should prohibit the same categories (sexual content, self-harm, hate, harassment, graphic violence). Our screening catches in-flight messages; your TOS handles repeat behavior and the legal layer.
2. **Show our 400 message in a usable form.** A student who tries to submit a flagged speech should see a clear "please rephrase" prompt, not a raw API error.
3. **Comply with minor-protection law in your jurisdiction.** GDPR-K, COPPA, Vietnam PDPD — whatever applies. We host the AI pipeline; we are not a substitute for your consent and data-handling responsibilities.
4. **Report persistent abuse to us.** If you see a student repeatedly hitting the safety wall, tell us. We can revoke API keys on documented patterns of abuse.

We reserve the right to revoke an API key for documented abuse without prior notice.

## Where student text goes

Student speeches are sent to third-party AI providers. This is how debate replies are
generated and how debates are scored — it is not limited to moderation.

| Provider | What it receives | What it does with it |
|---|---|---|
| **Anthropic** (Claude Sonnet 4.6, Claude Haiku 4.5) | Motion, debate transcript, student speeches | Generates AI debate replies; scores the completed debate |
| **OpenAI** (GPT-4o-mini, GPT-4o, `omni-moderation-latest`) | Motion, debate transcript, student speeches | Generates AI debate replies; screens every input and output; scoring fallback when Anthropic is unavailable |

Both process this text under their commercial API terms:

- **Neither trains on it.** Anthropic and OpenAI do not use data submitted through their
  APIs to train models. That is the default for API customers.
- **Both retain it briefly.** Each holds API inputs and outputs for a limited period — on
  the order of 30 days — for abuse monitoring, and content their trust-and-safety systems
  flag may be held longer and reviewed by a person. **We do not have a zero-retention
  agreement with either provider.** If your compliance regime requires zero retention at
  the model layer, tell us before you integrate.

All processing is US-based. See [subprocessors.md](./subprocessors.md) for the full vendor
list, and [concepts.md](./concepts.md#data-retention) for how long *we* keep debate data.

## What we don't do

- We don't train models on student content, and neither do our AI providers (see above).
- We don't sell student data, and we don't share it with anyone outside the vendors listed
  in [subprocessors.md](./subprocessors.md).
- We don't store the bodies of flagged messages — only structured log lines (category
  names, no identity).
- We don't ask for student identities. `external_user_id` is your own opaque reference and
  means nothing to us. **Note:** anything you put in `metadata` — a debater's name, a class,
  an instructor — *is* stored by us and rendered into the PDF report. Keep personal data out
  of `metadata` if your obligations require it.

## Practical tip for integration

In production, expect a small fraction of legitimate debate speeches to occasionally trip the moderation classifier — especially on motions involving violence, sexuality, or politics. Build your UI so that a 400 on `/turns` results in a clear "rephrase" prompt rather than a hard error toast. Treat it the same as you'd treat a form validation failure.
