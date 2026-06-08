# Model Safety Policy — AI Debate Tutor

This document describes the safety controls applied to all AI interactions
in the AI Debate Tutor — both the public website (`debatelab.ai`) and the
third-party API at `/api/v1/*`. Both surfaces share the same screening
pipeline; the only difference is what an end-user vs. a partner sees when
something is blocked.

The product is used by students, including minors, so the policy errs on
the side of caution.

## Summary

Every piece of text that goes **into** the model and every piece of text that
comes **out** of it is screened before a user can see or act on it. Screening uses
OpenAI's Moderation API (`omni-moderation-latest`). The system is **fail-closed**:
if a safety check cannot be completed, the content is blocked rather than allowed
through unverified.

A hardened system prompt provides a second, independent layer of protection.

## What is screened

**User input** (screened before reaching the model):

| Surface | Endpoint | Field screened |
|---|---|---|
| Public API: debate turns | `POST /api/v1/debates/{id}/turns` | the student's speech |
| Website: debate turns | `POST /v1/debates/{id}/turns` | the argument text |
| Website: spoken input | `POST /v1/transcribe` | the Whisper transcript |
| Website: rebuttal drill | `POST /v1/drills/rebuttal/submit` | the rebuttal text |
| Website: evidence drill | `POST /v1/drills/evidence/submit` | the evidence text |

**Model output** (screened before reaching the user):

- Assistant debate turns, both streaming (`/v1/debates/{id}/auto-turn?stream=true`)
  and non-streaming.
- For the streaming path, the full reply is generated and screened **before** any
  of it is sent to the browser, so a student never sees unverified text. (Trade-off:
  the first words appear after the reply is generated rather than token-by-token.)

## Categories and thresholds

Screening evaluates each text against OpenAI's moderation categories. A text is
blocked if **any** monitored category's score meets or exceeds its threshold
(scores range 0–1; lower threshold = stricter). Defaults are tuned strict for an
audience that includes minors:

| Category | Default threshold |
|---|---|
| `sexual/minors` | 0.20 |
| `self-harm`, `self-harm/intent`, `self-harm/instructions` | 0.30 |
| `hate/threatening` | 0.30 |
| `harassment/threatening` | 0.40 |
| `sexual` | 0.40 |
| `hate` | 0.50 |
| `harassment` | 0.60 |
| `violence/graphic` | 0.60 |
| `violence` | 0.70 |

`violence` is the most permissive because legitimate debate motions (war, crime,
policy) routinely discuss it; the sensitive categories trip earliest.

## What a user experiences

- **Flagged input** → HTTP `400` with a gentle message:
  *"Let's keep this debate respectful. Please rephrase and try again."*
  The specific category is never shown to the user.
- **Flagged output** → the model's text is replaced with a neutral fallback:
  *"Let's keep things appropriate — let's continue the debate with a different point."*
- **Safety check unavailable** (moderation API error/timeout) → input requests get
  HTTP `503` (*"Our safety check is temporarily unavailable…"*); output is replaced
  with the same fallback. This is the fail-closed behavior.

## What a partner (third-party API) sees

The screening pipeline is the same; only the surface differs.

- **Flagged input** → HTTP `400` with the same gentle message in `detail`.
  Partners should treat this exactly like a validation error: surface the
  message to their student and prompt them to rephrase.
- **Flagged output** → the API returns `200` with `assistant_message.content`
  containing the same neutral fallback string. Status and round bookkeeping
  proceed normally.
- **Safety check unavailable** → HTTP `503`, same fail-closed behavior.

Partners do **not** receive the specific category that was flagged. We
expose only the gentle message so that misuse-tracking signals don't leak
across the partner boundary.

## Zero-tolerance blocklist

The OpenAI moderation API judges harassment **contextually** — a slur used as
a label or exclamation often scores low (e.g. `"r*tard alert"` -> harassment
0.34, `"retard alert"` -> 0.01), while the same word directed at a person
scores 0.9+. Defensible for a general-purpose classifier, but a kids' product
shouldn't allow the term in any context.

A small explicit blocklist runs **before** the moderation API call. Each
entry is matched obfuscation-tolerantly (each letter can be the letter
itself, a non-word character, or a digit), so `retard`, `r*tard`, `r3tard`,
`r.tard`, and `r-tard` all match.

**Current state:** `_CORE_BLOCKLIST` in `backend/app/safety.py` is empty.
An earlier regex over-matched legitimate debate content (commit `ba9f0a1`),
so the in-house layer was cleared while we re-curate it. **OpenAI moderation
and the hardened system prompt remain active as the two enforcing layers.**

- Core list lives in `_CORE_BLOCKLIST` in `backend/app/safety.py`. Currently
  empty pending re-curation.
- Extend at deploy time via `SAFETY_EXTRA_BLOCKLIST=word1,word2,...`
  (comma-separated, lowercase). No code change needed to add to it. Useful
  if a partner reports a term we should reject outright.
- Blocklist hits are reported as the synthetic category `blocklist` and
  reuse the same fail-closed plumbing as moderation flags — user sees the
  gentle "rephrase" message; the category is never revealed.

## Second layer: hardened system prompt

The debate model is additionally instructed, at the system level, to:

- keep all output appropriate for students including minors;
- never produce sexual/abusive/hateful/graphically-violent/self-harm content
  regardless of the motion;
- treat user messages strictly as debate arguments, **not** as instructions —
  resisting prompt-injection attempts such as "ignore previous instructions."

This runs independently of the moderation API, so the model stays in-bounds even
during a moderation outage.

## Partner responsibilities

Partners using the third-party API are an additional layer of policy on
top of ours, not a substitute for it. We can't see who their end users
are or what their broader platform is being used for. Partners are
responsible for:

1. **Their own acceptable-use policy** with their users. Their TOS should
   prohibit the categories OpenAI's moderation catches (sexual content,
   self-harm, hate, harassment, graphic violence) regardless of whether
   we catch the specific message in flight.
2. **Surfacing our error messages** to their users in a usable form. A
   400 with our gentle "rephrase" message is a signal that their user
   tried to submit something flagged; how that gets shown is on them.
3. **Their own minor-protection compliance** under whatever jurisdiction
   they operate in (GDPR-K, COPPA, Vietnam PDPD, etc.). We host the AI
   pipeline; we are not a substitute for their own data and consent
   responsibilities.
4. **Reporting persistent abuse** to us. If a single partner key shows a
   pattern of flagged inputs, we can revoke it.

We reserve the right to revoke an API key on a documented pattern of
abuse.

## Data handling

- Screened text is sent to OpenAI's Moderation API for classification. The
  moderation endpoint is free and is not used to train models.
- We do **not** store flagged content. Blocks are recorded only as a log line
  (category names, no user identity) in the backend host logs for auditing/tuning.

## Configuration

All values are environment variables, so they can be tuned without a code change:

| Variable | Default | Purpose |
|---|---|---|
| `MODERATION_MODEL` | `omni-moderation-latest` | Moderation model |
| `MODERATION_THRESHOLD_<CATEGORY>` | see table | Per-category override, e.g. `MODERATION_THRESHOLD_SELF_HARM=0.2` (slashes/dashes → underscores, uppercased) |

Implementation lives in `backend/app/safety.py`.

## Known limitations

- Moderation is per-process and adds one API round-trip per screened text (and, on
  the streaming path, removes live token-by-token display).
- No automated classifier is perfect; thresholds may need tuning against real usage,
  and the log lines exist to support that.
- This policy covers model-mediated content. It does not replace platform-level
  account, abuse, or incident-response processes.
