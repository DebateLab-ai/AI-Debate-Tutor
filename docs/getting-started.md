# Getting started

## Get an API key

API keys are issued manually. Email us with:

- The name of your organization
- A short description of what you're building
- Expected monthly debate volume (rough estimate is fine)

You'll receive a key that looks like `sk_live_…`. We show it to you once — store it. We never see the raw key again; we only store its SHA-256 hash.

## Authenticate

Pass the key in the `X-API-Key` header on every request.

```bash
curl https://api.debatelab.ai/api/v1/debates \
  -H "X-API-Key: sk_live_..."
```

A missing or invalid key returns `401 Unauthorized`.

## Your first call

Create a debate.

```bash
curl https://api.debatelab.ai/api/v1/debates \
  -H "X-API-Key: sk_live_..." \
  -H "Content-Type: application/json" \
  -d '{
    "motion": "THW ban single-use plastics globally",
    "starter": "user",
    "num_rounds": 2,
    "mode": "casual",
    "difficulty": "intermediate"
  }'
```

You'll get back the new debate's state. Note the `id` — you'll use it for every subsequent call.

```json
{
  "id": "8c2f...",
  "motion": "THW ban single-use plastics globally",
  "starter": "user",
  "next_speaker": "user",
  "current_round": 1,
  "num_rounds": 2,
  "status": "active",
  "mode": "casual",
  "difficulty": "intermediate",
  "metadata": {},
  "created_at": "2026-06-07T12:00:00Z",
  "updated_at": "2026-06-07T12:00:00Z"
}
```

## A full debate lifecycle

A debate is four steps:

1. **Create** the debate with a motion and configuration.
2. **Open** (only if `starter` is `"assistant"`) — one AI opening speech via `POST /api/v1/debates/{id}/open`.
3. **Submit turns** until the debate is complete. Each turn = your student's speech in; AI counter-speech out.
4. **Finish** the debate to score it.
5. **Download the PDF report** for the student or instructor.

Pass an `Idempotency-Key` header on `/open` and `/turns` so retries after timeouts return the same successful response without double-charging the model.

### 1. Create

`POST /api/v1/debates` (above).

### 2. Submit a turn

```bash
curl https://api.debatelab.ai/api/v1/debates/8c2f.../turns \
  -H "X-API-Key: sk_live_..." \
  -H "Content-Type: application/json" \
  -d '{ "content": "Plastic waste is choking our oceans..." }'
```

The response includes both the student's saved message and the AI's reply. AI generation typically takes 5–15 seconds; use a long request timeout (60s recommended).

### 3. Finish

When `status` is `"completed"`, call `/finish` to generate the score.

```bash
curl -X POST https://api.debatelab.ai/api/v1/debates/8c2f.../finish \
  -H "X-API-Key: sk_live_..."
```

Returns a score breakdown with overall, per-metric scores (content/structure, engagement, strategy), and per-metric feedback text.

### 4. Get the PDF

```bash
curl https://api.debatelab.ai/api/v1/debates/8c2f.../report.pdf \
  -H "X-API-Key: sk_live_..." \
  --output report.pdf
```

Two sections: feedback and next steps, then a full transcript.

## What's next

- **[Concepts](./concepts.md)** — what `mode`, `difficulty`, and `metadata` actually do
- **[API reference](./reference.md)** — every parameter and response field
- **[Examples](./examples/)** — full lifecycle in Python and JavaScript
