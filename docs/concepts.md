# Concepts

## Authentication

Every request requires an `X-API-Key` header. Keys are stored as SHA-256 hashes; we never persist the raw value. Keys are scoped to your tenant — you can never read or write another tenant's data, and they can never read or write yours.

To rotate a key, request a new one and revoke the old one. Both require contacting us.

## Debate model

A debate consists of:

| Field | Required | Description |
|---|---|---|
| `motion` | yes | The debate topic. Max 500 chars. |
| `starter` | yes | Who speaks first: `"user"` (your student) or `"assistant"` (the AI). |
| `num_rounds` | yes | Number of exchanges. `1`–`3` for parliamentary modes; `1`–`10` for casual. |
| `mode` | no | `"casual"` (default), `"wsdc"`, or `"ap"`. |
| `difficulty` | no | `"beginner"`, `"intermediate"` (default), or `"advanced"`. |
| `external_user_id` | no | Your own student ID. We echo it back on every response; we never resolve it to a person. |
| `metadata` | no | Free-form key/value bag. Max 10 fields. Rendered into the PDF header. |

One **exchange** = one back-and-forth between the student and the AI.

## Modes

- **Casual** — conversational practice. Lighter AI responses, faster, optimized for learning rather than competition. Use this as the default for most students.
- **WSDC** — World Schools Debate Championship format. Tournament-style speeches with formal structure (roadmap, contentions, mechanisms, weighing, sign-off).
- **AP** — American Parliamentary format. Speeches calibrated for an "intelligent voter" judge: stripped of debate jargon, plain-spoken logical force.

WSDC and AP are capped at 3 rounds. Casual goes up to 10.

## Difficulty

- **Beginner** — shorter responses, gentler scoring, encouraging tone, no debate jargon. Score floor of 5.0 so students aren't crushed early.
- **Intermediate** — default. Standard length, standard scoring rubric.
- **Advanced** — tournament-grade. Longer responses, sharper critique, technical examples, fallacies named explicitly, harsh scoring.

## AI models

DebateLab AI uses Claude (Anthropic) and GPT (OpenAI) models. For advanced parliamentary speeches we use larger models that produce more rigorous argumentation; casual practice runs on faster, lighter models. Specific model versions change as we improve quality and cost.

You don't need to think about which model handles your request — it's chosen automatically from the `mode` and `difficulty` you pass.

## external_user_id

If you pass `external_user_id` on debate creation, we store it on the debate row and echo it back on every read. You can later list a single student's debates with:

```
GET /api/v1/debates?external_user_id=student-12345
```

We never resolve this ID to a human. It is your handle for your own student records.

## metadata

A free-form key/value bag, scalar values only, rendered into the PDF header. Use it for whatever you want on the report:

```json
{
  "metadata": {
    "Debater": "Nguyen An",
    "Class": "Advanced Wednesdays",
    "Instructor": "Ms. Linh",
    "Session": "Week 4"
  }
}
```

Limits:

- Max 10 fields per debate
- Keys: max 50 chars, non-empty strings
- Values: max 200 chars, strings or numbers (no nested objects)

Violations return `400 Bad Request`.

## Tenant isolation

Every read and write is scoped to your tenant at the database level. A request that tries to read a debate belonging to another tenant returns `404 Not Found` — indistinguishable from a debate that doesn't exist. There is no API path that exposes cross-tenant data.

Row-level security is enabled on every relevant table as a backstop.

## Rate limits

Default: 15 requests per minute per API key. Sliding-window enforcement.

Exceeding the limit returns `429 Too Many Requests` with a `Retry-After` header indicating when you can retry.

If you expect sustained higher throughput, ask us to raise the limit on your key.

## Data retention

Debates are deleted **7 days** after creation. This applies uniformly to every tenant. Cascade applies — when a debate is deleted, its messages and score are deleted with it.

PDFs are generated on demand from the stored debate; we do not cache them. **After 7 days the `/report.pdf` endpoint returns `404`.** Plan to fetch and store the PDF on your side (or any debate data you want to keep) within the retention window.

`api_usage` rows are retained indefinitely for billing and audit purposes. They contain no student content — only `tenant_id`, `api_key_id`, endpoint path, response status, and latency.

If you have a regulatory need for shorter retention, contact us. We can configure a per-tenant override.

## Versioning

The current API is `v1` (note the `/api/v1/` prefix on every endpoint). Breaking changes will ship under `/api/v2/`. We may add new optional request fields or new response fields under `v1` without warning; treat unknown fields as forward-compatible.
