# API reference

All endpoints share the prefix `/api/v1/`. All require the `X-API-Key` header. All scope by your tenant.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/debates` | Create a new debate |
| `POST` | `/api/v1/debates/{id}/open` | Generate the AI opening (when `starter` is `"assistant"`) |
| `POST` | `/api/v1/debates/{id}/turns` | Submit a student turn; get an AI reply |
| `POST` | `/api/v1/debates/{id}/finish` | Score the completed debate |
| `GET` | `/api/v1/debates/{id}` | Fetch a debate's full state |
| `GET` | `/api/v1/debates` | List debates with filters |
| `GET` | `/api/v1/debates/{id}/report.pdf` | Download the PDF report |
| `GET` | `/api/v1/usage` | Calls and amount due for a month, or all time |
| `GET` | `/api/v1/usage/events` | Per-call cost lines for that window |
| `POST` | `/api/v1/drills/rebuttal/start` | Generate a drill claim for the student to rebut |
| `POST` | `/api/v1/drills/rebuttal/submit` | Score a drill rebuttal; get the next claim |

---

## Create a debate

```
POST /api/v1/debates
```

Request body:

| Field | Type | Required | Description |
|---|---|---|---|
| `motion` | string | yes | The debate topic. 1–500 chars. |
| `starter` | `"user"` \| `"assistant"` | yes | Who speaks first. |
| `num_rounds` | integer | yes | 1–10 (1–3 for `wsdc` / `ap`). |
| `mode` | `"casual"` \| `"wsdc"` \| `"ap"` | no | Default `"casual"`. |
| `difficulty` | `"beginner"` \| `"intermediate"` \| `"advanced"` | no | Default `"intermediate"`. |
| `external_user_id` | string | no | Your student ID. Max 128 chars. |
| `metadata` | object | no | Free-form fields for the PDF header. See [Concepts](./concepts.md#metadata). |

Returns `201 Created` with the debate object:

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
  "external_user_id": "student-123",
  "metadata": { "Debater": "Nguyen An" },
  "created_at": "2026-06-07T12:00:00Z",
  "updated_at": "2026-06-07T12:00:00Z"
}
```

Common errors:

- `400` — `num_rounds > 3` with parliamentary mode; oversized or malformed `metadata`
- `401` — missing or invalid API key
- `429` — rate limited

---

## Open a debate (AI speaks first)

```
POST /api/v1/debates/{debate_id}/open
```

Required when you created the debate with `"starter": "assistant"`. Generates the AI opening speech. Skip this endpoint when `"starter": "user"` — go straight to `/turns`.

Optional header: `Idempotency-Key` (recommended). Re-sending the same key returns the cached `200` response without re-running the model.

AI generation takes 5–15 seconds; use a 60-second client timeout.

Returns `200 OK`:

```json
{
  "debate_id": "8c2f...",
  "assistant_message": {
    "id": "...",
    "round_no": 1,
    "speaker": "assistant",
    "content": "The proposition stands for...",
    "created_at": "2026-06-07T12:00:08Z"
  },
  "current_round": 1,
  "next_speaker": "user",
  "status": "active"
}
```

Common errors:

- `400` — debate is already `"completed"`
- `404` — debate not found (or belongs to another tenant)
- `409` — it is not the assistant's turn (already opened, or wrong `starter`)
- `502` — AI generation failed; safe to retry (no partial state is left behind)

---

## Submit a turn

```
POST /api/v1/debates/{debate_id}/turns
```

Submits one student speech and generates the AI counter-speech. AI generation takes 5–15 seconds; use a 60-second client timeout.

Optional header: `Idempotency-Key` (recommended). Re-sending the same key after a successful turn returns the cached response.

Request body:

| Field | Type | Required | Description |
|---|---|---|---|
| `content` | string | yes | The student's speech. 1–10,000 chars. |

Returns `200 OK`:

```json
{
  "debate_id": "8c2f...",
  "user_message": {
    "id": "...",
    "round_no": 1,
    "speaker": "user",
    "content": "Plastic waste is...",
    "created_at": "2026-06-07T12:01:00Z"
  },
  "assistant_message": {
    "id": "...",
    "round_no": 1,
    "speaker": "assistant",
    "content": "The opposition concedes the harm...",
    "created_at": "2026-06-07T12:01:08Z"
  },
  "current_round": 2,
  "next_speaker": "user",
  "status": "active"
}
```

When the student's final turn ends the debate without an AI reply, `assistant_message` is `null`.

When the debate completes, `status` becomes `"completed"` and `next_speaker` is `null`.

Common errors:

- `400` — debate is already `"completed"`
- `404` — debate not found (or belongs to another tenant)
- `409` — it is not the student's turn (call `/open` first if `next_speaker` is `"assistant"`)
- `502` — AI generation failed; safe to retry with the same body (partial writes are rolled back). Use `Idempotency-Key` after a successful turn to dedupe retries.

---

## Finish (score) a debate

```
POST /api/v1/debates/{debate_id}/finish
```

Generates the score and marks the debate `"completed"`. Idempotent — repeated calls return the same score.

No request body.

Returns `200 OK`:

```json
{
  "debate_id": "8c2f...",
  "overall": 6.5,
  "metrics": {
    "content_structure": 7.0,
    "engagement": 6.0,
    "strategy": 6.5
  },
  "feedback": "Strong opening with the Vietnam livelihoods example...",
  "content_structure_feedback": "Round 1 is well-signposted...",
  "engagement_feedback": "Direct clash on the displacement claim...",
  "strategy_feedback": "Smart pivot to weighing in Round 2...",
  "weakness_type": "weighing"
}
```

`weakness_type` is one of `"rebuttal"`, `"structure"`, `"weighing"`, `"evidence"`, `"strategy"`. The PDF uses this to recommend a next drill.

Scoring takes 3–8 seconds.

Common errors:

- `400` — debate has no turns yet
- `404` — debate not found
- `502` — scoring failed; safe to retry

---

## Get a debate

```
GET /api/v1/debates/{debate_id}
```

Returns the full debate state — debate object, all messages, and the score (if scored).

```json
{
  "id": "8c2f...",
  "motion": "...",
  "starter": "user",
  "next_speaker": null,
  "current_round": 2,
  "num_rounds": 2,
  "status": "completed",
  "mode": "casual",
  "difficulty": "intermediate",
  "external_user_id": "student-123",
  "metadata": { "Debater": "Nguyen An" },
  "created_at": "2026-06-07T12:00:00Z",
  "updated_at": "2026-06-07T12:01:08Z",
  "messages": [
    { "id": "...", "round_no": 1, "speaker": "user", "content": "...", "created_at": "..." },
    { "id": "...", "round_no": 1, "speaker": "assistant", "content": "...", "created_at": "..." }
  ],
  "score": {
    "debate_id": "8c2f...",
    "overall": 6.5,
    "metrics": { "content_structure": 7.0, "engagement": 6.0, "strategy": 6.5 },
    "feedback": "...",
    "content_structure_feedback": "...",
    "engagement_feedback": "...",
    "strategy_feedback": "...",
    "weakness_type": "weighing"
  }
}
```

`score` is `null` until `/finish` has been called.

Common errors:

- `404` — debate not found

---

## List debates

```
GET /api/v1/debates
```

Lists debates owned by your tenant, newest first.

Query parameters:

| Param | Type | Description |
|---|---|---|
| `external_user_id` | string | Return only debates for this student ID. |
| `since` | ISO-8601 timestamp | Return only debates created at or after this time. |
| `limit` | integer | 1–200. Default 50. |

Returns `200 OK` with an array of debate objects (same shape as the create response, no messages or score embedded).

Common pattern — pull all of one student's debates from the last week:

```
GET /api/v1/debates?external_user_id=student-123&since=2026-06-01T00:00:00Z
```

---

## Rebuttal drill

Stateless practice loop (same logic as the website's `/debate-drill-rebuttal`). No Supabase storage — your server keeps `claim` / `next_claim` between calls.

Typical flow after a debate: read `weakness_type` from `/finish`, pass it to `/start`, then loop `/submit`.

### Start drill

```
POST /api/v1/drills/rebuttal/start
```

```json
{
  "motion": "THW ban single-use plastics globally",
  "user_position": "for",
  "weakness_type": "rebuttal",
  "external_user_id": "student-123"
}
```

Returns `200 OK`:

```json
{
  "claim": "Single-use plastics are essential for medical sterility...",
  "claim_position": "against",
  "weakness_type": "rebuttal"
}
```

`claim_position` is always opposite `user_position`.

### Submit rebuttal

```
POST /api/v1/drills/rebuttal/submit
```

```json
{
  "motion": "THW ban single-use plastics globally",
  "claim": "...",
  "claim_position": "against",
  "rebuttal": "The proposition overstates medical dependence...",
  "weakness_type": "rebuttal"
}
```

Returns `200 OK`:

```json
{
  "overall_score": 6.5,
  "metrics": {
    "refutation_quality": 7.0,
    "evidence_examples": 6.0,
    "impact_comparison": 6.5
  },
  "feedback": "...",
  "next_claim": "...",
  "next_claim_position": "against",
  "weakness_type": "rebuttal"
}
```

Use `next_claim` / `next_claim_position` for the next practice round.

Common errors:

- `502` — claim generation or scoring failed; safe to retry

---

## Download PDF report

```
GET /api/v1/debates/{debate_id}/report.pdf
```

Returns the PDF report as raw bytes with `Content-Type: application/pdf` and `Content-Disposition: attachment; filename="debate_report_<id>.pdf"`.

The report has two sections:

1. **Feedback and next steps** — overall score, per-metric breakdown, judge's prose feedback, and one recommended next drill keyed off the score's `weakness_type`.
2. **Transcript** — round-by-round, speaker-labeled.

The PDF header renders any `metadata` fields you passed on debate creation (Debater, Class, Instructor, etc.).

Common errors:

- `400` — debate has not been scored; call `/finish` first
- `404` — debate not found, or older than the [retention window](./concepts.md#data-retention)
- `503` — PDF generation temporarily unavailable

> Debates are deleted after 7 days. If you want to keep PDFs longer, download and store them on your side within the retention window.

---

## Usage

```
GET /api/v1/usage
GET /api/v1/usage/events
```

Usage is available on your existing API key. Send it as `X-API-Key` on every request.

Money fields are decimal strings, not JSON numbers. That includes `amount_due_usd`, `amount_usd`, `amount_without_call_id_usd`, and each `amount_usd` inside `spend_by_kind`. Parse them with a decimal type. A float can change the 8th decimal place.

`GET /api/v1/usage` is the statement. `amount_due_usd` is the amount due for that window, in USD, to 8 decimal places. It is the vendor cost we recorded for each model call that the request delivered, with no markup. A model call that fails, and a request that does not succeed, are not recorded as charges. That number is the invoice.

`period` chooses the window. All times are UTC.

- Leave `period` off for the current month.
- `period=YYYY-MM` is that month, including the current one. During October 2026, `?period=2026-10` returns the same window as leaving `period` off: from `2026-10-01T00:00:00Z` inclusive to `2026-11-01T00:00:00Z` exclusive. A future month returns 400.
- `period=all` is every recorded charge up to this request. `period_start` is null. `period_end` is the UTC time of the request, and it is exclusive. A charge recorded at that exact instant waits for a later request.

```
GET /api/v1/usage?period=2026-10
```

```json
{
  "period": "2026-10",
  "period_start": "2026-10-01T00:00:00Z",
  "period_end": "2026-11-01T00:00:00Z",
  "currency": "USD",
  "amount_due_usd": "4.28190000",
  "monthly_cap_usd": null,
  "unpriced_events": 0,
  "per_call_available": true,
  "amount_without_call_id_usd": "1.25000000",
  "requests": 140,
  "requests_with_call_id": 120,
  "requests_by_endpoint": [
    { "endpoint": "POST /api/v1/debates/{id}/turns", "count": 80, "errors": 1 },
    { "endpoint": "POST /api/v1/debates/{id}/finish", "count": 25, "errors": 0 },
    { "endpoint": "POST /api/v1/debates", "count": 15, "errors": 0 },
    { "endpoint": "GET /api/v1/debates/{id}", "count": 12, "errors": 0 },
    { "endpoint": "GET /api/v1/debates", "count": 8, "errors": 0 }
  ],
  "spend_by_kind": [
    { "kind": "speech", "model_calls": 80, "amount_usd": "3.90000000" },
    { "kind": "score", "model_calls": 20, "amount_usd": "0.38190000" }
  ]
}
```

The same call with `?period=all` keeps every other field and sets `period` to `"all"`, `period_start` to null, and `period_end` to the time of that request.

`amount_due_usd` is the total for that period. It includes charges recorded before this usage API existed. In `spend_by_kind` the `amount_usd` values add up to `amount_due_usd`. One student turn is one HTTP request and can include several model calls, so `model_calls` can be higher than `requests`. Kinds are `speech`, `score`, `drill`, and `moderation`.

`monthly_cap_usd` is your monthly cap, or null if you have none. The cap is always the current month, including when `period` is `all`.

`requests` counts HTTP calls in the window, including calls from before per-call lines existed, plus reads and retries. In `requests_by_endpoint`, the `count` values add up to `requests`. `errors` counts recorded responses with status 400 or higher. A missing or invalid key (`401`), a revoked key or disabled account (`403`), and a rate-limit rejection (`429`) do not create a row, so they are not in `requests` or `errors`.

`requests_with_call_id` counts only calls that have a per-call line. `amount_without_call_id_usd` is the part of `amount_due_usd` with no line in this window, including calls made before per-call lines existed. It is already inside the total. `unpriced_events` should be 0. A non-zero count means those calls are in the total at zero until we reprice them. `per_call_available` is true when per-call lines can be read.

`GET /api/v1/usage/events` is the per-call ledger, newest first. It takes the same `period` values. Each line is one HTTP call made after per-call ids existed. `amount_usd` is every model charge that call produced, added together, and only charges inside the same window. A create, a read, a failed request, or a retry that sends the same `Idempotency-Key` after a successful turn is `"0.00000000"`. Calls from before per-call ids are not in this list.

`limit` is 1–200 and defaults to 50. `offset` defaults to 0. Read further pages until a page comes back shorter than `limit`.

For the same period, after every page:

```
sum of each line's amount_usd + amount_without_call_id_usd = amount_due_usd
```

Parse the strings as decimals before adding. One page is only part of the sum.

```json
{
  "period": "2026-10",
  "limit": 50,
  "offset": 0,
  "events": [
    {
      "request_id": "7b1e...",
      "created_at": "2026-10-04T15:12:08Z",
      "endpoint": "POST /api/v1/debates/{id}/turns",
      "response_status": 200,
      "debate_id": "8c2f...",
      "amount_usd": "0.02740000",
      "spend_by_kind": [
        { "kind": "speech", "model_calls": 2, "amount_usd": "0.02740000" }
      ]
    }
  ]
}
```

Debates are deleted 7 days after they are created. The charge and the event line are kept. On an older line, `debate_id` can point at a debate that has already been deleted, and `GET /api/v1/debates/{id}` then returns 404.

Both reads count toward the 15 requests per minute limit. They stay available if a monthly cap is reached, so you can still see what you owe.

A bad period returns 400. A missing or invalid key returns 401. If the ledger cannot be read completely, the response is 503 rather than a short total. Retry that.
