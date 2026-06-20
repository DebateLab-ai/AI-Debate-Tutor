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
