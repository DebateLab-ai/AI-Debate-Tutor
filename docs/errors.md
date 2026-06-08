# Errors

## Response shape

Errors return a non-2xx HTTP status and a JSON body with a `detail` field describing what went wrong.

```json
{
  "detail": "Debate must be scored before a report can be generated. Call /finish first."
}
```

Don't parse the `detail` string for logic — it is meant for humans. Branch on the HTTP status instead.

## Status codes

| Status | Meaning |
|---|---|
| `200` | Success |
| `201` | Created (only on `POST /api/v1/debates`) |
| `400` | Your request was malformed or violated a constraint (oversized metadata, wrong number of rounds for the mode, etc.) |
| `401` | The `X-API-Key` header is missing or invalid |
| `403` | Your account or API key has been disabled. Contact support. |
| `404` | The debate doesn't exist, or it belongs to another tenant |
| `409` | The action conflicts with the current debate state (e.g. it's the AI's turn, not the student's) |
| `422` | Request body failed validation (missing required field, wrong type) |
| `429` | Rate limit exceeded. The `Retry-After` header indicates when to retry. |
| `502` | An upstream AI provider failed; the request is safe to retry |
| `503` | A subsystem (e.g. PDF rendering) is temporarily unavailable |

## Common scenarios

### "Debate is completed"

You called `/turns` on a finished debate. Check `status` from the previous response before submitting another turn.

### "It is the assistant's turn"

The previous `/turns` call already advanced past the student's slot. Re-read the debate state with `GET /api/v1/debates/{id}` to recover.

### "Debate must be scored before a report can be generated"

Call `POST /api/v1/debates/{id}/finish` before requesting the PDF.

### "metadata may have at most 10 fields"

You exceeded one of the [metadata limits](./concepts.md#metadata): 10 fields max, 50-char keys, 200-char values, no nested objects.

### `502` on `/turns` or `/finish`

The AI provider (Claude or GPT) returned an error. Retry once. If it keeps failing on the same debate, capture the debate ID and email us.

### `429`

Your API key is sending requests faster than the rate limit allows. The default is 15 requests/minute. Back off using the `Retry-After` header value, or contact us about raising the limit.

## What we don't expose

- Internal stack traces
- Vendor (Anthropic / OpenAI) error messages
- Other tenants' existence

If you need debugging help, include the failing request body, the response status, and the response body in your email. We can correlate with our logs from there.
