# Subprocessors

The third parties that process data on our behalf when you use `/api/v1/*`.

**Last updated: 2026-09-19.**

## Current subprocessors

| Vendor | Purpose | Data it receives | Location |
|---|---|---|---|
| **Anthropic** | AI debate replies; debate scoring | Motion, debate transcript, student speeches | United States |
| **OpenAI** | AI debate replies; content moderation on every input and output; scoring fallback | Motion, debate transcript, student speeches | United States |
| **Supabase** | Primary database (Postgres) | Debates, messages, scores, API usage logs | United States |
| **Railway** | Backend application hosting | All API traffic, in transit and during processing | United States |

All four are US-based. No student data is processed in Vietnam or the EU.

## What each one holds, and for how long

**Anthropic and OpenAI** receive student text because generating and scoring a debate
requires a model to read it. Neither trains on data submitted through their APIs. Both
retain API inputs and outputs for a limited period — on the order of 30 days — for abuse
monitoring, and content flagged by their trust-and-safety systems may be held longer and
reviewed by a person. We do not hold a zero-retention agreement with either provider.

**Supabase** stores debates, messages and scores, which we delete 7 days after creation
(see [concepts.md](./concepts.md#data-retention)). `api_usage` rows are kept indefinitely
for billing and audit; they contain no student content — only tenant ID, key ID, endpoint
path, response status and latency. Data is encrypted in transit and at rest.

**Railway** runs the API process. It holds no data at rest beyond application logs, which
contain no student speech bodies.

## Not in your data path

- **Vercel** hosts the debatelab.ai website. Partner integrations do not touch it — you
  own your own frontend.
- **OpenAI Whisper** transcribes audio on the debatelab.ai website only. The partner API
  has no audio endpoint, so no audio from your students reaches it.

## Changes

We will update this page before adding or replacing a subprocessor. If you need advance
written notice of a change, raise it with us before you integrate and we will agree a
notice period in your contract.

## Questions this page does not answer

If you need a Data Processing Agreement, a zero-retention arrangement at the model layer,
or a specific minimum TLS version in writing, contact us directly. Those are contractual
matters, not defaults.
