# DebateLab AI — API Documentation

A server-side API for building debate-practice tools. Send a motion, a side, and a student's argument; we return an AI counter-speech, a final score, and a downloadable PDF report.

## Base URL

```
https://api.debatelab.ai
```

## Quick links

- **[Getting started](./getting-started.md)** — auth, your first call, a full debate lifecycle in 10 minutes
- **[Concepts](./concepts.md)** — modes, difficulty tiers, metadata, student IDs, rate limits
- **[API reference](./reference.md)** — every endpoint, every parameter, every response shape
- **[Errors](./errors.md)** — status codes and error response format
- **[Examples](./examples/)** — full debate lifecycle in [curl](./examples/curl.md), [Python](./examples/python.md), and [JavaScript](./examples/javascript.md)

## What you can build

- An embedded debate practice flow inside your existing student dashboard
- A standalone debate trainer in your own visual style
- A coach-facing tool that issues PDF reports to instructors after every practice round

You bring the UI. We handle motion generation, AI counter-speeches, scoring, and the report.

## Versioning

The API is `v1`. Breaking changes will ship under `v2`. Additive changes (new optional fields, new endpoints) will not bump the version.

## Support

Email the contact who issued your API key. We're a small team; we'll get back to you within a business day.
