# CLAUDE.md — AI Debate Tutor (DebateLab)

## LIVE SITE WARNING

**debatelab.ai is a live production site.** This repo is the source. Changes merged to `main` and pushed may trigger automatic deployments. Treat every change as potentially affecting real users.

Before making any changes:
- Understand what is currently deployed vs. what is local
- Test locally first; never use the production environment as a test bed
- Avoid force-pushes to `main`
- If a change could break the UI or the AI response pipeline, flag it before proceeding

---

## Active Initiative: Public Third-Party API

**Branch:** `feature/third-party-debates` (8 commits ahead of main as of 2026-06-07). UI polish lives on `ui/exchanges-and-selector-polish`. Neither is merged.

**Goal:** Let third-party developers build their own debate UIs powered by our backend. Server-side API only — they call us, they own the frontend. First partner: SuperJuniors (debate academy in Vietnam, pro-bono).

### Done
- **Tenant-scoped storage** — Supabase tables `debates`, `messages`, `scores` with `tenant_id` on every row and ON DELETE CASCADE. RLS enabled. See `backend/db/schema.sql`.
- **Six public endpoints** under `/api/v1/*` — create / submit-turn / finish-score / get / list / report.pdf. All require `X-API-Key`, all scope by `auth.tenant_id`. See `backend/app/api_v1.py`.
- **PDF report generation** — two-section partner report (feedback + next drill, then transcript). WeasyPrint + Jinja2 template. See `backend/app/pdf.py` and `backend/app/templates/debate_report.html`.
- **Haiku scoring with GPT-4o fallback** — Anthropic Haiku 4.5 is primary for `compute_debate_score`; falls back to OpenAI on any Anthropic failure. See `_call_scoring_llm` in `backend/app/main.py`.
- **Cost optimizations** — intermediate WSDC Pass 2 swapped Sonnet → Haiku, `rag_top_k` 6 → 4 for intermediate, prompt-cache hooks on Pass 1 (caching itself fires below threshold currently — see COST.md note).
- **7-day retention** — daily cleanup via `.github/workflows/retention-cleanup.yml` running `backend/scripts/run_retention.py`. `api_usage` rows are kept (billing/audit).
- **Partner-facing docs** — `docs/` directory, eight markdown files, Anthropic-style terse voice. Reveals brand-level AI detail only (Claude/GPT named, no two-pass pipeline disclosed).
- **Safety policy refresh** — `backend/SAFETY.md` updated for API surface; `docs/safety.md` is the partner-facing summary.

### What's left when you pick this back up
- **Tier 2 hardening** (not blocking integration):
  - Consistent error response shape across endpoints (currently FastAPI's default `{"detail": "..."}`)
  - Cache-hit logging on Anthropic Pass 1 to verify whether prompt caching fires
  - Per-key daily cost cap (defensive — prevent runaway loop spend)
- **Legal docs** (Tier 1 but deferred per Rico's call):
  - `TERMS.md` (Terms of Use)
  - `ACCEPTABLE_USE.md` (Acceptable Use Policy)
- **WSDC advanced Pass 2 → Haiku A/B test** — only intermediate is swapped today. Advanced still on Sonnet.

### Constraints
- Partners must NEVER hit endpoints that mutate or expose other tenants' data. Every authenticated endpoint scopes reads/writes by `auth.tenant_id`.
- The frontend's in-memory `DEBATES` dict in `backend/app/main.py` is the website's storage and is intentionally NOT touched by the public API code path. Two storage paths run in parallel. Don't conflate them.
- The frontend does NOT use API keys — it talks to the same backend via the older `/v1/*` endpoints (no `/api/` prefix). Don't break that.

### Waiting on external (not on us)
- PDPD / data-residency answer from SuperJuniors (US-hosted Supabase OK or not?)
- Realistic monthly debate volume + mode mix from SuperJuniors (determines whether the casual-default + tighter caps preset gets applied)
- Lawyer review of the legal docs once drafted

### Pre-merge checklist for someone-with-access
1. Set GitHub repo secrets `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` (powers the retention workflow).
2. Confirm the Railway build picks up `backend/nixpacks.toml` and Pango/Cairo install cleanly. If it fails, the package names may need updating — see comments in the file.
3. Confirm the API is publicly reachable at whatever URL we use (`https://api.debatelab.ai` is the placeholder used throughout `docs/`).
4. Manually provision a tenant + api_key row for SuperJuniors before they integrate (no self-serve flow exists yet).

### Key files for this initiative
| File | Purpose |
|---|---|
| `backend/app/api_v1.py` | The 6 public endpoints |
| `backend/app/auth.py` | API key auth (SHA-256 hash lookup) → `AuthContext` |
| `backend/app/debates_store.py` | Tenant-scoped Supabase CRUD; every fn requires `tenant_id` |
| `backend/app/pdf.py` | Lazy-imported PDF renderer; 503 if WeasyPrint sys-deps missing |
| `backend/app/templates/debate_report.html` | Jinja2 template for the PDF |
| `backend/app/retention.py` | 7-day cleanup function |
| `backend/app/ratelimit.py` | Per-API-key sliding window rate limit |
| `backend/app/usage.py` | Background-task usage logger → `api_usage` |
| `backend/app/safety.py` | Input moderation + (currently empty) blocklist |
| `backend/db/schema.sql` | All Supabase DDL (idempotent) |
| `backend/nixpacks.toml` | Railway build deps for WeasyPrint |
| `backend/scripts/smoke_api_v1_lifecycle.py` | End-to-end smoke test (15 assertions; `--with-ai` for full path) |
| `backend/scripts/measure_turn_cost.py` | Reproducible per-turn cost measurement |
| `backend/scripts/compare_scoring_models.py` | A/B harness for scoring models |
| `backend/scripts/run_retention.py` | Manual / cron-invoked retention cleanup |
| `docs/` | Partner-facing markdown docs (README + 7 more files) |
| `.github/workflows/retention-cleanup.yml` | Daily retention cron |
| `COST.md` | Empirical cost report + SuperJuniors briefing |
| `partner_reply_draft.txt` | Draft email reply to SuperJuniors (working doc) |

---

## Project Structure

```
AI-Debate-Tutor/
├── frontend/          # React + Vite SPA, deployed to Vercel
│   ├── src/
│   ├── public/
│   ├── vercel.json    # SPA rewrite rule (all routes → index.html)
│   └── package.json
├── backend/           # FastAPI app, deployed to a separate host
│   ├── app/
│   │   ├── main.py            # FastAPI entrypoint + /v1/* (website API)
│   │   ├── api_v1.py          # Public /api/v1/* router (partner API)
│   │   ├── auth.py            # X-API-Key auth
│   │   ├── debates_store.py   # Tenant-scoped Supabase CRUD
│   │   ├── pdf.py             # PDF report renderer
│   │   ├── retention.py       # 7-day cleanup
│   │   ├── safety.py          # Moderation + blocklist (currently empty)
│   │   ├── ratelimit.py       # Per-key sliding-window rate limiter
│   │   ├── usage.py           # Background-task usage logger
│   │   ├── difficulty.py      # Per-tier config (beginner/intermediate/advanced)
│   │   ├── wsdc.py            # Two-pass WSDC/AP speech pipeline (Anthropic)
│   │   ├── response.py        # OpenAI debate response (single-pass + RAG)
│   │   ├── transcriber_processor.py  # Whisper audio → text
│   │   ├── corpus/            # RAG corpus (debate speech texts)
│   │   └── templates/         # Jinja2 templates (PDF)
│   ├── db/
│   │   └── schema.sql         # Supabase DDL (idempotent)
│   ├── scripts/               # Smoke tests + measurement scripts
│   ├── Procfile               # uvicorn app.main:app --host 0.0.0.0 --port $PORT
│   ├── nixpacks.toml          # Railway build config (Pango/Cairo for WeasyPrint)
│   ├── requirements.txt
│   └── SAFETY.md              # Engineering-internal safety policy
├── docs/                       # Partner-facing API docs (markdown)
├── .github/workflows/          # GitHub Actions (retention-cleanup, etc.)
├── COST.md                     # Cost report + SuperJuniors briefing
└── partner_reply_draft.txt     # Working doc for SuperJuniors comms
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, react-router-dom v7, @vercel/analytics, Space Grotesk (Google Fonts) |
| Backend | FastAPI, Uvicorn, Pydantic v1, python-dotenv, supabase-py |
| AI generation | OpenAI (GPT-4o, GPT-4o-mini), Anthropic (Claude Sonnet 4.6, Claude Haiku 4.5) |
| AI scoring | Anthropic Haiku 4.5 (default), GPT-4o (fallback) |
| AI transcription | OpenAI Whisper-1 |
| AI moderation | OpenAI `omni-moderation-latest` |
| PDF rendering | WeasyPrint + Jinja2 |
| Database | Supabase (Postgres) |
| Deployment (frontend) | Vercel (auto-deploys on push to `main`) |
| Deployment (backend) | Railway via Procfile + nixpacks.toml |

---

## Routing — when does each AI model run?

| Path | Used for |
|---|---|
| Anthropic Sonnet 4.6 (Pass 1) → Haiku 4.5 (Pass 2) | WSDC/AP, intermediate–advanced. Two-pass: structured JSON outline → rendered prose |
| OpenAI GPT-4o-mini | Casual mode, beginner WSDC, any fallback when Anthropic two-pass fails |
| Anthropic Haiku 4.5 (default) / GPT-4o (fallback) | Scoring (every `/finish` call) |
| OpenAI Whisper-1 | Audio transcription on the website |
| OpenAI `omni-moderation-latest` | Every input + output safety screen |

The routing logic lives in `backend/app/main.py:generate_ai_turn_text`. Models are selected by `(mode, difficulty)` — partners don't pick models directly.

---

## Local Development

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Requires `.env` in `backend/` — see env vars table below.

### Frontend

```bash
cd frontend
npm install
npm run dev      # starts Vite dev server (port 5173 or 3001 if taken)
```

The frontend dev server proxies API calls to `localhost:8000` (confirm in `vite.config.js`).

### Running smoke tests

```bash
cd backend
source venv/bin/activate

# Tenant-scoping at the storage layer (free, ~1s, hits real Supabase):
python scripts/smoke_tenant_scoping.py

# Full lifecycle through /api/v1/* — auth + CRUD + scoping (free):
python scripts/smoke_api_v1_lifecycle.py

# Same, but also exercises AI turns + scoring + PDF (~$0.10, ~30s):
python scripts/smoke_api_v1_lifecycle.py --with-ai

# Per-turn cost measurement (~$0.10):
python scripts/measure_turn_cost.py

# Scoring model A/B comparison (~$0.05):
python scripts/compare_scoring_models.py
```

---

## Deployment

- **Frontend**: Vercel watches `main`. A push to `main` triggers a build (`npm run build` inside `frontend/`). The SPA rewrite in `vercel.json` routes all paths to `index.html`.
- **Backend**: Railway via `Procfile`. The host runs `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. The new `backend/nixpacks.toml` adds Pango/Cairo for WeasyPrint. Redeploy on push to whatever branch is tracked.

**Do not push untested changes to `main`.** Feature branch + PR + preview deploy first.

---

## Safe Development Workflow

1. Work on a **feature branch**, not directly on `main`.
2. Run and test locally (both frontend and backend).
3. Open a PR; review the Vercel preview deploy before merging.
4. After merging, monitor Vercel + Railway logs for errors.
5. If a deployment breaks the live site, revert the merge commit immediately — do not fix forward under pressure.

---

## Environment Variables

Never commit secrets (`backend/.env` is gitignored).

| Variable | Required? | Where used |
|---|---|---|
| `OPENAI_API_KEY` | yes | GPT-4o, GPT-4o-mini, Whisper, moderation |
| `ANTHROPIC_API_KEY` | yes (for WSDC/AP intermediate+, and scoring) | Sonnet, Haiku |
| `SUPABASE_URL` | yes (for `/api/v1/*` + retention) | Tenant-scoped storage |
| `SUPABASE_SERVICE_KEY` | yes (for `/api/v1/*` + retention) | Bypasses RLS |
| `SCORING_MODEL` | no | Override scoring model (default: Anthropic Haiku) |
| `MODERATION_MODEL` | no | Override moderation model (default `omni-moderation-latest`) |
| `MODERATION_THRESHOLD_<CATEGORY>` | no | Per-category moderation threshold override |
| `SAFETY_EXTRA_BLOCKLIST` | no | Comma-separated extra blocklist words |
| `API_RATE_LIMIT_PER_MIN` | no | Per-key rate limit (default 15) |
| `API_RATE_WINDOW_SECONDS` | no | Rate-limit window (default 60) |
| `MAX_DEBATES` | no | In-memory debate cap for website (default 1000) |
| `CORS_ORIGINS` | no | Comma-separated additional CORS origins |
| `SPEECH_CORPUS_DIR` | no | Override RAG corpus path |

GitHub Actions also needs `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` as repo secrets for the retention workflow.

---

## Key Files (quick reference)

### Frontend
| File | Purpose |
|---|---|
| `frontend/src/App.jsx` | Main app component (debate flow, scoring UI) |
| `frontend/src/LandingPage.jsx` | Landing page |
| `frontend/src/index.css` | All styles (single big stylesheet) |
| `frontend/index.html` | HTML entry; loads Google Fonts (Space Grotesk) |
| `frontend/vite.config.js` | Vite config + dev proxy |
| `frontend/vercel.json` | SPA routing rule |

### Backend — website (`/v1/*`)
| File | Purpose |
|---|---|
| `backend/app/main.py` | FastAPI entrypoint, website routes, AI generation, scoring |
| `backend/app/response.py` | OpenAI-side RAG + generation helpers |
| `backend/app/wsdc.py` | Anthropic two-pass WSDC/AP pipeline |
| `backend/app/difficulty.py` | Per-tier config (single source of truth) |
| `backend/app/transcriber_processor.py` | Whisper audio transcription |

### Backend — third-party API (`/api/v1/*`)
See the table in the [Active Initiative](#active-initiative-public-third-party-api) section above.
