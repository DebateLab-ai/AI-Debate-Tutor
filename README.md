# DebateLab

A web application for practicing debate with an AI opponent. Create structured debates, submit turns via text or audio, get streamed AI responses, scores, and (for partners) PDF reports.

Live site: [debatelab.ai](https://debatelab.ai) · Partner API docs: [`docs/`](./docs/)

## Tech Stack

### Frontend
- **React 18** + **Vite** + **react-router-dom**
- Deployed on **Vercel**

### Backend
- **FastAPI** + **Uvicorn** + **Pydantic v1**
- **OpenAI** — GPT-4o / GPT-4o-mini (casual & fallback), Whisper (transcription), moderation
- **Anthropic** — Claude Sonnet / Haiku (WSDC/AP two-pass speeches + primary scoring)
- **Supabase (Postgres)** — tenant-scoped storage for the partner API
- **WeasyPrint** + **Jinja2** — partner PDF reports
- Deployed on **Railway** (`api.debatelab.ai`)

### Storage (two paths)
| Path | Storage |
|---|---|
| Website (`/v1/*`) | In-memory (ephemeral) |
| Partner API (`/api/v1/*`) | Supabase, scoped by `tenant_id` |

## Prerequisites

- Python 3.9+
- Node.js 18+ and npm
- `OPENAI_API_KEY` (required for most features)
- `ANTHROPIC_API_KEY` (WSDC/AP intermediate+ and scoring)
- `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` (partner API + retention + CI smokes)

## Local setup

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `backend/.env` (never commit secrets):

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
SUPABASE_URL=...                 # needed for /api/v1/*
SUPABASE_SERVICE_KEY=...
```

```bash
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite proxies API calls to `localhost:8000` (see `frontend/vite.config.js`).

## API

### Website (`/v1/*`) — used by debatelab.ai
No API key. In-memory debates.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/health` | Health check |
| `POST` | `/v1/debates` | Create debate |
| `GET` | `/v1/debates/{id}` | Get debate + messages |
| `POST` | `/v1/debates/{id}/turns` | Submit user turn |
| `POST` | `/v1/debates/{id}/auto-turn` | AI turn (supports SSE stream) |
| `POST` | `/v1/debates/{id}/finish` | Mark completed |
| `POST` | `/v1/debates/{id}/score` | Score debate |
| `POST` | `/v1/transcribe` | Whisper audio → text |

Interactive docs when the server is running: `http://localhost:8000/docs`

### Partner API (`/api/v1/*`) — third-party integrations
Requires `X-API-Key`. Tenant-scoped Supabase storage. Keys are issued manually.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/debates` | Create debate |
| `POST` | `/api/v1/debates/{id}/turns` | Student turn → AI reply |
| `POST` | `/api/v1/debates/{id}/finish` | Score + complete |
| `GET` | `/api/v1/debates/{id}` | Full debate + score |
| `GET` | `/api/v1/debates` | List debates |
| `GET` | `/api/v1/debates/{id}/report.pdf` | PDF report |

Full partner docs: [`docs/README.md`](./docs/README.md)

### Local smoke tests (partner API)

```bash
cd backend && source venv/bin/activate
python scripts/smoke_tenant_scoping.py
python scripts/smoke_api_v1_lifecycle.py          # no LLM cost
python scripts/smoke_api_v1_lifecycle.py --with-ai  # optional, ~$0.10
```

## CI/CD

**CI** — GitHub Actions (`.github/workflows/ci.yml`) on every PR and push to `main`:

| Job | What it checks |
|---|---|
| `frontend` | `npm ci` + production build |
| `backend` | install deps + Python syntax (`compileall`) |
| `api-smokes` | tenant scoping + `/api/v1` lifecycle (no LLM; uses Supabase secrets) |

**CD** — on merge to `main`:
- **Vercel** deploys the frontend
- **Railway** deploys the backend (`api.debatelab.ai`)

Branch protection on `main` requires the CI jobs to pass before merge, so production only ships after green checks.

Separate scheduled workflow: `retention-cleanup` (7-day partner debate cleanup).

## Deployment notes

- Frontend root on Vercel: `frontend/`
- Backend on Railway via `Procfile` + `backend/nixpacks.toml` (WeasyPrint system deps)
- Set Railway env vars for OpenAI, Anthropic, Supabase, and `CORS_ORIGINS` for your Vercel domain
- Set Vercel `VITE_API_BASE_URL` to the Railway backend URL
- GitHub Actions secrets for smokes/retention: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`

Do not push untested changes straight to `main` — open a PR and wait for CI.
