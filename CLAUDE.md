# CLAUDE.md — AI Debate Tutor (DebateLab)

## LIVE SITE WARNING

**debatelab.ai is a live production site.** This repo is the source. Changes merged to `main` and pushed may trigger automatic deployments. Treat every change as potentially affecting real users.

Before making any changes:
- Understand what is currently deployed vs. what is local
- Test locally first; never use the production environment as a test bed
- Avoid force-pushes to `main`
- If a change could break the UI or the AI response pipeline, flag it before proceeding

---

## Project Structure

```
AI-Debate-Tutor/
├── frontend/          # React + Vite SPA, deployed to Vercel
│   ├── src/
│   ├── public/
│   ├── vercel.json    # SPA rewrite rule (all routes → index.html)
│   └── package.json
└── backend/           # FastAPI app, deployed to a separate host
    ├── app/
    │   ├── main.py
    │   ├── response.py
    │   ├── transcriber_processor.py
    │   └── corpus/
    └── Procfile       # uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, react-router-dom v7, @vercel/analytics |
| Backend | FastAPI, Uvicorn, Pydantic, python-dotenv |
| AI | OpenAI GPT-4o-mini (debate responses), Whisper-1 (audio transcription) |
| Deployment (frontend) | Vercel |
| Deployment (backend) | Process-based host via Procfile (e.g. Railway, Render, Heroku) |

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

Requires a `.env` file in `backend/` with at least `OPENAI_API_KEY`.

### Frontend

```bash
cd frontend
npm install
npm run dev      # starts Vite dev server (usually :5173)
```

The frontend dev server proxies API calls to `localhost:8000` (confirm in `vite.config.js`).

---

## Deployment

- **Frontend**: Vercel watches `main`. A push to `main` triggers a Vercel build (`npm run build` inside `frontend/`). The SPA rewrite in `vercel.json` routes all paths to `index.html`.
- **Backend**: Deployed via `Procfile`. The host runs `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. Redeploy is triggered by a push to whatever branch the host is tracking (confirm in the host dashboard).

**Do not push untested changes to `main`** if the backend host auto-deploys from it. Use a feature branch and open a PR.

---

## Safe Development Workflow

1. Work on a **feature branch**, not directly on `main`.
2. Run and test locally (both frontend and backend).
3. Open a PR for review before merging.
4. After merging, monitor Vercel deployment logs and backend host logs for errors.
5. If a deployment breaks the live site, revert the merge commit immediately — do not attempt to fix forward under pressure.

---

## Environment Variables

Never commit secrets. Required variables:

| Variable | Where used |
|---|---|
| `OPENAI_API_KEY` | Backend — GPT-4o-mini and Whisper |

Set these in the host's environment dashboard, not in committed files.

---

## Key Files

| File | Purpose |
|---|---|
| `backend/app/main.py` | FastAPI entrypoint, route definitions |
| `backend/app/response.py` | AI debate response generation |
| `backend/app/transcriber_processor.py` | Whisper audio transcription logic |
| `frontend/src/` | React components and app logic |
| `frontend/vite.config.js` | Vite config (check proxy settings here) |
| `frontend/vercel.json` | Vercel SPA routing rule |
| `backend/Procfile` | Backend start command for host |
