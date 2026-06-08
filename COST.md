# DebateLab AI — Costs & Partner-Briefing Reference

For: Rico. Internal working doc. Sections 1–3 you can adapt and send to SuperJuniors; Section 4 is my read on why their questions are reasonable.

Measured 2026-06-07 on current production code: Sonnet 4.6 Pass 1 + Haiku 4.5 Pass 2 + **Haiku 4.5 scoring** (switched from GPT-4o after A/B test — see `backend/scripts/compare_scoring_models.py`).

---

## 1. Costs

### Per-call (measured, not estimated)

| Call | Model | Cost |
|---|---|---:|
| AI debate speech — Pass 1 (structure JSON) | Claude Sonnet 4.6 | ~$0.026 |
| AI debate speech — Pass 2 (render to prose) | Claude Haiku 4.5 | ~$0.009 |
| **AI speech total** | | **~$0.035** |
| Scoring (judge transcript + feedback) | **Claude Haiku 4.5** | **~$0.006** |

Scoring is **down from $0.014 with GPT-4o** — Haiku produced more specific, debate-coaching-quality feedback at less than half the price. GPT-4o remains a configured fallback if Anthropic is unavailable.

### Per debate

A typical SuperJuniors debate is **3 AI turns + 1 scoring call ≈ $0.11**.

A casual-mode debate (single-pass GPT-4o-mini, no two-pass pipeline) is **~$0.005**. If most students are doing casual practice, your bill is 20× smaller than the WSDC numbers below.

### Monthly projections

| Debates/mo | WSDC intermediate | Casual mode |
|---:|---:|---:|
| 100 | ~$11 | <$1 |
| 500 | ~$55 | ~$3 |
| 1,000 | ~$110 | ~$5 |
| 5,000 | ~$550 | ~$25 |
| 10,000 | ~$1,100 | ~$50 |

### Cost-cutting levers if needed

These are dials I can turn on the backend with no partner-side change.

| Lever | Saves | Quality risk |
|---|---|---|
| **A. Default students to casual mode** | 20× | Loses the formal speech structure. Probably fine for learning-mode users. |
| **B. Switch advanced WSDC Pass 2 to Haiku too** | ~50% on advanced | Less polished tournament voice |
| **C. Switch scoring to GPT-4o-mini** | ~5× on scoring | Generic feedback ("add more examples") |
| **D. Cap rounds to 2 in WSDC** | ~33% | Shorter debate experience |
| **E. Per-student daily caps** | Caps worst case | None — just rate limits |

**Vietnam-friendly preset (A + C + E):** 5,000 debates/month stays under $50.

---

## 2. Technical architecture — for the stack/AI/security question

This is what to tell SuperJuniors when they ask "how does it work." Honest and adequate; doesn't give away your prompts or RAG corpus.

### Stack

- **Backend:** FastAPI (Python), deployed on Railway
- **Database:** Supabase (Postgres), hosted in the US
- **Frontend (their integration is server-side, so this is just our reference UI):** React + Vite on Vercel

### AI models in active use

Routed automatically based on debate format and difficulty tier:

| Path | Pipeline | Why |
|---|---|---|
| WSDC / AP, intermediate–advanced | Sonnet 4.6 (Pass 1: argument structure JSON) → Haiku 4.5 (Pass 2: render to speech) | Two-pass separates *reasoning* from *delivery*. Sonnet does the thinking; Haiku is enough for the rendering. |
| WSDC, beginner | GPT-4o-mini with prompt-tier addendum | Shorter, gentler, no jargon |
| Casual mode | GPT-4o-mini single-pass | Conversational, cheap |
| Scoring (every debate) | Haiku 4.5 (GPT-4o fallback) | A/B tested 2026-06-07 — best feedback specificity per dollar |
| Audio transcription | OpenAI Whisper-1 | If students speak instead of type |

### Argument generation in plain English

Competitive-format debates use a **two-pass pipeline**:

1. The model outputs a structured outline as strict JSON — opening hook, contentions with logical mechanisms, weighing, rebuttals, clashes, close. Schema-enforced.
2. A second model renders that outline into a spoken-style speech with proper signposting and natural delivery, targeting tournament-realistic length (1,200–1,300 words for WSDC).

Casual mode is single-pass and conversational.

Retrieval-augmented from a curated corpus of high-quality competitive debate material to ground reasoning. We never let the AI fabricate sources — outputs say "as we see with Amazon" or "consider COVID-19" rather than "research shows" or "studies indicate."

### Safety layer

Every student input passes through:

1. **OpenAI's moderation endpoint** — catches violence, sexual content, self-harm signals, hate speech
2. **Our own input-safety check** (`backend/app/safety.py`) — narrower domain rules
3. **Prompt hardening** — system instructions prevent the AI from being redirected by malicious student input ("ignore previous instructions" attacks)

For scoring, we apply a per-tier scoring tone: beginner gets encouraging feedback with a score floor; advanced gets tournament-grade critique.

### Auth & data flow

- **API key auth:** every request to `/api/v1/*` requires `X-API-Key`. We store only the SHA-256 hash of the key — the raw key is shown to you once at creation and never persisted.
- **Tenant isolation:** every debate, message, and score is tagged with your `tenant_id` at write. Every read filters by `tenant_id`. A wrong-tenant lookup returns 404, indistinguishable from a non-existent record.
- **Row-Level Security** is enabled at the Postgres layer as a backstop.
- **Per-key rate limiting:** 15 requests per minute by default. Adjustable per partner.
- **HTTPS** end-to-end.
- **Usage logged** per call (tenant, key, endpoint, status, latency) into an `api_usage` table — auditable and the basis for any future metered billing.

### Reliability posture

Small operation. No formal SLA. No 24/7 oncall. I monitor it and respond when something breaks. Recommend SuperJuniors handle transient errors gracefully (retry once on 5xx, surface a user-friendly message if a turn fails).

### What partners control via the API

- `motion`, `side`, `mode` (casual/WSDC/AP), `difficulty` (beginner/intermediate/advanced)
- `external_user_id` — your own student ID, echoed back on every response. Lets you map debates to your records without us learning who the student actually is.
- `metadata` — free-form key/value bag (max 10 fields). Rendered into the PDF report header. Use for "Debater Name", "Class", "Instructor", etc.

---

## 3. Payment model — for the costs/payment question

Currently **pro bono**. The flow is simple:

- **My Anthropic account** pays for all Sonnet + Haiku calls (debate speeches + scoring)
- **My OpenAI account** pays for all GPT-4o-mini, Whisper, moderation, and the GPT-4o scoring fallback
- **SuperJuniors pays $0.** No invoice, no per-call charge.

### If this stops being sustainable

The trigger is probably north of **3,000–5,000 debates/month sustained** — at which point we're talking several hundred USD/month coming out of my pocket. If we get there, three reasonable paths:

1. **Pass-through reimbursement.** I send a monthly statement of real usage (already logged per-call in our database). Simple invoice, no margin. Cheapest for SuperJuniors — you'd pay at vendor cost, not retail SaaS rates.
2. **Bring-your-own-keys.** You sign up for OpenAI + Anthropic directly, give me your keys, and the bills go to you at vendor pricing. Best for data sovereignty / control. Modest setup on your side.
3. **Metered billing through DebateLab.** Stripe per-debate pricing. The "real SaaS" path. Not building this for v1.

**No decision needed now.** Infrastructure for all three is already there — every call writes a usage row tied to your tenant. Decide when volume tells us it matters.

### Recommended ask back to SuperJuniors

Before any of this matters, get two numbers from them:

1. **Realistic monthly debate volume target** ("20 students × 2 debates/week" is enough)
2. **Mode mix** — mostly casual practice, or mostly WSDC tournament training?

If the answer is "mostly casual, maybe 500 debates/month," it stays free forever and we never need to revisit this.

---

## 4. Why their concerns are valid

You asked. Here's my honest read.

### The technical question is exactly what you want a real partner to ask

A partner that doesn't ask "what's your stack, what AI, how's security" is a partner that hasn't thought about what they're integrating. SuperJuniors is putting an external AI in front of paying students — many of them minors. Three legitimate concerns sit behind that question:

1. **They're betting their academy's experience on your uptime.** If your backend goes down at 8pm on a debate-class evening, their students see broken UI and their instructors look incompetent. Knowing what you're built on lets them estimate that risk.
2. **They have compliance obligations under Vietnam's PDPD.** They handle minors' data. If they pass student personal info through to a US-hosted backend without understanding the data flow, that's an exposure for them, not you. The question is them being responsible, not nosy.
3. **They need to defend the choice internally.** When an academic director or a parent asks "what AI is grading my child?", "some startup" is not an answer. "We use Claude and GPT-4o, with content moderation, and our partner stores transcripts in encrypted Postgres in the US" is one they can stand behind.

If they hadn't asked, that would be the warning sign.

### The cost question reflects Vietnam's economic reality, and is doubly reasonable

Vietnam's GDP per capita is roughly 1/15th of the US. A SaaS price that's "totally fine, just $50/mo" in San Francisco is genuinely a hard ask in Hanoi — not because they're being cheap, but because their unit economics on tuition are very different. A debate academy in Vietnam charging $20–$50/month in tuition cannot absorb a $30/month API bill per active student. The math just doesn't work.

Their question — *can we afford this, and what's the payment method* — is them doing exactly the financial planning a serious partner does before integrating. They're trying to avoid signing up for a dependency they can't sustain. **That's good for both sides:** if they integrate and then can't pay, your project gets ripped out 6 months later and you've burned the relationship. Better to size it correctly now.

The Vietnam-friendly preset I described in Section 1 (casual mode default, Haiku scoring, per-student caps) brings 5,000 debates/month under $50 in total spend. That's the kind of envelope a Vietnamese academy can sustain. Without those levers, the same usage at "vanilla SaaS" rates would be ~$600/month, which would not work for them long-term.

### One concern of theirs you should take more seriously than you might

Of everything they asked, the **PDPD / data residency** thread is the one to actually engage with carefully. If their compliance team comes back and says "student data cannot leave Vietnam" or "we need a DPA before any integration", you have a problem that isn't solved by clever cost tuning — it's a structural data-location issue. Surface this now, before code-level integration; don't discover it after they've built their side.

Suggested move: in your reply, explicitly ask whether they need a Data Processing Agreement and whether US-based storage is acceptable under their compliance review. If they say "no", we figure out the workaround (regional Supabase project? anonymized `external_user_id` only?) before either of you invests more time.

### The meta-signal here is positive

A partner asking sharp technical and financial questions is a partner who's planning to actually use the thing. SuperJuniors is doing the diligence you'd want any serious partner to do. The right response isn't to feel pressured — it's to send them clear, specific answers (most of which are already in Sections 2 and 3 above), and to ask the return questions that protect *your* time:

- Who on their side does the integration?
- Realistic volume?
- Mode mix?
- Do they need a DPA?
- Is US data storage OK under their PDPD review?

Get those answers; the rest is execution.
