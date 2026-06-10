"""Public third-party API under /api/v1/*.

This is the partner-facing surface. Every endpoint requires X-API-Key and
scopes all reads/writes by auth.tenant_id. Storage is Supabase via
debates_store — completely separate from the frontend's in-memory DEBATES dict.

AI generation and scoring functions are reused from app.main via lazy imports
inside endpoint bodies (avoids a circular import: main mounts this router).
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.auth import AuthContext, verify_api_key
from app.usage import log_usage
from app import debates_store
from app.safety import assert_input_safe, screen_output

router = APIRouter(prefix="/api/v1", tags=["public-api"])


# ---------- Public-facing Pydantic schemas ----------

Speaker = Literal["user", "assistant"]
Status = Literal["active", "completed"]
Mode = Literal["casual", "wsdc", "ap"]
Difficulty = Literal["beginner", "intermediate", "advanced"]


class DebateCreateIn(BaseModel):
    motion: str = Field(min_length=1, max_length=500)
    starter: Speaker = Field(description="Who speaks first: 'user' (student) or 'assistant' (AI)")
    num_rounds: int = Field(ge=1, le=10)
    mode: Mode = "casual"
    difficulty: Difficulty = "intermediate"
    external_user_id: Optional[str] = Field(default=None, max_length=128, description="Your own student/user ID; echoed back on every response")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Arbitrary key/value fields rendered on the PDF report header. Max 10 fields, scalar values only.")


class DebateOut(BaseModel):
    id: UUID
    motion: Optional[str]
    starter: Speaker
    next_speaker: Speaker
    current_round: int
    num_rounds: int
    status: Status
    mode: Mode
    difficulty: Difficulty
    external_user_id: Optional[str]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    id: UUID
    round_no: int
    speaker: Speaker
    content: str
    created_at: datetime


class TurnIn(BaseModel):
    content: str = Field(min_length=1, max_length=10000)


class TurnOut(BaseModel):
    debate_id: UUID
    user_message: MessageOut
    assistant_message: MessageOut
    current_round: int
    next_speaker: Optional[Speaker]
    status: Status


class ScoreMetricsOut(BaseModel):
    content_structure: float
    engagement: float
    strategy: float


class ScoreOut(BaseModel):
    debate_id: UUID
    overall: float
    metrics: ScoreMetricsOut
    feedback: str
    content_structure_feedback: str
    engagement_feedback: str
    strategy_feedback: str
    weakness_type: Optional[str]


class DebateWithMessagesOut(DebateOut):
    messages: list[MessageOut]
    score: Optional[ScoreOut]


# ---------- Adapters: DB dicts → internal Pydantic objects ----------
# The AI generation and scoring functions in app.main were written against the
# in-memory Debate/Message Pydantic classes (where motion is stored in `title`).
# We translate at the boundary so the AI code remains untouched.

def _db_to_internal_debate(d: dict[str, Any]):
    from app.main import Debate as InternalDebate  # lazy to avoid circular import
    return InternalDebate(
        id=UUID(d["id"]),
        title=d.get("motion"),
        num_rounds=d["num_rounds"],
        starter=d["starter"],
        current_round=d["current_round"],
        next_speaker=d["next_speaker"],
        status=d["status"],
        created_at=_parse_dt(d["created_at"]),
        updated_at=_parse_dt(d["updated_at"]),
        mode=d["mode"],
        difficulty=d["difficulty"],
    )


def _db_to_internal_messages(rows: list[dict[str, Any]]):
    from app.main import Message as InternalMessage
    return [
        InternalMessage(
            id=UUID(r["id"]),
            debate_id=UUID(r["debate_id"]),
            round_no=r["round_no"],
            speaker=r["speaker"],
            content=r["content"],
            created_at=_parse_dt(r["created_at"]),
        )
        for r in rows
    ]


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    # Supabase returns ISO 8601 with trailing 'Z' or offset; fromisoformat handles offsets.
    s = str(value).replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def _debate_out(d: dict[str, Any]) -> DebateOut:
    return DebateOut(
        id=UUID(d["id"]),
        motion=d.get("motion"),
        starter=d["starter"],
        next_speaker=d["next_speaker"],
        current_round=d["current_round"],
        num_rounds=d["num_rounds"],
        status=d["status"],
        mode=d["mode"],
        difficulty=d["difficulty"],
        external_user_id=d.get("external_user_id"),
        metadata=d.get("metadata") or {},
        created_at=_parse_dt(d["created_at"]),
        updated_at=_parse_dt(d["updated_at"]),
    )


def _message_out(m: dict[str, Any]) -> MessageOut:
    return MessageOut(
        id=UUID(m["id"]),
        round_no=m["round_no"],
        speaker=m["speaker"],
        content=m["content"],
        created_at=_parse_dt(m["created_at"]),
    )


def _score_out(s: dict[str, Any]) -> ScoreOut:
    return ScoreOut(
        debate_id=UUID(s["debate_id"]),
        overall=float(s["overall"]),
        metrics=ScoreMetricsOut(
            content_structure=float(s["content_structure"]),
            engagement=float(s["engagement"]),
            strategy=float(s["strategy"]),
        ),
        feedback=s.get("feedback") or "",
        content_structure_feedback=s.get("content_structure_feedback") or "",
        engagement_feedback=s.get("engagement_feedback") or "",
        strategy_feedback=s.get("strategy_feedback") or "",
        weakness_type=s.get("weakness_type"),
    )


# ---------- Helpers ----------

def _second_speaker(starter: Speaker) -> Speaker:
    return "assistant" if starter == "user" else "user"


def _advance_state(
    *, starter: Speaker, current_round: int, num_rounds: int, just_spoke: Speaker
) -> tuple[int, Speaker, Status]:
    """Mirror the state machine in main._append_message_and_advance, but pure."""
    if just_spoke == _second_speaker(starter):
        if current_round >= num_rounds:
            return current_round, starter, "completed"
        return current_round + 1, starter, "active"
    return current_round, _second_speaker(just_spoke), "active"


def _log(bg: BackgroundTasks, auth: AuthContext, endpoint: str, status_code: int, latency_ms: int) -> None:
    bg.add_task(
        log_usage,
        tenant_id=auth.tenant_id,
        api_key_id=auth.api_key_id,
        endpoint=endpoint,
        response_status=status_code,
        latency_ms=latency_ms,
    )


# ---------- Endpoints ----------

@router.post("/debates", response_model=DebateOut, status_code=201)
def create_debate(
    body: DebateCreateIn,
    background_tasks: BackgroundTasks,
    auth: AuthContext = Depends(verify_api_key),
):
    start = time.monotonic()
    if body.mode in ("wsdc", "ap") and body.num_rounds > 3:
        _log(background_tasks, auth, "POST /api/v1/debates", 400, int((time.monotonic() - start) * 1000))
        raise HTTPException(400, "Parliamentary modes support up to 3 rounds")

    # Safety screen the motion — it is fed into every prompt and rendered on the
    # PDF report, same exposure as a debate turn.
    assert_input_safe(body.motion, where="debate motion")

    try:
        debate = debates_store.create_debate(
            tenant_id=auth.tenant_id,
            motion=body.motion,
            starter=body.starter,
            num_rounds=body.num_rounds,
            mode=body.mode,
            difficulty=body.difficulty,
            external_user_id=body.external_user_id,
            metadata=body.metadata,
        )
    except debates_store.MetadataError as e:
        _log(background_tasks, auth, "POST /api/v1/debates", 400, int((time.monotonic() - start) * 1000))
        raise HTTPException(400, str(e))

    _log(background_tasks, auth, "POST /api/v1/debates", 201, int((time.monotonic() - start) * 1000))
    return _debate_out(debate)


@router.post("/debates/{debate_id}/turns", response_model=TurnOut)
def submit_turn(
    debate_id: UUID,
    body: TurnIn,
    background_tasks: BackgroundTasks,
    auth: AuthContext = Depends(verify_api_key),
):
    start = time.monotonic()
    endpoint = "POST /api/v1/debates/{id}/turns"

    debate = debates_store.get_debate(tenant_id=auth.tenant_id, debate_id=debate_id)
    if not debate:
        _log(background_tasks, auth, endpoint, 404, int((time.monotonic() - start) * 1000))
        raise HTTPException(404, "Debate not found")
    if debate["status"] != "active":
        _log(background_tasks, auth, endpoint, 400, int((time.monotonic() - start) * 1000))
        raise HTTPException(400, f"Debate is {debate['status']}")
    if debate["next_speaker"] != "user":
        _log(background_tasks, auth, endpoint, 409, int((time.monotonic() - start) * 1000))
        raise HTTPException(409, "It is the assistant's turn — the student already spoke this round")

    # Safety screen on the partner-supplied student speech.
    assert_input_safe(body.content, where="debate turn")

    # 1. Persist the human turn.
    user_msg = debates_store.append_message(
        tenant_id=auth.tenant_id,
        debate_id=debate["id"],
        round_no=debate["current_round"],
        speaker="user",
        content=body.content.strip(),
    )

    # 2. Advance state after the human turn.
    new_round, new_next, new_status = _advance_state(
        starter=debate["starter"],
        current_round=debate["current_round"],
        num_rounds=debate["num_rounds"],
        just_spoke="user",
    )

    # If the human was the second speaker and we're done, don't generate an AI turn.
    if new_status == "completed" or new_next == "user":
        debates_store.update_debate_state(
            tenant_id=auth.tenant_id,
            debate_id=debate["id"],
            current_round=new_round,
            next_speaker=new_next,
            status=new_status,
        )
        _log(background_tasks, auth, endpoint, 200, int((time.monotonic() - start) * 1000))
        # Partner asked for a turn; we accepted it, but there's no AI reply.
        # Surface this by returning the user message twice — caller can compare IDs.
        return TurnOut(
            debate_id=UUID(debate["id"]),
            user_message=_message_out(user_msg),
            assistant_message=_message_out(user_msg),
            current_round=new_round,
            next_speaker=None if new_status == "completed" else new_next,
            status=new_status,
        )

    # 3. Generate AI reply against the freshly-extended transcript.
    debate_for_round = {**debate, "current_round": new_round, "next_speaker": new_next}
    all_msgs = debates_store.list_messages(tenant_id=auth.tenant_id, debate_id=debate["id"])

    from app.main import generate_ai_turn_text  # lazy
    internal_debate = _db_to_internal_debate(debate_for_round)
    internal_msgs = _db_to_internal_messages(all_msgs)
    try:
        # Same output safety gate as the website path: a flagged or unverifiable
        # generation is replaced with a safe fallback (fail-closed).
        ai_text = screen_output(generate_ai_turn_text(internal_debate, internal_msgs))
    except Exception as e:
        print(f"[api_v1] AI generation failed: {type(e).__name__}: {e}")
        _log(background_tasks, auth, endpoint, 502, int((time.monotonic() - start) * 1000))
        raise HTTPException(502, "AI generation temporarily unavailable. Please retry.")

    # 4. Persist AI message and advance state again.
    ai_msg = debates_store.append_message(
        tenant_id=auth.tenant_id,
        debate_id=debate["id"],
        round_no=new_round,
        speaker="assistant",
        content=ai_text,
    )
    final_round, final_next, final_status = _advance_state(
        starter=debate["starter"],
        current_round=new_round,
        num_rounds=debate["num_rounds"],
        just_spoke="assistant",
    )
    debates_store.update_debate_state(
        tenant_id=auth.tenant_id,
        debate_id=debate["id"],
        current_round=final_round,
        next_speaker=final_next,
        status=final_status,
    )

    _log(background_tasks, auth, endpoint, 200, int((time.monotonic() - start) * 1000))
    return TurnOut(
        debate_id=UUID(debate["id"]),
        user_message=_message_out(user_msg),
        assistant_message=_message_out(ai_msg),
        current_round=final_round,
        next_speaker=None if final_status == "completed" else final_next,
        status=final_status,
    )


@router.post("/debates/{debate_id}/finish", response_model=ScoreOut)
def finish_debate(
    debate_id: UUID,
    background_tasks: BackgroundTasks,
    auth: AuthContext = Depends(verify_api_key),
):
    start = time.monotonic()
    endpoint = "POST /api/v1/debates/{id}/finish"

    debate = debates_store.get_debate(tenant_id=auth.tenant_id, debate_id=debate_id)
    if not debate:
        _log(background_tasks, auth, endpoint, 404, int((time.monotonic() - start) * 1000))
        raise HTTPException(404, "Debate not found")

    # Idempotent (as documented in docs/reference.md): if the debate has already
    # been scored, return the stored score instead of re-running the scoring LLM.
    # Keeps retries free and deterministic.
    existing = debates_store.get_score(tenant_id=auth.tenant_id, debate_id=debate["id"])
    if existing:
        _log(background_tasks, auth, endpoint, 200, int((time.monotonic() - start) * 1000))
        return _score_out(existing)

    msgs = debates_store.list_messages(tenant_id=auth.tenant_id, debate_id=debate["id"])
    if not msgs:
        _log(background_tasks, auth, endpoint, 400, int((time.monotonic() - start) * 1000))
        raise HTTPException(400, "Cannot score a debate with no turns")

    from app.main import compute_debate_score  # lazy
    internal_debate = _db_to_internal_debate(debate)
    internal_msgs = _db_to_internal_messages(msgs)
    breakdown = compute_debate_score(internal_debate, internal_msgs)

    saved = debates_store.upsert_score(
        tenant_id=auth.tenant_id,
        debate_id=debate["id"],
        overall=breakdown.overall,
        content_structure=breakdown.metrics.content_structure,
        engagement=breakdown.metrics.engagement,
        strategy=breakdown.metrics.strategy,
        feedback=breakdown.feedback,
        content_structure_feedback=breakdown.content_structure_feedback,
        engagement_feedback=breakdown.engagement_feedback,
        strategy_feedback=breakdown.strategy_feedback,
        weakness_type=breakdown.weakness_type,
    )

    # If the debate wasn't already completed, mark it so. Scoring is a terminal action.
    if debate["status"] != "completed":
        debates_store.update_debate_state(
            tenant_id=auth.tenant_id,
            debate_id=debate["id"],
            current_round=debate["current_round"],
            next_speaker=debate["next_speaker"],
            status="completed",
        )

    _log(background_tasks, auth, endpoint, 200, int((time.monotonic() - start) * 1000))
    return _score_out(saved)


@router.get("/debates/{debate_id}", response_model=DebateWithMessagesOut)
def get_debate(
    debate_id: UUID,
    background_tasks: BackgroundTasks,
    auth: AuthContext = Depends(verify_api_key),
):
    start = time.monotonic()
    endpoint = "GET /api/v1/debates/{id}"

    debate = debates_store.get_debate(tenant_id=auth.tenant_id, debate_id=debate_id)
    if not debate:
        _log(background_tasks, auth, endpoint, 404, int((time.monotonic() - start) * 1000))
        raise HTTPException(404, "Debate not found")

    msgs = debates_store.list_messages(tenant_id=auth.tenant_id, debate_id=debate["id"])
    score = debates_store.get_score(tenant_id=auth.tenant_id, debate_id=debate["id"])

    out = _debate_out(debate)
    _log(background_tasks, auth, endpoint, 200, int((time.monotonic() - start) * 1000))
    return DebateWithMessagesOut(
        **out.dict(),
        messages=[_message_out(m) for m in msgs],
        score=_score_out(score) if score else None,
    )


@router.get("/debates/{debate_id}/report.pdf", responses={200: {"content": {"application/pdf": {}}}})
def get_debate_report(
    debate_id: UUID,
    background_tasks: BackgroundTasks,
    auth: AuthContext = Depends(verify_api_key),
):
    """Render the debate report as a PDF. Two sections: feedback + transcript.

    400 if the debate hasn't been scored yet (nothing to report on).
    503 if the PDF renderer can't load (missing system deps in production).
    """
    start = time.monotonic()
    endpoint = "GET /api/v1/debates/{id}/report.pdf"

    debate = debates_store.get_debate(tenant_id=auth.tenant_id, debate_id=debate_id)
    if not debate:
        _log(background_tasks, auth, endpoint, 404, int((time.monotonic() - start) * 1000))
        raise HTTPException(404, "Debate not found")

    score = debates_store.get_score(tenant_id=auth.tenant_id, debate_id=debate["id"])
    if not score:
        _log(background_tasks, auth, endpoint, 400, int((time.monotonic() - start) * 1000))
        raise HTTPException(400, "Debate must be scored before a report can be generated. Call /finish first.")

    messages = debates_store.list_messages(tenant_id=auth.tenant_id, debate_id=debate["id"])

    from app.pdf import render_pdf  # lazy: missing system deps shouldn't kill the whole API on boot
    try:
        pdf_bytes = render_pdf(debate=debate, messages=messages, score=score)
    except RuntimeError as e:
        print(f"[api_v1] PDF render failed: {e}")
        _log(background_tasks, auth, endpoint, 503, int((time.monotonic() - start) * 1000))
        raise HTTPException(503, "PDF generation is temporarily unavailable.")

    filename = f"debate_report_{str(debate['id'])[:8]}.pdf"
    _log(background_tasks, auth, endpoint, 200, int((time.monotonic() - start) * 1000))
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/debates", response_model=list[DebateOut])
def list_debates(
    background_tasks: BackgroundTasks,
    auth: AuthContext = Depends(verify_api_key),
    external_user_id: Optional[str] = Query(default=None),
    since: Optional[datetime] = Query(default=None, description="ISO-8601; returns debates created at-or-after this time"),
    limit: int = Query(default=50, ge=1, le=200),
):
    start = time.monotonic()
    endpoint = "GET /api/v1/debates"

    rows = debates_store.list_debates(
        tenant_id=auth.tenant_id,
        external_user_id=external_user_id,
        since=since,
        limit=limit,
    )
    _log(background_tasks, auth, endpoint, 200, int((time.monotonic() - start) * 1000))
    return [_debate_out(r) for r in rows]
