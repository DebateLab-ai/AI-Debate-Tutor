# app/main.py
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

import os
from datetime import datetime
from typing import Literal, Dict, List, Optional
import json
from uuid import uuid4, UUID
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import RAG functionality
from app.response import SimpleRAG, generate_debate_with_coach_loop, generate_rebuttal_speech

# ---------- Types ----------
load_dotenv()
Speaker = Literal["user", "assistant"]
Status = Literal["active", "completed"]

app = FastAPI(title="Debate MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- RAG System ----------
# Global RAG instance - will be initialized on startup
rag_system: Optional[SimpleRAG] = None

@app.on_event("startup")
def startup_event():
    """Initialize RAG system with corpus on app startup"""
    global rag_system
    corpus_dir = os.getenv("SPEECH_CORPUS_DIR", "./app/corpus")

    print(f"[RAG] Initializing RAG system with corpus from: {corpus_dir}")
    rag_system = SimpleRAG()

    if os.path.exists(corpus_dir):
        rag_system.add_corpus_folder(corpus_dir, pattern=r".*\.txt$")
        print(f"[RAG] Indexed {len(rag_system.docs)} documents")
    else:
        print(f"[RAG] Warning: Corpus directory not found: {corpus_dir}")
        print(f"[RAG] RAG system initialized but no documents loaded")


# ---------- In-memory "DB" ----------
class Debate(BaseModel):
    id: UUID
    title: Optional[str] = None
    num_rounds: int
    starter: Speaker
    current_round: int = 1
    next_speaker: Speaker
    status: Status = "active"
    created_at: datetime
    updated_at: datetime

class Message(BaseModel):
    id: UUID
    debate_id: UUID
    round_no: int
    speaker: Speaker
    content: str
    created_at: datetime

DEBATES: Dict[UUID, Debate] = {}
MESSAGES: Dict[UUID, List[Message]] = {}  # keyed by debate_id
SCORES: Dict[UUID, "ScoreBreakdown"] = {}

# ---------- Schemas (I/O) ----------
class DebateCreate(BaseModel):
    title: Optional[str] = None
    num_rounds: int = Field(ge=1, le=3)
    starter: Speaker

class DebateOut(BaseModel):
    id: UUID
    title: Optional[str]
    num_rounds: int
    starter: Speaker
    current_round: int
    next_speaker: Speaker
    status: Status
    created_at: datetime
    updated_at: datetime

class MessageOut(BaseModel):
    id: UUID
    round_no: int
    speaker: Speaker
    content: str
    created_at: datetime

class DebateWithMessages(DebateOut):
    messages: List[MessageOut]

class TurnIn(BaseModel):
    speaker: Speaker
    content: str = Field(min_length=1)

class TurnOut(BaseModel):
    round_no: int
    accepted: bool
    next_speaker: Optional[Speaker]
    current_round: int
    status: Status
    message_id: UUID

class TranscribeOut(BaseModel):
    text: str
    language: Optional[str] = None


class ScoreMetrics(BaseModel):
    content_structure: float
    engagement: float
    strategy: float


class ScoreBreakdown(BaseModel):
    overall: float
    metrics: ScoreMetrics
    feedback: str = "No overall feedback provided."
    content_structure_feedback: str = "No content/structure feedback provided."
    engagement_feedback: str = "No engagement feedback provided."
    strategy_feedback: str = "No strategy feedback provided."


class ScoreOut(ScoreBreakdown):
    debate_id: UUID

# ---------- RAG Schemas ----------
class RAGSpeechRequest(BaseModel):
    motion: str = Field(min_length=1)
    side: Literal["Government", "Opposition"] = "Government"
    format: str = "WSDC"
    use_rag: bool = True
    top_k: int = Field(default=6, ge=1, le=20)
    min_score: float = Field(default=0.1, ge=0.0, le=1.0)
    model: str = "gpt-4o-mini"
    temp_low: float = Field(default=0.3, ge=0.0, le=2.0)
    temp_high: float = Field(default=0.8, ge=0.0, le=2.0)

class RAGRebuttalRequest(BaseModel):
    motion: str = Field(min_length=1)
    opponent_speech: str = Field(min_length=1)
    side: Literal["Government", "Opposition"] = "Opposition"
    format: str = "WSDC"
    use_rag: bool = True
    top_k: int = Field(default=6, ge=1, le=20)
    min_score: float = Field(default=0.1, ge=0.0, le=1.0)
    model: str = "gpt-4o-mini"
    temp_low: float = Field(default=0.3, ge=0.0, le=2.0)
    temp_high: float = Field(default=0.8, ge=0.0, le=2.0)

class RAGSpeechResponse(BaseModel):
    speech: str
    context_count: int
    avg_score: Optional[float] = None

class CorpusStatsResponse(BaseModel):
    total_documents: int
    corpus_available: bool

# ---------- Helpers ----------
def second_speaker_for_round(starter: Speaker) -> Speaker:
    return "assistant" if starter == "user" else "user"

def _append_message_and_advance(debate: Debate, speaker: Speaker, content: str) -> Message:
    """Save message, advance debate state; returns the saved message."""
    mid = uuid4()
    msg = Message(
        id=mid,
        debate_id=debate.id,
        round_no=debate.current_round,
        speaker=speaker,
        content=content.strip(),
        created_at=datetime.utcnow(),
    )
    MESSAGES[debate.id].append(msg)

    # Advance state
    if speaker == second_speaker_for_round(debate.starter):
        if debate.current_round >= debate.num_rounds:
            debate.status = "completed"
        else:
            debate.current_round += 1
            debate.next_speaker = debate.starter
    else:
        debate.next_speaker = "assistant" if speaker == "user" else "user"

    debate.updated_at = datetime.utcnow()
    return msg

# --- Optional OpenAI client (used for /auto-turn now; Whisper soon) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI  # OpenAI Python SDK ≥ 1.0
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None

def generate_ai_turn_text(debate: Debate, messages: List[Message]) -> str:
    """
    Produce assistant text using RAG-powered generation when available.
    Falls back to basic GPT-4o-mini if RAG is not initialized.
    """
    motion = debate.title if debate.title else "General debate topic"

    # If we have RAG and this is a rebuttal (not the first turn)
    if rag_system and len(messages) > 0:
        # Determine side - assistant is Opposition if starter is user, Government if starter is assistant
        side = "Opposition" if debate.starter == "user" else "Government"

        # Get the last opponent message to rebut
        opponent_messages = [m for m in messages if m.speaker != "assistant"]
        if opponent_messages:
            last_opponent = opponent_messages[-1]
            try:
                result = generate_rebuttal_speech(
                    rag=rag_system,
                    motion=motion,
                    opponent_speech=last_opponent.content,
                    side=side,
                    format="WSDC",
                    use_rag=True,
                    top_k=6,
                    min_score=0.1,
                    model="gpt-4o-mini",
                    temp_low=0.3,
                    temp_high=0.8
                )
                return result["rebuttal_speech"]
            except Exception as e:
                print(f"[RAG] Rebuttal generation failed: {e}, falling back to basic generation")

    # If this is the first turn and we have RAG
    if rag_system and len(messages) == 0:
        side = "Government" if debate.starter == "assistant" else "Opposition"
        try:
            result = generate_debate_with_coach_loop(
                rag=rag_system,
                motion=motion,
                side=side,
                format="WSDC",
                use_rag=True,
                top_k=6,
                min_score=0.1,
                model="gpt-4o-mini",
                temperature_gen=0.5,
                temperature_rev=0.2,
                temp_low=0.3,
                temp_high=0.8
            )
            return result["initial_speech"]
        except Exception as e:
            print(f"[RAG] Speech generation failed: {e}, falling back to basic generation")

    # Fallback: use similar style to fine-tuned prompts in response.py
    topic_context = f"Debate topic: {motion}"
    sys = """You are a world-class competitive debater. Your goal is to WIN through sharp, incisive argumentation—not through aggression. Be strategic, precise, and respectful. Fill gaps with compelling real-world examples that any educated voter would recognize—think NYT front page, not academic journals. Use concrete mechanisms and numbers. Make clear, fair comparisons that demonstrate why your case is stronger. Sound like a human champion debater who wins through superior logic and analysis, not an essay or a robot.
If the
When delivering your speech:
- SIGNPOST HEAVILY: Label everything clearly
- Provide 2-3 well-developed contentions/arguments
- Each argument needs: PREMISE → 2-3 WELL-DEVELOPED MECHANISMS → IMPACT
- Develop EACH mechanism with 2-3 sentences (explain the causal chain, then elaborate)
- Weigh arguments explicitly on magnitude, probability, timeframe
- Zero filler, no pleasantries
- Sound human, not robotic - vary sentence length
- Use concrete examples voters recognize (Amazon, climate disasters, iPhone)
- NEVER cite sources - use examples as common knowledge
- Make it sound like you're SPEAKING, not reading an essay
- There is no point in saying stuff that both sides would agree to : for example when debating about aid, both teams can agree that "aid is a powerful tool that can either stabilize or destabilize regions". THis is useless.
"""
    convo = []
    for m in messages:
        role = "user" if m.speaker == "user" else "assistant"
        tag = f"[Round {m.round_no} · {m.speaker.upper()}]"
        convo.append({"role": role, "content": f"{tag} {m.content}"})

    # Determine if this is first speech or rebuttal
    opponent_messages = [m for m in messages if m.speaker != debate.next_speaker]
    if opponent_messages:
        # Rebuttal situation
        prompt_now = f"""Motion: {topic_context}
Current round: {debate.current_round} of {debate.num_rounds}

Deliver a rebuttal speech that:
1. Tears down the opponent's key arguments (address their strongest points first).
- When rebutting, think about NEGATING first. If they claim something, explain a proper reason why that something is NOT true
- If that's hard, mitigate it. If they claim sometthing, explain why it's not as big of an issue as they claimed. 
- If that's also too hard, last resort: concede to it, but explain why your argument is still more important. 
2. Presents 1 new constructive arguments, explicitly label this as an "extension" or "spike"
3. Does comparative weighing

Be sharp, precise, and demonstrate why your case is stronger."""
    else:
        # Opening speech
        prompt_now = f"""Motion: {topic_context}
Current round: {debate.current_round} of {debate.num_rounds}

Deliver an opening speech following the structure:
1. Opening hook (2-3 sentences with real-world example)
2. Framing & burdens
3. 2-3 contentions (each with premise, mechanisms, weighing)
4. Conclusion

Make your case compelling and well-structured."""

    if client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": sys}] + convo + [
                    {"role": "user", "content": prompt_now}
                ],
                temperature=0.7,  # Higher temp since no RAG context (matches response.py adaptive logic)
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass  # fall back to stub

    # Stub output for local dev without API key
    return f"(AI {debate.next_speaker} R{debate.current_round}) Brief, signposted response."

def transcribe_with_whisper(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    if not client:
        raise RuntimeError("OpenAI client not configured")
    from io import BytesIO
    bio = BytesIO(audio_bytes)
    bio.name = filename
    resp = client.audio.transcriptions.create(model="whisper-1", file=bio)
    return resp.text


def compute_debate_score(debate: Debate, messages: List[Message]) -> ScoreBreakdown:
    if not client:
        raise HTTPException(503, "Scoring requires OpenAI API key")

    if not messages:
        return ScoreBreakdown(
            overall=0.0,
            metrics=ScoreMetrics(content_structure=0.0, engagement=0.0, strategy=0.0),
            feedback="No debate content available to score.",
            content_structure_feedback="No content/structure feedback available.",
            engagement_feedback="No engagement feedback available.",
            strategy_feedback="No strategy feedback available."
        )

    convo: List[Dict[str, str]] = []
    for msg in messages:
        role = "user" if msg.speaker == "user" else "assistant"
        speaker = msg.speaker.upper()
        convo.append({
            "role": role,
            "content": f"[Round {msg.round_no} · {speaker}] {msg.content}"
        })

    system_prompt = (
        "You are DebateJudgeGPT, an expert debate adjudicator across APDA, Public Forum, and WSDC formats.\n"
        "You will be given the full transcript of a debate between a human debater (`user`) and an AI sparring partner (`assistant`).\n"
        "Your task is to evaluate ONLY the human debater's performance. Consider how much they engaged with the content of the opposition, if applicable. If the user is proposition\n"
        "and there is only one proposition and one opposition block, then DO NOT consider engagement"
        "Score the human on these metrics (0-10 each, integers only):\n"
        "1. Content & Structure – arguments are understandable and well-explained; logical links are explicit; easy to follow; jargon is handled; clear signposting/roadmap.\n"
        "2. Engagement – direct refutation if applicable (ONLY if the user speaks after at least one AI response); comparison; impact weighing; turns/defense.\n"
        "3. Strategy – prioritizes win conditions; allocates time well across offense/defense; collapses to strongest arguments; avoids overinvesting in weak lines.\n"

        "(MUST FOLLOW) for strategy: As long as the user makes arguments that supports their side, that is justification for a score of >= 5\n"

        "Engagement applicability rule (MUST FOLLOW):\n"
        "- Engagement is scorable ONLY if the user has a speech AFTER at least one assistant speech.\n"
        "- If not scorable, set engagement_score=0 and engagement_feedback must be exactly: "
        "'Not scorable: the user had no opportunity to respond to the opposition.'\n"
        "- Do NOT mention lack of engagement as a weakness anywhere else when it is not scorable.\n"

        "Anti-vagueness requirement (MUST FOLLOW):\n"
        "- Every positive claim must include: (a) a short quote from the user (<=12 words) OR a very specific described behavior, and (b) why that helps win rounds.\n"
        "- Every criticism must include: (a) what was missing, (b) a CONCRETE EXAMPLE OF what it should have looked like, and (c) one concrete next-step drill.\n"
        "- Do NOT use generic phrases like 'well articulated', 'clear point', 'add evidence' unless immediately followed by a specific example.\n"

        "Provide:\n"
        "- `overall_score`: holistic score (0-10) for the human debater. Weighted average: 40% content, 30% strategy, 30% Engagement\n"
        "`feedback` must be 4–6 sentences total and follow this structure:\n"
        "Sentence 1: Biggest strength + quote/behavior + why it matters strategically.\n"
        "Sentence 2: Second strength + quote/behavior + why it matters.\n"
        "Sentence 3: Biggest weakness + what was missing + what it should have looked like. *MUST FOLLOW; if score was 0 then DONT say a weakness is 'not responsive to assistant'\n"
        "Sentence 4: Concrete next step drill (one drill) + what to measure next time.\n"
        "(Optional sentence 5-6): Strategy collapse advice tied to this specific speech.\n"

        "Return ONLY a JSON object with keys: overall_score, feedback, content_structure_score, content_structure_feedback, engagement_score, engagement_feedback, strategy_score, strategy_feedback.\n"
        "Evidence requirement: each *_feedback must reference at least one specific behavior from the transcript; include 1–2 short quotes (<=12 words) from the user when possible.\n"
        "Do not include any additional text outside the JSON."
    )

    prompt = [
        {"role": "system", "content": system_prompt},
        *convo,
        {
            "role": "user",
            "content": (
                "Judge this debate according to the instructions. "
                "Be fair, constructive, and reference specific debate behaviors in your feedback."
            ),
        },
    ]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
            temperature=0.4,
        )
    except Exception as exc:
        raise HTTPException(502, f"Scoring failed: {exc}") from exc

    raw = resp.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(502, f"Scoring response malformed: {raw}")

    def _get_float(key: str) -> float:
        value = parsed.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        return 0.0

    metrics = ScoreMetrics(
        content_structure=round(_get_float("content_structure_score"), 1),
        engagement=round(_get_float("engagement_score"), 1),
        strategy=round(_get_float("strategy_score"), 1),
    )

    overall = round(_get_float("overall_score"), 1)

    return ScoreBreakdown(
        overall=overall,
        metrics=metrics,
        feedback=parsed.get("feedback", "No overall feedback provided."),
        content_structure_feedback=parsed.get("content_structure_feedback", "No content/structure feedback provided."),
        engagement_feedback=parsed.get("engagement_feedback", "No engagement feedback provided."),
        strategy_feedback=parsed.get("strategy_feedback", "No strategy feedback provided."),
    )


def _score_out_from_breakdown(debate_id: UUID, breakdown: ScoreBreakdown) -> ScoreOut:
    # Ensure we have a ScoreBreakdown instance with all fields populated
    if not isinstance(breakdown, ScoreBreakdown):
        breakdown = ScoreBreakdown.model_validate(breakdown)

    metrics = breakdown.metrics or ScoreMetrics(
        content_structure=0.0, engagement=0.0, strategy=0.0
    )

    return ScoreOut(
        debate_id=debate_id,
        overall=breakdown.overall,
        metrics=metrics,
        feedback=breakdown.feedback,
        content_structure_feedback=breakdown.content_structure_feedback,
        engagement_feedback=breakdown.engagement_feedback,
        strategy_feedback=breakdown.strategy_feedback,
    )

# ---------- 1) Create debate ----------
@app.post("/v1/debates", response_model=DebateOut, status_code=201)
def create_debate(body: DebateCreate):
    did = uuid4()
    now = datetime.utcnow()
    debate = Debate(
        id=did,
        title=body.title,
        num_rounds=body.num_rounds,
        starter=body.starter,
        current_round=1,
        next_speaker=body.starter,
        status="active",
        created_at=now,
        updated_at=now,
    )
    DEBATES[did] = debate
    MESSAGES[did] = []
    return debate

# ---------- 2) Submit a turn ----------
@app.post("/v1/debates/{debate_id}/turns", response_model=TurnOut)
def submit_turn(debate_id: UUID, body: TurnIn):
    debate = DEBATES.get(debate_id)
    if not debate:
        raise HTTPException(404, "Debate not found")
    if debate.status != "active":
        raise HTTPException(400, f"Debate is {debate.status}")
    if body.speaker != debate.next_speaker:
        raise HTTPException(409, f"It is {debate.next_speaker}'s turn.")

    msg = _append_message_and_advance(debate, body.speaker, body.content)

    return TurnOut(
        round_no=msg.round_no,
        accepted=True,
        next_speaker=debate.next_speaker if debate.status == "active" else None,
        current_round=debate.current_round,
        status=debate.status,
        message_id=msg.id,
    )

# ---------- 3) GET state + messages ----------
@app.get("/v1/debates/{debate_id}", response_model=DebateWithMessages)
def get_debate(debate_id: UUID):
    debate = DEBATES.get(debate_id)
    if not debate:
        raise HTTPException(404, "Debate not found")
    msgs = [
        MessageOut(
            id=m.id,
            round_no=m.round_no,
            speaker=m.speaker,
            content=m.content,
            created_at=m.created_at,
        )
        for m in MESSAGES.get(debate_id, [])
    ]
    return DebateWithMessages(**debate.model_dump(), messages=msgs)

# ---------- 4) Auto-turn (assistant generates its move) ----------
class AutoTurnOut(BaseModel):
    message_id: UUID
    content: str
    round_no: int
    next_speaker: Optional[Speaker]
    current_round: int
    status: Status

@app.post("/v1/debates/{debate_id}/auto-turn", response_model=AutoTurnOut)
def auto_turn(debate_id: UUID):
    debate = DEBATES.get(debate_id)
    if not debate:
        raise HTTPException(404, "Debate not found")
    if debate.status != "active":
        raise HTTPException(400, f"Debate is {debate.status}")
    if debate.next_speaker != "assistant":
        raise HTTPException(409, "It's not the assistant's turn.")

    history = MESSAGES.get(debate_id, [])
    ai_text = generate_ai_turn_text(debate, history)

    msg = _append_message_and_advance(debate, "assistant", ai_text)

    return AutoTurnOut(
        message_id=msg.id,
        content=msg.content,
        round_no=msg.round_no,
        next_speaker=debate.next_speaker if debate.status == "active" else None,
        current_round=debate.current_round,
        status=debate.status,
    )

# ---------- 5) Finish early ----------
class FinishOut(BaseModel):
    status: Status
    current_round: int
    next_speaker: Optional[Speaker]

@app.post("/v1/debates/{debate_id}/finish", response_model=FinishOut)
def finish_debate(debate_id: UUID):
    debate = DEBATES.get(debate_id)
    if not debate:
        raise HTTPException(404, "Debate not found")
    debate.status = "completed"
    debate.updated_at = datetime.utcnow()
    # Optional: nullify next speaker when completed
    next_sp = None
    return FinishOut(status=debate.status, current_round=debate.current_round, next_speaker=next_sp)

@app.post("/v1/transcribe", response_model=TranscribeOut)
async def transcribe(file: UploadFile = File(...)):
    # Basic content-type check
    if not file.content_type or "audio" not in file.content_type:
        # Some browsers send octet-stream; still allow if filename looks like audio
        allowed = (file.filename or "").lower().endswith((".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm"))
        if not allowed:
            raise HTTPException(400, "Please upload an audio file")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty file")

    try:
        text = transcribe_with_whisper(audio_bytes, filename=file.filename or "audio.wav")
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        # Surface a concise error; log details in real app
        raise HTTPException(502, f"Transcription failed: {e}")

    return TranscribeOut(text=text)


@app.post("/v1/debates/{debate_id}/score", response_model=ScoreOut)
def score_debate(debate_id: UUID):
    debate = DEBATES.get(debate_id)
    if not debate:
        raise HTTPException(404, "Debate not found")
    if debate.status != "completed":
        raise HTTPException(409, "Debate must be completed before scoring")

    messages = MESSAGES.get(debate_id, [])
    breakdown = compute_debate_score(debate, messages)
    SCORES[debate_id] = breakdown

    return _score_out_from_breakdown(debate_id, breakdown)


@app.get("/v1/debates/{debate_id}/score", response_model=ScoreOut)
def get_score(debate_id: UUID):
    debate = DEBATES.get(debate_id)
    if not debate:
        raise HTTPException(404, "Debate not found")

    breakdown = SCORES.get(debate_id)
    if not breakdown:
        raise HTTPException(404, "Score not found. Score the debate first.")

    # Backfill missing feedback fields for legacy scores
    if not getattr(breakdown, "feedback", None):
        messages = MESSAGES.get(debate_id, [])
        try:
            breakdown = compute_debate_score(debate, messages)
            SCORES[debate_id] = breakdown
        except HTTPException:
            breakdown = ScoreBreakdown(
                overall=getattr(breakdown, "overall", 0.0),
                metrics=getattr(
                    breakdown,
                    "metrics",
                    ScoreMetrics(content_structure=0.0, engagement=0.0, strategy=0.0),
                ),
            )

    return _score_out_from_breakdown(debate_id, breakdown)


# ---------- Health ----------
@app.get("/v1/health")
def health():
    return {"status": "ok"}
