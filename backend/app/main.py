# app/main.py
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

import os
import re
from datetime import datetime
from typing import Literal, Dict, List, Optional
import json
from uuid import uuid4, UUID
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request, status
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import RAG functionality
from app.response import SimpleRAG, generate_debate_with_coach_loop, generate_rebuttal_speech

# ---------- Types ----------
# Load .env file from backend directory (parent of app directory) for local development
# In production (Railway), environment variables are set directly and take precedence
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
    print(f"[Config] Loaded .env from: {env_path}")
else:
    # In production, environment variables are set directly (Railway, etc.)
    load_dotenv()  # This will still check for .env in current directory, but env vars take precedence
    print(f"[Config] Using environment variables (production mode)")

# Check if API key is available (from .env file or environment variables)
api_key_loaded = bool(os.getenv('OPENAI_API_KEY'))
print(f"[Config] API Key loaded: {api_key_loaded}")
if not api_key_loaded:
    print("[WARNING] OPENAI_API_KEY not found! AI features will not work.")
Speaker = Literal["user", "assistant"]
Status = Literal["active", "completed"]

app = FastAPI(
    title="Debate MVP",
    description="AI-powered debate practice platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Custom error handler for validation errors - show user-friendly messages
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return user-friendly validation error messages instead of raw Pydantic errors"""
    errors = exc.errors()
    error_messages = []
    for error in errors:
        field_path = ".".join(str(loc) for loc in error["loc"] if loc != "body")
        msg = error["msg"]
        error_type = error.get("type", "")  # Get error type at the start
        
        # Map Pydantic errors to user-friendly messages
        if "max_length" in msg:
            if "content" in field_path.lower() or "rebuttal" in field_path.lower():
                error_messages.append("Your response is too long. Maximum length is 10,000 characters.")
            elif "motion" in field_path.lower() or "title" in field_path.lower():
                error_messages.append("Topic is too long. Maximum length is 500 characters.")
            else:
                error_messages.append(f"{field_path}: Value exceeds maximum length")
        elif "min_length" in msg:
            error_messages.append("This field cannot be empty.")
        elif "value_error" in error_type or "string_type" in error_type:
            error_messages.append(f"Invalid value for {field_path}")
        else:
            # Generic fallback
            error_messages.append(f"{field_path}: {msg}")
    
    # Return first error message (most relevant)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": error_messages[0] if error_messages else "Validation failed. Please check your input.",
            "error": error_messages[0] if error_messages else "Invalid input"
        }
    )

# CORS configuration - allow specific origins in production
# IMPORTANT: Cannot use allow_credentials=True with wildcard origins "*"
# So we always specify explicit origins to allow credentials
CORS_ORIGINS_ENV = os.getenv("CORS_ORIGINS", "").strip()

# Default production origins - always include these for safety
default_origins = [
    "https://debatelab.ai",
    "https://www.debatelab.ai",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

if CORS_ORIGINS_ENV == "*" or not CORS_ORIGINS_ENV:
    # If wildcard or empty, use default production origins
    CORS_ORIGINS = default_origins
else:
    # Parse comma-separated list and merge with defaults
    origins_list = [origin.strip() for origin in CORS_ORIGINS_ENV.split(",") if origin.strip()]
    # Combine and deduplicate, preserving order (custom origins first)
    CORS_ORIGINS = list(dict.fromkeys(origins_list + default_origins))

print(f"[CORS] Allowing origins: {CORS_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ---------- RAG System ----------
# Global RAG instance - will be initialized on startup
rag_system: Optional[SimpleRAG] = None

@app.on_event("startup")
def startup_event():
    """Initialize RAG system with corpus on app startup"""
    global rag_system
    try:
        corpus_dir = os.getenv("SPEECH_CORPUS_DIR", "./app/corpus")
        # Handle both relative and absolute paths
        if not os.path.isabs(corpus_dir):
            # Relative path - try from app directory
            app_dir = os.path.dirname(__file__)
            corpus_dir = os.path.join(app_dir, "corpus")

        print(f"[RAG] Initializing RAG system with corpus from: {corpus_dir}")
        rag_system = SimpleRAG()

        if os.path.exists(corpus_dir):
            try:
                rag_system.add_corpus_folder(corpus_dir, pattern=r".*\.txt$")
                print(f"[RAG] Indexed {len(rag_system.docs)} documents")
            except Exception as e:
                print(f"[RAG] Warning: Failed to load corpus from {corpus_dir}: {e}")
                print(f"[RAG] RAG system initialized but no documents loaded")
        else:
            print(f"[RAG] Warning: Corpus directory not found: {corpus_dir}")
            print(f"[RAG] RAG system initialized but no documents loaded")
    except Exception as e:
        print(f"[RAG] Error during startup: {e}")
        import traceback
        traceback.print_exc()
        # Initialize empty RAG system to prevent crashes
        rag_system = SimpleRAG()
    
    print(f"[CLEANUP] Maximum debates limit: {MAX_DEBATES} (configure via MAX_DEBATES env var)")
    print(f"[CLEANUP] Cleanup threshold: {CLEANUP_THRESHOLD} (cleanup runs when {MAX_DEBATES + CLEANUP_THRESHOLD} debates reached)")


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
    mode: Literal["casual", "parliamentary"] = "casual"

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

# Maximum number of debates to keep in memory (configurable via env var)
MAX_DEBATES = int(os.getenv("MAX_DEBATES", "1000"))
# Cleanup threshold: only cleanup when we're this many debates over the limit (buffer zone)
CLEANUP_THRESHOLD = int(os.getenv("CLEANUP_THRESHOLD", "50"))  # Default: cleanup when 50 over limit

def cleanup_old_debates(max_debates: int = MAX_DEBATES, threshold: int = CLEANUP_THRESHOLD):
    """
    Remove oldest debates to keep total count under max_debates.
    Uses a buffer zone to avoid cleanup on every request.
    Only runs cleanup when count exceeds (max_debates + threshold).
    Removes from DEBATES, MESSAGES, and SCORES dictionaries.
    """
    current_count = len(DEBATES)
    
    # Only cleanup if we're significantly over the limit (buffer zone)
    if current_count <= max_debates + threshold:
        return 0  # No cleanup needed - still within buffer zone
    
    # Calculate how many to remove (bring it back to max_debates)
    num_to_remove = current_count - max_debates
    
    # Sort debates by created_at (oldest first) - only when we need to cleanup
    debates_sorted = sorted(
        DEBATES.items(),
        key=lambda x: x[1].created_at
    )
    
    removed_count = 0
    
    # Remove oldest debates
    for debate_id, _ in debates_sorted[:num_to_remove]:
        # Remove from all dictionaries
        if debate_id in DEBATES:
            del DEBATES[debate_id]
        if debate_id in MESSAGES:
            del MESSAGES[debate_id]
        if debate_id in SCORES:
            del SCORES[debate_id]
        removed_count += 1
    
    if removed_count > 0:
        print(f"[CLEANUP] Removed {removed_count} old debates. Current count: {len(DEBATES)} (limit: {max_debates})")
    
    return removed_count

# ---------- Schemas (I/O) ----------
class DebateCreate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=500)  # Max 500 characters for topic
    num_rounds: int = Field(ge=1, le=10)  # Max 10 for casual mode, validated in endpoint
    starter: Speaker
    mode: Literal["casual", "parliamentary"] = "casual"

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
    mode: Literal["casual", "parliamentary"] = "casual"

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
    content: str = Field(min_length=1, max_length=10000)  # Max 10,000 characters per argument

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
    weakness_type: Optional[Literal["rebuttal", "structure", "weighing", "evidence", "strategy"]] = None


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

# ---------- Drill Schemas ----------
class DrillStartRequest(BaseModel):
    motion: str = Field(min_length=1, max_length=500)  # Max 500 characters for motion
    user_position: Literal["for", "against"]  # The position the user took in the debate
    weakness_type: Optional[Literal["rebuttal", "structure", "weighing", "evidence", "strategy"]] = None  # Type of drill to focus on

class DrillClaimResponse(BaseModel):
    claim: str
    claim_position: Literal["for", "against"]  # The position of the claim (opposite of user)

class DrillRebuttalSubmit(BaseModel):
    motion: str = Field(min_length=1, max_length=500)
    claim: str = Field(min_length=1, max_length=2000)
    claim_position: Literal["for", "against"]
    rebuttal: str = Field(min_length=1, max_length=10000)  # Max 10,000 characters per rebuttal
    weakness_type: Optional[Literal["rebuttal", "structure", "weighing", "evidence", "strategy"]] = None

class DrillRebuttalMetrics(BaseModel):
    refutation_quality: float  # 0-10: How well they negate/mitigate the claim
    evidence_examples: float   # 0-10: Quality of supporting evidence or counter-examples
    impact_comparison: float   # 0-10: Whether they weigh their response against the claim

class DrillRebuttalScore(BaseModel):
    overall_score: float  # 0-10
    metrics: DrillRebuttalMetrics
    feedback: str  # Specific feedback on what they did well and what to improve
    next_claim: str  # Next claim to practice with
    next_claim_position: Literal["for", "against"]

# Evidence Drill Schemas
class EvidenceClaimResponse(BaseModel):
    claim: str  # The claim the user needs to provide evidence for (5-6 sentences, on their side)
    claim_position: Literal["for", "against"]  # Same position as user

class EvidenceSubmit(BaseModel):
    motion: str = Field(min_length=1, max_length=500)
    claim: str = Field(min_length=1, max_length=2000)
    claim_position: Literal["for", "against"]
    evidence: str = Field(min_length=1, max_length=5000)  # User's evidence for the claim

class EvidenceMetrics(BaseModel):
    authenticity: float  # 0-10: Is it a real, verifiable example?
    recognition: float   # 0-10: Is it well-known? (NYT front page test)
    strategic_support: float  # 0-10: Does it meaningfully support the claim?

class EvidenceScore(BaseModel):
    overall_score: float  # 0-10
    metrics: EvidenceMetrics
    feedback: str  # Specific feedback on the evidence quality
    next_claim: str  # Next claim to practice with
    next_claim_position: Literal["for", "against"]

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
# Get API key from environment (works in both local .env and Railway env vars)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = None

# Initialize OpenAI client
if OPENAI_API_KEY:
    try:
        from openai import OpenAI  # OpenAI Python SDK ≥ 1.0
        client = OpenAI(api_key=OPENAI_API_KEY)
        print(f"[OpenAI] Client initialized successfully")
    except Exception as e:
        client = None
        print(f"[OpenAI] Failed to initialize client: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
else:
    client = None
    print("[OpenAI] No API key found - AI features will not work")

def generate_ai_turn_text(debate: Debate, messages: List[Message]) -> str:
    """
    Produce assistant text using RAG-powered generation when available.
    Falls back to basic GPT-4o-mini if RAG is not initialized.
    In casual mode, skips RAG and uses a more conversational prompt.
    """
    motion = debate.title if debate.title else "General debate topic"
    use_rag = debate.mode == "parliamentary" and rag_system is not None

    # If we have RAG, parliamentary mode, and this is a rebuttal (not the first turn)
    if use_rag and len(messages) > 0:
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

    # If this is the first turn and we have RAG (parliamentary mode)
    if use_rag and len(messages) == 0:
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

    # Fallback: use mode-appropriate prompts
    topic_context = f"Debate topic: {motion}"
    
    if debate.mode == "casual":
        # Casual mode: more conversational, no RAG, less formal structure, optimized for speed
        sys = """Sharp and confrontational. Match their depth.

CRITICAL RULE - ACCURACY (MUST FOLLOW):
- ONLY address arguments they ACTUALLY made in their message
- NEVER invent, assume, or respond to arguments they did NOT make
- If you want to address a point, you MUST be able to quote or reference the exact words they used
- If they didn't mention something, DO NOT act like they did
- When rebutting, explicitly reference what they said: "The proposition claims X..." where X is their actual words

HARD RULE - OPPONENT IDENTIFICATION:
- NEVER say "my opponent" or "the opponent"
- ALWAYS identify them as "the proposition" or "the opposition" based on their side
- If they are arguing FOR the motion, call them "the proposition"
- If they are arguing AGAINST the motion, call them "the opposition"

Build from FIRST PRINCIPLES: state premise, derive each step logically where each step FOLLOWS NECESSARILY from the previous.
Use ONLY well-known examples (Amazon, iPhone, COVID-19). NEVER "research by X showed Y".
Always weigh. No filler. Vary your language. Do NOT include tags like "[Round X · ASSISTANT]" in your response."""
    else:
        # Parliamentary mode: formal debate structure
        sys = """You are a competitive debater. Win through sharp logic, not aggression.

CRITICAL RULE - ACCURACY (MUST FOLLOW):
- ONLY address arguments they ACTUALLY made in their messages
- NEVER invent, assume, or respond to arguments they did NOT make
- If you want to address a point, you MUST be able to quote or reference the exact words they used
- If they didn't mention something, DO NOT act like they did
- When rebutting, explicitly reference what they said with quotes or specific paraphrases
- Review the conversation history carefully - only respond to what is actually there

HARD RULE - OPPONENT IDENTIFICATION:
- NEVER say "my opponent" or "the opponent"
- ALWAYS identify them as "the proposition" or "the opposition" based on their side
- If they are arguing FOR the motion, call them "the proposition"
- If they are arguing AGAINST the motion, call them "the opposition"

REQUIREMENTS:
- Match their depth/detail (if they wrote 3 developed arguments, you write 3-4)
- Only rebut what they ACTUALLY said - no invented arguments
- HARD RULE FOR OPPOSITION: Make at least ONE new argument, but NO MORE than their number of arguments. If they made 2 arguments, you make 1-2 new arguments (not 0, not 3+).
- Build from FIRST PRINCIPLES: state premise, derive each step logically where each step FOLLOWS NECESSARILY from the previous. No logical leaps.
- Show causal chains step-by-step
- Use ONLY well-known examples (Amazon, iPhone, COVID-19, major events). NEVER cite "research by X showed Y"
- Weigh constantly using probability/magnitude/timeframe
- Signpost clearly but vary language naturally
- Sound like spoken debate, not essay. No filler. Do NOT include tags like "[Round X · ASSISTANT]" in your response."""
    convo = []
    for m in messages:
        role = "user" if m.speaker == "user" else "assistant"
        tag = f"[Round {m.round_no} · {m.speaker.upper()}]"
        convo.append({"role": role, "content": f"{tag} {m.content}"})

    # Determine if this is first speech or rebuttal
    opponent_messages = [m for m in messages if m.speaker != debate.next_speaker]
    if opponent_messages:
        # Rebuttal situation
        if debate.mode == "casual":
            # Get the last user message to reference their specific points - include FULL message
            last_user_msg = opponent_messages[-1].content if opponent_messages else ""
            prompt_now = f"""Topic: {topic_context}
Round {debate.current_round} of {debate.num_rounds}

CRITICAL: Here is what they ACTUALLY said (respond ONLY to this, nothing else):
"{last_user_msg}"

HARD RULE - ACCURACY:
- ONLY rebut arguments that appear in the text above
- If you mention something they said, you MUST be able to point to where they said it
- NEVER respond to arguments they did NOT make
- If they didn't mention X, do NOT say "they claim X" or "they argue X"
- When rebutting, quote or paraphrase their actual words

Respond (max 2 paragraphs). Start with roadmap. Build from FIRST PRINCIPLES: identify their premise → show why it fails step-by-step (each step follows necessarily) → establish your premise → derive conclusion. Well-known examples only."""
        else:
            # Include full opponent message for reference
            last_opponent_msg = opponent_messages[-1].content if opponent_messages else ""
            prompt_now = f"""Motion: {topic_context}
Round {debate.current_round} of {debate.num_rounds}

CRITICAL: Here is what the opponent ACTUALLY said (respond ONLY to this):
"{last_opponent_msg}"

Deliver a rebuttal speech that:
1. Starts with roadmap (vary language naturally)
2. Tears down opponent's key arguments (address strongest first):
   - ONLY rebut arguments that appear in the text above
   - Build rebuttals from FIRST PRINCIPLES: identify their premise, show step-by-step why the logical chain breaks down
   - When rebutting, quote or paraphrase their actual words - do NOT invent what they said
   - Negate first (show why claim is NOT true), then mitigate (show it's smaller), then concede+outweigh
   - Use ONLY well-known examples. NEVER cite "research by X showed Y"
   - HARD RULE: If they didn't mention X, do NOT say "they claim X" or respond to X
3. HARD RULE: Presents new arguments proportional to opponent's count:
   - Count their arguments, then make at least ONE new argument, but NO MORE than they made
   - If they made 1 arg → you make 1 new arg
   - If they made 2 args → you make 1-2 new args
   - If they made 3 args → you make 1-3 new args
4. Weighs comparatively throughout

Sharp, precise, rigorous. Only respond to what they actually said."""
    else:
        # Opening speech
        if debate.mode == "casual":
            prompt_now = f"""Topic: {topic_context}
Round {debate.current_round} of {debate.num_rounds}

Max 2 paragraphs. Start with roadmap (vary language). Build 1+ argument from FIRST PRINCIPLES: state premise → derive each step (each follows necessarily) → conclusion. Well-known examples only. Conversational but rigorous."""
        else:
            prompt_now = f"""Motion: {topic_context}
Round {debate.current_round} of {debate.num_rounds}

Deliver an opening speech with this structure:
1. Roadmap (1-2 sentences, vary language naturally)
2. Opening hook (2-3 sentences with well-known example)
3. Framing & burdens (define lens strategically)
4. Contentions (1-3 arguments):
   - Each needs: PREMISE → MECHANISMS (2-3) from FIRST PRINCIPLES → WEIGHING
   - Build mechanisms like mathematical proofs: state premise, derive each step where each FOLLOWS NECESSARILY from previous
   - Use 2-3 sentences per mechanism
   - Use ONLY well-known examples (Amazon, iPhone, COVID-19). NEVER "research by X showed Y"
5. Conclusion

Compelling, rigorous, well-structured."""

    if client:
        try:
            # Use lower max_tokens for casual mode to ensure faster, shorter responses
            max_tokens = 300 if debate.mode == "casual" else None
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": sys}] + convo + [
                    {"role": "user", "content": prompt_now}
                ],
                temperature=0.7,  # Higher temp since no RAG context (matches response.py adaptive logic)
                max_tokens=max_tokens,
            )
            ai_response = resp.choices[0].message.content.strip()
            # Strip any [Round X · ASSISTANT] or [Round X · USER] tags that the AI might have included
            ai_response = re.sub(r'\[Round \d+ · (ASSISTANT|USER)\]\s*', '', ai_response, flags=re.IGNORECASE)
            return ai_response.strip()
        except Exception as e:
            print(f"[ERROR] OpenAI API call failed: {type(e).__name__}: {e}")
            # fall back to stub

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

    # Determine user position and number of rounds from debate
    user_is_for = "User: for" in (debate.title or "").lower() or "(User: FOR" in (debate.title or "")
    num_rounds = debate.num_rounds
    is_casual_mode = debate.mode == "casual"

    # Check if this is an opening speech (user went first with only one speech delivered)
    user_went_first = messages[0].speaker == "user" if messages else False
    user_messages_count = sum(1 for m in messages if m.speaker == "user")
    is_opening_speech = user_went_first and user_messages_count == 1
    
    system_prompt = (
        "You are DebateJudgeGPT, an expert debate adjudicator across APDA, Public Forum, and WSDC formats.\n"
        "You will be given the full transcript of a debate between a human debater and an AI sparring partner.\n"
        "IMPORTANT: Messages labeled with '[Round X · USER]' are from the HUMAN DEBATER. Messages labeled with '[Round X · ASSISTANT]' are from the AI.\n"
        "Your task is to evaluate ONLY the HUMAN DEBATER's performance (messages labeled 'USER'). Do NOT evaluate the AI's performance.\n"
        "Only quote and reference statements made by USER, not ASSISTANT.\n\n"
        f"Context: The human debater (USER) is {'FOR' if user_is_for else 'AGAINST'} the motion. The debate has {num_rounds} round(s).\n"
        f"Debate mode: {'CASUAL (practice/learning mode - be more lenient and encouraging)' if is_casual_mode else 'PARLIAMENTARY (competitive mode - use standard tournament scoring)'}\n"
        f"Speech type: {'OPENING SPEECH (user went first, no opponent speech yet)' if is_opening_speech else 'REBUTTAL/RESPONSE (opponent has spoken)'}\n\n"
        "Score the human on these metrics (0-10 each, integers only):\n"
        "1. Content & Structure – arguments are understandable and well-explained; logical links are explicit; easy to follow; jargon is handled; clear signposting/roadmap.\n"
        "2. Engagement/Opening Quality – (see scoring rule below - changes based on speech type)\n"
        "3. Strategy – prioritizes win conditions; allocates time well across offense/defense; collapses to strongest arguments; avoids overinvesting in weak lines.\n\n"
    )
    
    # Add mode-specific scoring rubric
    if is_casual_mode:
        system_prompt += (
            "CASUAL MODE SCORING RUBRIC (MUST FOLLOW - be GENEROUS but still ACCURATE):\n"
            "This is a practice/learning mode. The goal is to encourage improvement, not tournament-level precision.\n"
            "HOWEVER: Relevance and topic engagement are still CRITICAL even in casual mode.\n\n"
            "SCORING GUIDELINES FOR CASUAL MODE:\n"
            "- If the message is IRRELEVANT to the topic or doesn't address the debate motion → score 2-4 (don't inflate these!)\n"
            "- If the message is partially relevant but off-topic → score 4-5\n"
            "- If the message addresses the topic but is weak/unclear → score 5-6\n"
            "- If the message is relevant and makes reasonable arguments → score 6-7\n"
            "- If the message is relevant and well-developed → score 7-9\n"
            "- Only give 9-10 for clearly strong, relevant, well-structured performances\n\n"
            "BE GENEROUS with structure and effort, but STRICT about relevance. An irrelevant message should NOT get above 4-5 even in casual mode.\n\n"
            "SCORING RANGES FOR CASUAL MODE:\n"
            "9-10: Strong performance - Relevant, well-developed arguments, good structure, clear engagement.\n"
            "7-8: Good performance - Relevant arguments with reasonable development, decent structure, clear topic engagement.\n"
            "5-6: Adequate performance - Relevant to topic, makes arguments that support their side, basic structure, some understanding.\n"
            "3-4: Developing or OFF-TOPIC - Partially relevant but missing the point, OR shows effort but very unclear/underdeveloped.\n"
            "1-2: Weak or IRRELEVANT - Completely off-topic, incoherent, no substantive content, or doesn't address the debate motion.\n\n"
        )
    else:
        system_prompt += (
            "SCORING RUBRIC (MUST FOLLOW - use this to calibrate your scores):\n"
            "9-10: Exceptional - Tournament-winning level. Multiple well-developed arguments with clear mechanisms, strong weighing, excellent structure.\n"
            "7-8: Strong - Clearly competitive. Good arguments with logical development, decent engagement/weighing, mostly clear structure.\n"
            "5-6: Adequate - Makes reasonable arguments that support their side. Some logical development, basic engagement if applicable, understandable structure.\n"
            "3-4: Developing - Has noticeable flaws but shows genuine effort. Arguments present but underdeveloped, weak engagement, unclear structure.\n"
            "1-2: Weak - Minimal substantive engagement. Very underdeveloped or off-topic.\n\n"
        )

    system_prompt += (
        "STRATEGY SCORING GUIDELINES (MUST FOLLOW):\n"
        "- Strategy is hard to judge perfectly, so be generous with scoring:\n"
        "- If there is AT LEAST a conceivable way for the user to win the debate (even if not optimal) → score >= 6\n"
        "- If there is good weighing (explicit comparison of probability/magnitude/timeframe) → score >= 7\n"
        "- Only score < 6 if the strategy is fundamentally flawed (e.g., arguing wrong side, completely ignoring win conditions, no path to victory)\n"
        "- As long as the user makes arguments that support their side, that alone justifies >= 5\n\n"

        f"{'OPENING SPEECH' if is_opening_speech else 'REBUTTAL'} - Engagement/Opening Quality Scoring (MUST FOLLOW):\n"
    )

    if is_opening_speech:
        system_prompt += (
            "CRITICAL: This is an OPENING SPEECH. The user went first and has NOT had opportunity to engage with opponent.\n"
            "DO NOT score based on clash/refutation/engagement - there is nothing to engage with yet!\n"
            "Instead, for 'engagement_score', evaluate OPENING SPEECH QUALITY (0-10):\n"
            "- Does the argument clearly support the CORRECT side of the motion (FOR or AGAINST as specified)?\n"
            "- Is the material COMPARATIVE and likely to generate good debate? (Not just assertions, but comparative analysis)\n"
            "- Is the argument SUBSTANTIVE and not disingenuous? (Real mechanisms, not strawmen or bad faith)\n"
            "- Does it set up clash opportunities for the opposition to respond to?\n"
            "SCORING GUIDE for opening speeches:\n"
            "9-10: Argument clearly on correct side, highly comparative, substantive mechanisms, sets up excellent debate\n"
            "7-8: Argument on correct side, reasonably comparative, decent substance, generates debate\n"
            "5-6: Argument on correct side, somewhat comparative, adequate substance\n"
            "3-4: Argument on correct side but poorly developed OR comparative but not substantive\n"
            "1-2: Wrong side, disingenuous, or no comparative material\n"
            "0: Completely off-topic or incoherent\n"
            "For 'engagement_feedback', discuss: (1) whether argument is on correct side, (2) how comparative/substantive it is, (3) how well it sets up the debate.\n"
            "DO NOT mention lack of clash/refutation as a weakness - this is expected for opening speeches.\n"
        )
    else:
        system_prompt += (
            "This is a REBUTTAL/RESPONSE. The user has opportunity to engage with opponent's arguments.\n"
            "For 'engagement_score', evaluate ENGAGEMENT (0-10):\n"
            "- Direct refutation of opponent's arguments\n"
            "- Comparison and impact weighing\n"
            "- Turns and defense\n"
            "- Responsiveness to opponent's case\n"
            "For 'engagement_feedback', discuss quality of clash, refutation, and comparative analysis.\n"
        )

    system_prompt += (
        "\n"
        "Anti-vagueness requirement (MUST FOLLOW):\n"
        "- Every positive claim must include: (a) a short quote from the user (<=12 words) OR a very specific described behavior, and (b) why that helps win rounds.\n"
        "- Every criticism must include: (a) what was missing, (b) a CONCRETE EXAMPLE OF what it should have looked like, and (c) one concrete next-step drill.\n"
        "- Do NOT use generic phrases like 'well articulated', 'clear point', 'add evidence' unless immediately followed by a specific example.\n"

        "Provide:\n"
        f"- `overall_score`: holistic score (0-10) for the human debater. Weighted average: 40% content, 30% strategy, 30% Engagement{' (CASUAL MODE: Add +1 to +1.5 to the calculated weighted average to be more encouraging)' if is_casual_mode else ''}\n"
        "`feedback` must be 4–6 sentences total and follow this structure:\n"
        "Sentence 1: Biggest strength + quote/behavior + why it matters strategically.\n"
        "Sentence 2: Second strength + quote/behavior + why it matters.\n"
        "Sentence 3: Biggest weakness + what was missing + what it should have looked like. *MUST FOLLOW; if score was 0 then DONT say a weakness is 'not responsive to assistant'\n"
        "Sentence 4: Concrete next step drill (one drill) + what to measure next time.\n"
        "(Optional sentence 5-6): Strategy collapse advice tied to this specific speech.\n"

        "CRITICAL: You MUST return a JSON object with EXACTLY these 9 keys. Missing any key will cause the scoring to fail.\n\n"
        "MANDATORY JSON structure - you MUST include ALL 9 keys:\n"
        "{\n"
        '  "overall_score": <number 0-10 (integer)>,\n'
        '  "feedback": "<4-6 sentence string>",\n'
        '  "content_structure_score": <number 0-10 (integer)>,\n'
        '  "content_structure_feedback": "<2 sentence string with specific examples>",\n'
        '  "engagement_score": <number 0-10 (integer)>,\n'
        f'  "engagement_feedback": "<2 sentence string - for {"opening speeches, discuss correctness of side, comparativeness, and substantiveness" if is_opening_speech else "rebuttals, discuss clash, refutation, and comparative analysis"}>",\n'
        '  "strategy_score": <number 0-10 (integer)>,\n'
        '  "strategy_feedback": "<2 sentence string with specific examples>",\n'
        '  "weakness_type": "<one of: rebuttal, structure, weighing, evidence, strategy>"\n'
        "}\n\n"
        "For weakness_type: Identify the PRIMARY area that needs improvement based on the scores and feedback:\n"
        "- 'rebuttal': If engagement_score is lowest (or < 6) and feedback mentions refutation, direct engagement, or responding to opponent\n"
        "- 'structure': If content_structure_score is lowest (or < 6) and feedback mentions organization, signposting, clarity, or logical flow\n"
        "- 'weighing': If feedback mentions impact comparison, probability/magnitude/timeframe, or comparative analysis\n"
        "- 'evidence': If feedback mentions lack of examples, data, concrete scenarios, or real-world evidence\n"
        "- 'strategy': If strategy_score is lowest (or < 6) and feedback mentions time allocation, prioritization, or collapsing to strongest arguments\n"
        "If multiple areas are weak, choose the one with the lowest score. If scores are tied, prioritize: engagement > structure > strategy.\n\n"
        "YOU MUST PROVIDE ALL 8 KEYS. Do not omit content_structure_score, engagement_score, or strategy_score.\n"
        "Evidence requirement: each *_feedback must reference at least one specific behavior from the USER's messages in the transcript; include 1–2 short quotes (<=12 words) from USER messages when possible. Do NOT quote or reference ASSISTANT messages.\n"
        "CRITICAL: When providing feedback, only discuss what the USER (human debater) did. Never mention or evaluate what the ASSISTANT (AI) said.\n"
        "Return ONLY the JSON object. No markdown code blocks, no explanations, no additional text."
    )

    prompt = [
        {"role": "system", "content": system_prompt},
        *convo,
        {
            "role": "user",
            "content": (
                "Judge this debate according to the instructions. "
                "Evaluate ONLY the HUMAN DEBATER (messages labeled 'USER'). "
                "Be fair, constructive, and reference specific behaviors from USER messages only. "
                "Do NOT evaluate or quote ASSISTANT messages."
            ),
        },
    ]

    # Use gpt-4o for scoring - better JSON schema compliance than gpt-4o-mini
    scoring_model = os.getenv("SCORING_MODEL", "gpt-4o")
    
    try:
        resp = client.chat.completions.create(
            model=scoring_model,
            messages=prompt,
            temperature=0.5,
            response_format={"type": "json_object"},  # Force JSON output
        )
    except Exception as exc:
        # Log internally but never expose to user
        print(f"[ERROR] Scoring API call failed: {exc}")
        raise HTTPException(502, "Unable to score debate at this time. Please try again.")

    raw = resp.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Log internally but NEVER expose raw response to user
        print(f"[ERROR] Scoring response malformed, length: {len(raw)}")
        raise HTTPException(502, "Scoring service temporarily unavailable. Please try again.")

    # Log what keys the LLM actually returned
    print(f"[DEBUG] LLM returned keys: {list(parsed.keys())}")
    print(f"[DEBUG] Sample values: {str(parsed)[:500]}")  # Print first 500 chars of parsed response for debugging
    
    def _get_float(key: str, default: float = 0.0) -> float:
        value = parsed.get(key)
        if value is None:
            print(f"[WARNING] Key '{key}' is missing from LLM response. Available keys: {list(parsed.keys())}")
            return default
        if isinstance(value, (int, float)):
            result = float(value)
            print(f"[DEBUG] {key} = {result}")
            return result
        if isinstance(value, str):
            try:
                result = float(value)
                print(f"[DEBUG] {key} (from string) = {result}")
                return result
            except ValueError:
                print(f"[WARNING] Could not convert '{key}' value '{value}' to float")
                return default
        print(f"[WARNING] Unexpected type for '{key}': {type(value)}, value: {value}")
        return default

    # Get raw scores from LLM
    raw_content = _get_float("content_structure_score")
    raw_engagement = _get_float("engagement_score")
    raw_strategy = _get_float("strategy_score")
    raw_overall = _get_float("overall_score")
    
    # Apply casual mode bonus: add 0.2-0.5 points, but ONLY if the score is already reasonable
    # Don't inflate irrelevant or very poor responses - relevance still matters!
    if is_casual_mode:
        # Only apply bonus if raw score is >= 5 (meaning it's at least adequate/relevant)
        # For lower scores (< 5), don't apply bonus - these are likely irrelevant or off-topic
        if raw_content >= 5.0:
            content_bonus = min(0.5, max(0.2, 0.2 + (6.0 - raw_content) * 0.06))
            raw_content = min(10.0, raw_content + content_bonus)
        else:
            content_bonus = 0.0
            
        if raw_engagement >= 5.0:
            engagement_bonus = min(0.5, max(0.2, 0.2 + (6.0 - raw_engagement) * 0.06))
            raw_engagement = min(10.0, raw_engagement + engagement_bonus)
        else:
            engagement_bonus = 0.0
            
        if raw_strategy >= 5.0:
            strategy_bonus = min(0.5, max(0.2, 0.2 + (6.0 - raw_strategy) * 0.06))
            raw_strategy = min(10.0, raw_strategy + strategy_bonus)
        else:
            strategy_bonus = 0.0
            
        if raw_overall >= 5.0:
            overall_bonus = min(0.5, max(0.2, 0.2 + (6.0 - raw_overall) * 0.06))
            raw_overall = min(10.0, raw_overall + overall_bonus)
        else:
            overall_bonus = 0.0
        
        if content_bonus > 0 or engagement_bonus > 0 or strategy_bonus > 0 or overall_bonus > 0:
            print(f"[DEBUG] Casual mode bonus applied: content +{content_bonus:.1f}, engagement +{engagement_bonus:.1f}, strategy +{strategy_bonus:.1f}, overall +{overall_bonus:.1f}")
        else:
            print(f"[DEBUG] Casual mode: No bonus applied (scores < 5 indicate irrelevant/poor quality responses)")
    
    metrics = ScoreMetrics(
        content_structure=round(raw_content, 1),
        engagement=round(raw_engagement, 1),
        strategy=round(raw_strategy, 1),
    )

    overall = round(raw_overall, 1)

    # Extract weakness_type with validation
    weakness_type_raw = parsed.get("weakness_type", "").lower()
    valid_weakness_types = ["rebuttal", "structure", "weighing", "evidence", "strategy"]
    weakness_type = weakness_type_raw if weakness_type_raw in valid_weakness_types else None
    
    # Fallback: determine weakness_type from scores if not provided or invalid
    if not weakness_type:
        scores = {
            "engagement": metrics.engagement,
            "content_structure": metrics.content_structure,
            "strategy": metrics.strategy,
        }
        # Find lowest score, but only if engagement is scorable
        if metrics.engagement == 0 and "Not scorable" in parsed.get("engagement_feedback", ""):
            # Engagement not scorable, compare structure vs strategy
            if metrics.content_structure <= metrics.strategy:
                weakness_type = "structure"
            else:
                weakness_type = "strategy"
        else:
            lowest = min(scores.items(), key=lambda x: x[1])
            if lowest[0] == "engagement":
                weakness_type = "rebuttal"
            elif lowest[0] == "content_structure":
                weakness_type = "structure"
            else:
                weakness_type = "strategy"

    return ScoreBreakdown(
        overall=overall,
        metrics=metrics,
        feedback=parsed.get("feedback", "No overall feedback provided."),
        content_structure_feedback=parsed.get("content_structure_feedback", "No content/structure feedback provided."),
        engagement_feedback=parsed.get("engagement_feedback", "No engagement feedback provided."),
        strategy_feedback=parsed.get("strategy_feedback", "No strategy feedback provided."),
        weakness_type=weakness_type,
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
        weakness_type=breakdown.weakness_type,
    )

# ---------- 1) Create debate ----------
@app.post("/v1/debates", response_model=DebateOut, status_code=201)
def create_debate(body: DebateCreate):
    # Validate num_rounds based on mode
    if body.mode == "parliamentary" and body.num_rounds > 3:
        raise HTTPException(400, "Parliamentary mode supports up to 3 rounds")
    if body.mode == "casual" and body.num_rounds > 10:
        raise HTTPException(400, "Casual mode supports up to 10 rounds")
    
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
        mode=body.mode,
    )
    DEBATES[did] = debate
    MESSAGES[did] = []
    
    # Cleanup old debates to prevent memory overflow
    cleanup_old_debates()
    
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
    return DebateWithMessages(**debate.dict(), messages=msgs)

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
        # Log internally but never expose to user
        print(f"[ERROR] Transcription runtime error: {e}")
        raise HTTPException(500, "Transcription service not configured.")
    except Exception as e:
        # Log internally but NEVER expose error details
        print(f"[ERROR] Transcription failed: {e}")
        raise HTTPException(502, "Unable to transcribe audio. Please try again.")

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


# ---------- Drill System ----------
def generate_drill_claim(
    motion: str, 
    claim_position: Literal["for", "against"],
    weakness_type: Optional[Literal["rebuttal", "structure", "weighing", "evidence", "strategy"]] = None
) -> str:
    """Generate a claim for the drill based on the motion, position, and weakness type."""
    if not client:
        return f"Sample claim {claim_position} the motion: {motion}"

    position_text = "FOR" if claim_position == "for" else "AGAINST"
    
    # Base prompt for all drill types
    base_instructions = (
        "You are a debate argument generator. Generate a single, strong claim that a debater might make (2-3 sentences max).\n\n"
        "CRITICAL - Build from FIRST PRINCIPLES with MATHEMATICAL RIGOR:\n"
        "- Start with a premise (axiom) and derive conclusions step-by-step\n"
        "- Each step must FOLLOW NECESSARILY from the previous one - no logical leaps\n"
        "- Explain the causal chain: if X, then Y happens BECAUSE Z\n"
        "- Provide clear reasoning for WHY the claim is true, not just WHAT happens\n\n"
        "EXAMPLES MUST BE WELL-KNOWN ONLY:\n"
        "- Use only examples laypeople recognize (Amazon, iPhone, COVID-19, major historical events)\n"
        "- NEVER cite 'research by X' or 'a study showed Y' - these can be fabricated\n"
        "- Debate is a game of LOGIC, not who cites more sources\n\n"
        "Do NOT include labels like 'Claim:', just output the argument directly."
    )
    
    # Weakness-specific instructions
    weakness_instructions = {
        "rebuttal": (
            "Focus: Generate a claim that requires direct refutation.\n"
            "- Make a clear, testable assertion that can be challenged\n"
            "- Include a mechanism that has potential flaws or assumptions"
        ),
        "structure": (
            "Focus: Generate a claim that needs clear organization.\n"
            "- Be complex enough to require signposting and structure\n"
            "- Have multiple components that need logical ordering"
        ),
        "weighing": (
            "Focus: Generate a claim with significant impacts.\n"
            "- Emphasize probability, magnitude, or timeframe of impacts\n"
            "- Present impacts that can be weighed against alternatives"
        ),
        "evidence": (
            "Focus: Generate a claim needing concrete evidence.\n"
            "- Make factual assertions supported with WELL-KNOWN examples only\n"
            "- Reference well-known scenarios (Amazon, iPhone, major events) - NOT obscure studies"
        ),
        "strategy": (
            "Focus: Generate a claim requiring strategic prioritization.\n"
            "- Present multiple potential responses\n"
            "- Have varying levels of strength needing strategic allocation"
        ),
    }
    
    if weakness_type and weakness_type in weakness_instructions:
        system_prompt = f"{base_instructions}\n\n{weakness_instructions[weakness_type]}"
    else:
        system_prompt = base_instructions

    user_prompt = f"Motion: {motion}\n\nGenerate one strong argument {position_text} this motion."

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=150,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[DRILL] Claim generation failed: {e}")
        return f"Sample claim {claim_position} the motion: {motion}"


def score_drill_rebuttal(
    motion: str, 
    claim: str, 
    claim_position: str, 
    rebuttal: str,
    weakness_type: Optional[Literal["rebuttal", "structure", "weighing", "evidence", "strategy"]] = None
) -> dict:
    """Score a drill rebuttal and return metrics + feedback, tailored to weakness type."""
    if not client:
        raise HTTPException(503, "Drill scoring requires OpenAI API key")

    # Base evaluation criteria
    base_criteria = (
        "You are a debate coach evaluating a student's drill response.\n\n"
        "The student was given a claim and asked to respond to it. Evaluate their response on:\n"
        "1. Refutation Quality (0-10): How well do they negate or mitigate the claim? Do they:\n"
        "   - Build rebuttals from FIRST PRINCIPLES (state premise, derive conclusions step-by-step)?\n"
        "   - Show each logical step FOLLOWS NECESSARILY from the previous one (no logical leaps)?\n"
        "   - Identify flaws in the opponent's logical chain?\n"
        "2. Evidence/Examples (0-10): Do they use WELL-KNOWN examples only?\n"
        "   - REWARD: Examples laypeople recognize (Amazon, iPhone, COVID-19, major historical events)\n"
        "   - PENALIZE HEAVILY: Citing 'research by X' or 'a study showed Y' (can be fabricated - debate is about LOGIC)\n"
        "   - REWARD: Building from logical premises rather than relying on citations\n"
        "3. Impact Comparison (0-10): Do they weigh their response against the claim? Do they explain why their point matters more or undermines the claim's significance?\n\n"
    )
    
    # Weakness-specific focus areas
    weakness_focus = {
        "rebuttal": (
            "FOCUS: Pay special attention to Refutation Quality. The student should:\n"
            "- Directly address the claim's logic and assumptions\n"
            "- Identify specific flaws or gaps in reasoning\n"
            "- Show clear negation or mitigation of the claim\n"
            "Weight Refutation Quality more heavily in overall_score.\n\n"
        ),
        "structure": (
            "FOCUS: Pay special attention to organization and clarity. The student should:\n"
            "- Use clear signposting and structure\n"
            "- Have logical flow and explicit links\n"
            "- Make it easy to follow their argument\n"
            "Weight clarity and organization more heavily in overall_score.\n\n"
        ),
        "weighing": (
            "FOCUS: Pay special attention to Impact Comparison. The student should:\n"
            "- Explicitly compare probability, magnitude, and timeframe\n"
            "- Make clear comparative statements\n"
            "- Explain why their response matters more\n"
            "Weight Impact Comparison more heavily in overall_score.\n\n"
        ),
        "evidence": (
            "FOCUS: Pay special attention to Evidence/Examples. The student should:\n"
            "- Use WELL-KNOWN examples only (Amazon, iPhone, major events) - NOT obscure research\n"
            "- PENALIZE if they cite 'research by X' or 'a study showed Y'\n"
            "- REWARD building from logical premises with well-known examples laypeople can verify\n"
            "- Reference WELL-KNOWN real-world scenarios as illustrations, not as proof\n"
            "Weight Evidence/Examples more heavily in overall_score.\n\n"
        ),
        "strategy": (
            "FOCUS: Pay special attention to strategic choices. The student should:\n"
            "- Prioritize the most important responses\n"
            "- Allocate time/space appropriately\n"
            "- Make clear strategic decisions about what to emphasize\n"
            "Weight strategic thinking more heavily in overall_score.\n\n"
        ),
    }
    
    if weakness_type and weakness_type in weakness_focus:
        system_prompt = base_criteria + weakness_focus[weakness_type]
    else:
        system_prompt = base_criteria
    
    system_prompt += (
        "Provide:\n"
        "- overall_score: Weighted average emphasizing the focus area above (0-10)\n"
        "- refutation_quality_score, evidence_examples_score, impact_comparison_score (0-10 each)\n"
        "- feedback: 2-3 sentences with ONE specific strength (with quote) and ONE concrete improvement focused on the weakness area (with example of what they could have said)\n\n"
        "Return ONLY a JSON object with keys: overall_score, refutation_quality_score, evidence_examples_score, impact_comparison_score, feedback\n"
        "Do not include any additional text outside the JSON."
    )

    user_prompt = (
        f"Motion: {motion}\n\n"
        f"Claim ({claim_position}): {claim}\n\n"
        f"Student's Response: {rebuttal}\n\n"
        f"Evaluate this response."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content.strip())

        # Extract scores with defaults
        overall = float(result.get("overall_score", 0))
        refutation = float(result.get("refutation_quality_score", 0))
        evidence = float(result.get("evidence_examples_score", 0))
        impact = float(result.get("impact_comparison_score", 0))
        feedback = result.get("feedback", "Good attempt. Keep practicing!")

        return {
            "overall_score": round(overall, 1),
            "refutation_quality": round(refutation, 1),
            "evidence_examples": round(evidence, 1),
            "impact_comparison": round(impact, 1),
            "feedback": feedback,
        }
    except Exception as e:
        # Log internally but NEVER expose error details
        print(f"[ERROR] Drill scoring failed: {e}")
        raise HTTPException(502, "Unable to score rebuttal. Please try again.")


@app.post("/v1/drills/rebuttal/start", response_model=DrillClaimResponse)
def start_rebuttal_drill(body: DrillStartRequest):
    """Start a drill - generates first claim for user to respond to, tailored to weakness type."""
    # Generate claim on the opposite side of the user's position
    claim_position = "against" if body.user_position == "for" else "for"
    claim = generate_drill_claim(body.motion, claim_position, body.weakness_type)

    return DrillClaimResponse(
        claim=claim,
        claim_position=claim_position
    )


@app.post("/v1/drills/rebuttal/submit", response_model=DrillRebuttalScore)
def submit_rebuttal_drill(body: DrillRebuttalSubmit):
    """Submit a rebuttal and get scored + next claim."""
    # Score the rebuttal (with weakness_type if provided)
    score_result = score_drill_rebuttal(
        body.motion,
        body.claim,
        body.claim_position,
        body.rebuttal,
        body.weakness_type
    )

    # Generate next claim (same position as before - user keeps responding to claims from opposite side)
    next_claim = generate_drill_claim(body.motion, body.claim_position, body.weakness_type)

    return DrillRebuttalScore(
        overall_score=score_result["overall_score"],
        metrics=DrillRebuttalMetrics(
            refutation_quality=score_result["refutation_quality"],
            evidence_examples=score_result["evidence_examples"],
            impact_comparison=score_result["impact_comparison"],
        ),
        feedback=score_result["feedback"],
        next_claim=next_claim,
        next_claim_position=body.claim_position,
    )


# ---------- Evidence Drill System ----------
def generate_evidence_claim(motion: str, claim_position: Literal["for", "against"]) -> str:
    """Generate a claim that needs evidence (5-6 sentences, on the user's side)."""
    if not client:
        return f"Sample claim {claim_position} the motion: {motion}"

    position_text = "FOR" if claim_position == "for" else "AGAINST"

    system_prompt = (
        "You are a debate argument generator. Generate a well-developed claim for evidence practice.\n\n"
        "REQUIREMENTS:\n"
        "- 5-6 sentences that present a complete argument\n"
        "- Build from FIRST PRINCIPLES: state premise, derive conclusions step-by-step\n"
        "- Each logical step must FOLLOW NECESSARILY from the previous one\n"
        "- Make a claim that REQUIRES concrete evidence to be convincing\n"
        "- DO NOT include examples yourself - leave room for the student to provide evidence\n"
        "- Focus on mechanisms and causal chains that need real-world validation\n\n"
        "The claim should be strong enough that with good evidence it would be persuasive,\n"
        "but incomplete without specific, well-known examples to support it.\n\n"
        "Do NOT include labels like 'Claim:', just output the argument directly."
    )

    user_prompt = f"Motion: {motion}\n\nGenerate a claim {position_text} this motion that requires evidence to be persuasive."

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[EVIDENCE DRILL] Claim generation failed: {e}")
        return f"Sample claim {claim_position} the motion: {motion}"


def score_evidence(motion: str, claim: str, evidence: str) -> dict:
    """Score evidence based on authenticity, recognition (NYT test), and strategic support."""
    if not client:
        raise HTTPException(503, "Evidence scoring requires OpenAI API key")

    system_prompt = (
        "You are a debate coach evaluating a student's evidence for a claim.\n\n"
        "Evaluate the evidence on three criteria (0-10 each):\n\n"
        "1. AUTHENTICITY (0-10): Is this a REAL example?\n"
        "   - 9-10: Clearly real, verifiable, specific details\n"
        "   - 7-8: Likely real, plausible details\n"
        "   - 5-6: Could be real but vague or generic\n"
        "   - 3-4: Questionable, lacks specificity\n"
        "   - 0-2: Clearly fabricated, impossible, or completely vague\n\n"
        "2. RECOGNITION (0-10): NYT Front Page Test - Is this well-known?\n"
        "   - 9-10: Major household name (Amazon, COVID-19, iPhone, World War II)\n"
        "   - 7-8: Well-known to educated audiences (major tech companies, famous historical events)\n"
        "   - 5-6: Moderately known (smaller companies, less famous events)\n"
        "   - 3-4: Obscure (local events, unknown companies, niche topics)\n"
        "   - 0-2: Completely unknown or cited as 'research by X' without specifics\n"
        "   CRITICAL: HEAVILY PENALIZE citations like 'a study by X' or 'research shows' - these fail the NYT test\n\n"
        "3. STRATEGIC SUPPORT (0-10): Does it meaningfully support the claim?\n"
        "   - 9-10: Directly proves the mechanism, perfect fit\n"
        "   - 7-8: Strongly supports the claim, clear connection\n"
        "   - 5-6: Somewhat relevant, loose connection\n"
        "   - 3-4: Tangentially related, weak connection\n"
        "   - 0-2: Irrelevant or contradicts the claim\n\n"
        "Provide:\n"
        "- overall_score: Average of the three metrics (0-10)\n"
        "- authenticity_score, recognition_score, strategic_support_score (0-10 each)\n"
        "- feedback: 2-3 sentences with:\n"
        "  * ONE specific strength (with quote if applicable)\n"
        "  * ONE concrete improvement (with example of what they could have used instead)\n\n"
        "Return ONLY a JSON object with keys: overall_score, authenticity_score, recognition_score, strategic_support_score, feedback\n"
        "Do not include any additional text outside the JSON."
    )

    user_prompt = (
        f"Motion: {motion}\n\n"
        f"Claim: {claim}\n\n"
        f"Student's Evidence: {evidence}\n\n"
        f"Evaluate this evidence."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content.strip())

        # Extract scores with defaults
        overall = float(result.get("overall_score", 0))
        authenticity = float(result.get("authenticity_score", 0))
        recognition = float(result.get("recognition_score", 0))
        strategic = float(result.get("strategic_support_score", 0))
        feedback = result.get("feedback", "Good attempt. Keep practicing!")

        return {
            "overall_score": round(overall, 1),
            "authenticity": round(authenticity, 1),
            "recognition": round(recognition, 1),
            "strategic_support": round(strategic, 1),
            "feedback": feedback,
        }
    except Exception as e:
        print(f"[ERROR] Evidence scoring failed: {e}")
        raise HTTPException(502, "Unable to score evidence. Please try again.")


@app.post("/v1/drills/evidence/start", response_model=EvidenceClaimResponse)
def start_evidence_drill(body: DrillStartRequest):
    """Start an evidence drill - generates a claim on the user's side that needs evidence."""
    # Generate claim on the SAME side as the user's position (not opposite)
    claim = generate_evidence_claim(body.motion, body.user_position)

    return EvidenceClaimResponse(
        claim=claim,
        claim_position=body.user_position
    )


@app.post("/v1/drills/evidence/submit", response_model=EvidenceScore)
def submit_evidence_drill(body: EvidenceSubmit):
    """Submit evidence and get scored + next claim."""
    # Score the evidence
    score_result = score_evidence(body.motion, body.claim, body.evidence)

    # Generate next claim (same position - user keeps providing evidence for claims on their side)
    next_claim = generate_evidence_claim(body.motion, body.claim_position)

    return EvidenceScore(
        overall_score=score_result["overall_score"],
        metrics=EvidenceMetrics(
            authenticity=score_result["authenticity"],
            recognition=score_result["recognition"],
            strategic_support=score_result["strategic_support"],
        ),
        feedback=score_result["feedback"],
        next_claim=next_claim,
        next_claim_position=body.claim_position,
    )


# ---------- Health ----------
@app.get("/v1/health")
def health():
    return {
        "status": "ok",
        "debates_count": len(DEBATES),
        "max_debates": MAX_DEBATES,
        "messages_count": sum(len(msgs) for msgs in MESSAGES.values()),
        "scores_count": len(SCORES)
    }
