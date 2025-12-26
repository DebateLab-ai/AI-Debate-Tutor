# app/main.py
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

import os
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
CORS_ORIGINS_ENV = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = ["*"] if CORS_ORIGINS_ENV == "*" else [origin.strip() for origin in CORS_ORIGINS_ENV.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
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

# Debug: Check all environment variables that start with OPENAI
print(f"[OpenAI] Checking for API key...")
print(f"[OpenAI] OPENAI_API_KEY exists: {bool(OPENAI_API_KEY)}")
if OPENAI_API_KEY:
    print(f"[OpenAI] API Key length: {len(OPENAI_API_KEY)}")
    print(f"[OpenAI] API Key starts with: {OPENAI_API_KEY[:10]}...")
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
    print("[OpenAI] No API key found in environment variables")
    print(f"[OpenAI] All env vars with 'OPENAI' in name: {[k for k in os.environ.keys() if 'OPENAI' in k.upper()]}")

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
        sys = """You directly challenge their arguments. Be sharp and confrontational.

RULES:
- START WITH ROADMAP: "Here's what I'll do: first, rebut your X. Second, make our case on Y."
- Match their depth/detail (if they wrote 2 developed points, you write 2-3)
- Only address what they ACTUALLY said - no invented arguments
- Show logical flaws step-by-step: "You claim X→Y, but this breaks down because..."
- When you cite examples, clarify: "This shows X, not Y..."
- Always weigh: "Even granting your point, our side wins because..."
- Every sentence must matter - no filler
- Be direct, not polite"""
    else:
        # Parliamentary mode: formal debate structure
        sys = """You are a competitive debater. Win through sharp logic, not aggression.

RULES:
- START WITH ROADMAP: "Here's my speech. First, I'll address their X. Second, I'll build our case on Y."
- Match their depth/detail (if they wrote 3 developed arguments, you write 3-4)
- Only rebut what they ACTUALLY said - no invented arguments
- Show causal chains step-by-step: "They claim X→Y. This breaks down because [gap]. In reality, X→Z because [step-by-step]..."
- Examples illustrate but don't prove: "Consider X. This shows Y, but doesn't prove Z because..."
- Always weigh: "Even granting their point, we win on [probability/magnitude/timeframe] because..."
- Signpost clearly: "First, their X argument. Second, their Y argument."
- Sound like spoken debate, not essay
- Every sentence must add substance - no filler, no platitudes
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
        if debate.mode == "casual":
            # Get the last user message to reference their specific points
            last_user_msg = opponent_messages[-1].content if opponent_messages else ""
            prompt_now = f"""Topic: {topic_context}
Round {debate.current_round} of {debate.num_rounds}

They just said: "{last_user_msg[:200]}{'...' if len(last_user_msg) > 200 else ''}"

CRITICAL: Start with a roadmap: "Here's what I'll do: first, rebut your X. Second, make our case."

Respond (max 2 paragraphs). First, refute their argument directly—attack their reasoning, challenge their examples, explain why their logic fails, and point out flaws. Then, make your own points to support your position. Be direct and confrontational. Don't acknowledge or agree with anything they said. Quote specific parts of what they said and explain why those points are wrong, then build your own case."""
        else:
            prompt_now = f"""Motion: {topic_context}
Current round: {debate.current_round} of {debate.num_rounds}

CRITICAL: Start with a roadmap: "Here's my speech. First, I'll address their X argument. Second, I'll extend our case on Y."

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
        if debate.mode == "casual":
            prompt_now = f"""Topic: {topic_context}
Round {debate.current_round} of {debate.num_rounds}

CRITICAL: Start with a roadmap: "Here's what I'll do: I'll make [number] arguments about [topic]."

Then share your perspective briefly (max 2 paragraphs). Build at least ONE clear argument with logical reasoning and examples. Keep it conversational and concise—like explaining your position quickly to a friend."""
        else:
            prompt_now = f"""Motion: {topic_context}
Current round: {debate.current_round} of {debate.num_rounds}

CRITICAL: Start with a roadmap: "Here's our case. First, [framework]. Second, I'll make [number] arguments: [brief labels]."

Deliver an opening speech following the structure:
1. Roadmap (1-2 sentences flagging what you'll do)
2. Opening hook (2-3 sentences with real-world example)
3. Framing & burdens
4. MINIMUM 1 contention, 2-3 preferred (each with premise, mechanisms, weighing)
5. Conclusion

Make your case compelling and well-structured."""

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
            return resp.choices[0].message.content.strip()
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
    
    system_prompt = (
        "You are DebateJudgeGPT, an expert debate adjudicator across APDA, Public Forum, and WSDC formats.\n"
        "You will be given the full transcript of a debate between a human debater and an AI sparring partner.\n"
        "IMPORTANT: Messages labeled with '[Round X · USER]' are from the HUMAN DEBATER. Messages labeled with '[Round X · ASSISTANT]' are from the AI.\n"
        "Your task is to evaluate ONLY the HUMAN DEBATER's performance (messages labeled 'USER'). Do NOT evaluate the AI's performance.\n"
        "Only quote and reference statements made by USER, not ASSISTANT.\n\n"
        f"Context: The human debater (USER) is {'FOR' if user_is_for else 'AGAINST'} the motion. The debate has {num_rounds} round(s).\n\n"
        "Score the human on these metrics (0-10 each, integers only):\n"
        "1. Content & Structure – arguments are understandable and well-explained; logical links are explicit; easy to follow; jargon is handled; clear signposting/roadmap.\n"
        "2. Engagement – direct refutation, comparison, impact weighing, turns/defense. (See engagement applicability rule below for when this is scorable.)\n"
        "3. Strategy – prioritizes win conditions; allocates time well across offense/defense; collapses to strongest arguments; avoids overinvesting in weak lines.\n\n"

        "SCORING RUBRIC (MUST FOLLOW - use this to calibrate your scores):\n"
        "9-10: Exceptional - Tournament-winning level. Multiple well-developed arguments with clear mechanisms, strong weighing, excellent structure.\n"
        "7-8: Strong - Clearly competitive. Good arguments with logical development, decent engagement/weighing, mostly clear structure.\n"
        "5-6: Adequate - Makes reasonable arguments that support their side. Some logical development, basic engagement if applicable, understandable structure.\n"
        "3-4: Developing - Has noticeable flaws but shows genuine effort. Arguments present but underdeveloped, weak engagement, unclear structure.\n"
        "1-2: Weak - Minimal substantive engagement. Very underdeveloped or off-topic.\n\n"

        "(MUST FOLLOW) for strategy: As long as the user makes arguments that supports their side, that is justification for a score of >= 5\n"

        "Engagement applicability rule (MUST FOLLOW):\n"
        "- Engagement is NOT scorable ONLY if: the user is FOR (proposition) AND there is only 1 round total.\n"
        "- In ALL other cases (user is AGAINST, OR user is FOR with 2+ rounds), engagement IS scorable and should be evaluated normally.\n"
        "- If engagement is not scorable (user is FOR with 1 round), set engagement_score=0 and engagement_feedback must be exactly: "
        "'Not scorable: the user had no opportunity to respond to the opposition.'\n"
        "- If engagement IS scorable, evaluate it normally (0-10) based on direct refutation, comparison, impact weighing, turns/defense.\n"
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

        "CRITICAL: You MUST return a JSON object with EXACTLY these 9 keys. Missing any key will cause the scoring to fail.\n\n"
        "MANDATORY JSON structure - you MUST include ALL 9 keys:\n"
        "{\n"
        '  "overall_score": <number 0-10 (integer)>,\n'
        '  "feedback": "<4-6 sentence string>",\n'
        '  "content_structure_score": <number 0-10 (integer)>,\n'
        '  "content_structure_feedback": "<2 sentence string with specific examples>",\n'
        '  "engagement_score": <number 0-10 (integer)>,\n'
        '  "engagement_feedback": "<2 sentence string OR exact text if not scorable>",\n'
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

    metrics = ScoreMetrics(
        content_structure=round(_get_float("content_structure_score"), 1),
        engagement=round(_get_float("engagement_score"), 1),
        strategy=round(_get_float("strategy_score"), 1),
    )

    overall = round(_get_float("overall_score"), 1)

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
        "You are a debate argument generator. Your job is to generate a single, strong claim "
        "that a debater might make in a debate.\n\n"
        "CRITICAL: Focus on LOGICAL FLOW, not examples. Think of this like a mathematical proof.\n\n"
        "The claim should:\n"
        "- Be one clear argument (2-3 sentences max)\n"
        "- Explain the causal chain: if X, then Y happens BECAUSE Z (show the logical mechanism)\n"
        "- Provide clear reasoning for WHY the claim is true, not just WHAT happens\n"
        "- Build logical connections step-by-step (e.g., 'If schools are overcrowded, class sizes increase, "
        "which means teachers have less time per student, therefore learning outcomes decline')\n"
        "- Be realistic and arguable (not obviously true/false)\n\n"
        "Do NOT just cite examples. Do NOT say 'For example, in Country X...'. "
        "Instead, explain the logical reason WHY something would occur.\n\n"
        "Do NOT include labels like 'Claim:', just output the argument directly."
    )
    
    # Weakness-specific instructions
    weakness_instructions = {
        "rebuttal": (
            "Focus: Generate a claim that requires direct refutation. The claim should:\n"
            "- Make a clear, testable assertion that can be challenged\n"
            "- Include a mechanism that has potential flaws or assumptions\n"
            "- Be structured so a good rebuttal would identify logical gaps or counter-examples"
        ),
        "structure": (
            "Focus: Generate a claim that needs clear organization. The claim should:\n"
            "- Be complex enough to require signposting and clear structure\n"
            "- Have multiple components that need logical ordering\n"
            "- Benefit from explicit framing and roadmap"
        ),
        "weighing": (
            "Focus: Generate a claim with significant impacts that need comparison. The claim should:\n"
            "- Emphasize probability, magnitude, or timeframe of impacts\n"
            "- Present impacts that can be weighed against alternatives\n"
            "- Require explicit impact calculus and comparative analysis"
        ),
        "evidence": (
            "Focus: Generate a claim that needs concrete evidence. The claim should:\n"
            "- Make specific factual assertions that can be supported with examples\n"
            "- Reference real-world scenarios or data points\n"
            "- Benefit from concrete counter-examples or case studies"
        ),
        "strategy": (
            "Focus: Generate a claim that requires strategic prioritization. The claim should:\n"
            "- Present multiple potential responses, requiring the debater to choose which to emphasize\n"
            "- Have varying levels of strength that need strategic allocation\n"
            "- Require decisions about time investment and argument selection"
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
        "1. Refutation Quality (0-10): How well do they negate or mitigate the claim? Do they identify flaws in logic, challenge assumptions, or show why the claim doesn't hold?\n"
        "2. Evidence/Examples (0-10): Do they use concrete counter-examples, data, or real-world scenarios to support their response?\n"
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
            "- Use concrete, specific examples or data\n"
            "- Reference real-world scenarios\n"
            "- Provide substantial evidence to support their response\n"
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


# ---------- Health ----------
@app.get("/v1/health")
def health():
    return {"status": "ok"}
