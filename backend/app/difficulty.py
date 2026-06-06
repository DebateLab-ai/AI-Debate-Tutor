"""Difficulty-tier configuration for AI debate generation and scoring.

Single source of truth for everything that varies by tier. Other modules just
read `debate.difficulty` and ask `get_config(...)` what to do — no tier-specific
branching scattered through the codebase.

Intentionally minimal:
  * `intermediate` defaults to "do exactly what today's code does" — empty
    addendums, no overrides — so the existing website experience is untouched
    when no tier is specified.
  * `beginner` shortens responses, calms temperature, gentle-scoring prompt, and
    floors the score so a beginner never gets a crushing number.
  * `advanced` raises temperature, widens RAG retrieval, can read from a
    separate `corpus/advanced/` folder (falls back to the standard corpus if
    that folder is missing or empty), and asks for sharper critique.

Tuning these values does not require touching call sites.
"""

from dataclasses import dataclass
from typing import Literal, Optional

Difficulty = Literal["beginner", "intermediate", "advanced"]
DEFAULT_DIFFICULTY: Difficulty = "intermediate"


@dataclass(frozen=True)
class DifficultyConfig:
    # Appended to debate system prompts (after SAFETY_PREAMBLE). Empty = no change.
    prompt_addendum: str = ""

    # Hard ceiling on response length for non-RAG generation. None = leave as-is.
    max_tokens: Optional[int] = None

    # Temperature for non-RAG generation. None = leave as-is.
    temperature: Optional[float] = None

    # RAG temperature band (passed through to generate_debate_with_coach_loop /
    # generate_rebuttal_speech as temp_low/temp_high).
    rag_temp_low: float = 0.3
    rag_temp_high: float = 0.8

    # RAG retrieval depth. Higher top_k + lower min_score = wider net.
    rag_top_k: int = 6
    rag_min_score: float = 0.1

    # Whether this tier should use the RAG path at all when otherwise eligible.
    # Beginner skips RAG so the prompt_addendum (which controls vocabulary and
    # length) actually governs the response — the RAG generation functions use
    # their own internal prompts that aren't easily augmented per-tier.
    use_rag: bool = True

    # Corpus subfolder relative to backend/app/corpus/. "" = use the root corpus.
    # Falls back to the root corpus if the subfolder is missing or empty.
    corpus_subdir: str = ""

    # Appended to compute_debate_score's system prompt (at the end, so the
    # difficulty has the last word on tone).
    scoring_addendum: str = ""

    # Minimum allowed value for each metric and the overall score. None = no floor.
    score_floor: Optional[float] = None


DIFFICULTY: dict[Difficulty, DifficultyConfig] = {
    "beginner": DifficultyConfig(
        prompt_addendum=(
            "DIFFICULTY: BEGINNER\n"
            "- Aim for 1-2 short paragraphs (roughly 7-10 sentences) — enough to "
            "develop the point properly, not so much it overwhelms.\n"
            "- When a debate term or specialized concept comes up, take a sentence "
            "or two to explain it the way you would to a friend who's new to debate: "
            "patient, plain, no jargon-dropping, and never a 'you should already "
            "know this' framing.\n"
            "- Make ONE clear point per turn and support it simply.\n"
            "- Do not introduce more than one new argument at a time.\n"
            "- Tone is collaborative — like a fellow debater walking through the "
            "idea with them, not a coach lecturing from above.\n\n"
        ),
        max_tokens=500,
        temperature=0.5,
        rag_temp_low=0.3,
        rag_temp_high=0.6,
        rag_top_k=4,
        rag_min_score=0.15,
        use_rag=False,  # beginner skips RAG so the prompt addendum governs
        scoring_addendum=(
            "BEGINNER DIFFICULTY OVERRIDE: Lead with what the human did well. Frame any "
            "critique gently and constructively. Use encouraging language and avoid harsh "
            "phrasing. The goal is to help them improve, not to discourage them.\n\n"
        ),
        score_floor=5.0,
    ),
    # Intermediate = today's behavior, byte-for-byte.
    "intermediate": DifficultyConfig(),
    "advanced": DifficultyConfig(
        prompt_addendum=(
            "DIFFICULTY: ADVANCED\n"
            "- Sharp, aggressive, and creative. Do not pull punches.\n"
            "- Surface non-obvious framings and contrarian angles.\n"
            "- Steelman the opponent's strongest interpretation, then dismantle it.\n"
            "- Use specialist or technical examples when they sharpen the argument.\n"
            "- Demand precision; exploit logical gaps mercilessly.\n\n"
        ),
        temperature=0.85,
        rag_temp_low=0.4,
        rag_temp_high=1.0,
        rag_top_k=10,
        rag_min_score=0.05,
        corpus_subdir="advanced",
        scoring_addendum=(
            "ADVANCED DIFFICULTY OVERRIDE: Apply rigorous, high-standards critique. Assume "
            "an experienced debater who can handle direct feedback. Reward technical "
            "precision and sophisticated reasoning; do not inflate scores for effort.\n\n"
        ),
    ),
}


def get_config(difficulty: Optional[Difficulty]) -> DifficultyConfig:
    """Resolve any difficulty value (including None or unknown) to a config.
    Falls back to the intermediate default so missing/legacy data is safe."""
    if difficulty in DIFFICULTY:
        return DIFFICULTY[difficulty]  # type: ignore[index]
    return DIFFICULTY[DEFAULT_DIFFICULTY]


def apply_score_floor(value: float, floor: Optional[float]) -> float:
    """Clamp a score to >= floor; no-op when floor is None."""
    if floor is None:
        return value
    return max(value, floor)
