"""
Two-pass WSDC speech generation for intermediate and advanced difficulty.

Pass 1 — Structure: LLM outputs strict JSON covering opening, arguments,
          clashes/rebuttals, close. Low temperature; schema enforced via
          response_format=json_object.

Pass 2 — Polish: LLM renders the JSON to spoken WSDC prose. Slightly higher
          temperature; voice and delivery rules enforced.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from app.response import client as openai_client


# ── Schema templates shown to the model ───────────────────────────────────

_FIRST_SPEAKER_SCHEMA = """\
{
  "opening": "<rhetorical question that reframes the moral stakes — OR a strategic real-world example that pre-loads burden on the other team. NEVER repeat the motion.>",
  "arguments": [
    {
      "label": "Our first argument is [name]",
      "mechanisms": [
        "1. <logical chain — intro-level, no jargon. Then: for example, [specific real case that grounds the logic].>",
        "2. <same pattern>",
        "3. <same pattern — omit if only 2 mechanisms make sense>"
      ],
      "preemption": "<null — OR: flag what they will say and why it is not enough. Use natural language: 'I just want to quickly flag something the opposition might say. They cannot just come up here and say X — that is not enough because Y.'>",
      "weighing": "<null — OR: why this argument wins even if they contest the pragmatics, e.g. principled/moral claim>"
    }
  ],
  "close": {
    "tagline": "<null — OR one punchy sentence deliverable in a single breath. If it needs more than one sentence, set null.>",
    "sign_off": "<'Proud to propose' OR 'Proud to oppose'>"
  }
}"""

_REBUTTAL_SCHEMA = """\
{
  "roadmap": "<Brief, natural flag of what the speech will cover. Start with 'In this speech' or similar. State what you'll refute and whether you're adding arguments, then preview them: 'In this speech, I'll first refute their arguments on X and Y, then proceed with our own two arguments — first, that [A], second, that [B].' BANNED words/phrases: 'today', 'I will address the arguments on X', 'I will be arguing'. Keep it to 2-3 sentences max.>",
  "rebuttals": [
    {
      "target": "their argument on [X]",
      "response": "<Patient step-by-step refutation. Logic chain first — show where their premise fails and why, step by step. Then ground with a specific example.>"
    }
  ],
  "clashes": [
    {
      "question": "On the question of [X vs Y], which matters more?",
      "their_position": "<their claim on this question>",
      "our_response": "<show their argument actually works for our side — a turn — or why our answer is correct>",
      "why_we_win": "<why winning this question wins the round>"
    }
  ],
  "new_argument": {
    "label": "Our next argument is [name]",
    "mechanisms": [
      "1. <logical chain + example>",
      "2. <same pattern>"
    ],
    "weighing": "<null OR principled/type-of-impact claim>"
  },
  "close": {
    "tagline": "<null OR one punchy sentence>",
    "sign_off": "<'Proud to propose' OR 'Proud to oppose'>"
  }
}
IMPORTANT: Use rebuttals[] OR clashes[] — not both. Set the unused array to [].
new_argument is optional — set to null if full-sending rebuttal makes more strategic sense."""


# ── Pass 1 ─────────────────────────────────────────────────────────────────

def _pass1_prompt(
    motion: str,
    side: str,
    is_rebuttal: bool,
    opponent_speech: Optional[str],
    context_block: str,
    difficulty: str,
) -> str:
    sign_off = "Proud to propose" if side == "Government" else "Proud to oppose"
    schema = _REBUTTAL_SCHEMA if is_rebuttal else _FIRST_SPEAKER_SCHEMA

    if difficulty == "advanced":
        difficulty_note = (
            "ADVANCED — fill every field completely and precisely. "
            "Mechanisms must be tight logical chains with no gaps. "
            "Preemptions and weighing should appear wherever they are strategically useful. "
            "Even if it feels like filling in a template — fill it perfectly. "
            "This is tournament-level WSDC."
        )
    else:
        difficulty_note = (
            "INTERMEDIATE — fill every field thoughtfully. "
            "The schema is a guide; prioritise clear logical chains and real examples."
        )

    opponent_block = ""
    if is_rebuttal and opponent_speech:
        opponent_block = (
            f"\n\nOPPONENT'S SPEECH — respond ONLY to arguments actually made here:\n"
            f"{opponent_speech}"
        )

    return f"""You are generating a WSDC debate speech.
Output STRICT JSON only — no prose outside the JSON, no markdown, no commentary.

Motion: {motion}
Side: {side} ({'PROPOSE the motion' if side == 'Government' else 'OPPOSE the motion'})
Speaker type: {'Rebuttal / later speaker' if is_rebuttal else 'First speaker'}
Sign-off: {sign_off}

{difficulty_note}

CONTENT RULES:
- NEVER repeat the motion anywhere in the speech
- Opening (first speaker only): rhetorical question OR strategic world example — not both
- Mechanisms: logical chain first (clear, intro-level, zero jargon), then one specific real-world example to ground it
- Tone: patient and composed. Patiently explain why the other side is wrong — never contemptuous, never personal
- Clashes (rebuttal only): reframe the round around the key questions it turns on. A clash is a turn — show their argument works for you, or that your answer to the question beats theirs
- Tagline: one breath, punchy and memorable. If it needs more than one sentence, set null
- Accessible to a complete layman — if a concept needs intro-micro-level explanation, give it{opponent_block}

{context_block}

OUTPUT THIS EXACT JSON SCHEMA (fill every field; use null where marked optional):
{schema}"""


def _run_pass1(
    motion: str,
    side: str,
    is_rebuttal: bool,
    opponent_speech: Optional[str],
    context_block: str,
    difficulty: str,
) -> dict:
    prompt = _pass1_prompt(motion, side, is_rebuttal, opponent_speech, context_block, difficulty)
    temp = 0.35 if difficulty == "advanced" else 0.4
    model = "gpt-4o" if difficulty == "advanced" else "gpt-4o-mini"

    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content.strip()

    # Strip markdown fences if the model wraps anyway
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


# ── Pass 2 ─────────────────────────────────────────────────────────────────

def _pass2_prompt(speech_json: dict, difficulty: str) -> str:
    if difficulty == "advanced":
        difficulty_note = (
            "ADVANCED — every structural beat must be fully and precisely delivered. "
            "Sound like a well-drilled tournament speaker: airtight, authoritative, "
            "never improvised. The pattern should feel natural, not robotic."
        )
    else:
        difficulty_note = (
            "INTERMEDIATE — the structure is a guide. Sound natural and human. "
            "The beats should land cleanly but not feel like a template being read out."
        )

    return f"""You are a WSDC debater rendering a structured speech outline into spoken prose.
Render the JSON below as a single continuous speech — no headers, no JSON, no stage directions.

{difficulty_note}

VOICE RULES:
- Patient and composed throughout — patiently explain why the other side is wrong, never attack them personally
- Every logical chain must be clear step by step — a layman must follow every link
- Example always comes AFTER the logic — it grounds the argument, not replaces it
- Vary signposting naturally — do not repeat "firstly, secondly, thirdly" robotically
- Sound like someone speaking on their feet, not writing an essay
- Opening must land hard — do not soften or hedge it
- Roadmap: natural and conversational. Start with "In this speech..." or similar. Preview what you'll cover briefly. NEVER say "today", "I will address the arguments on X", or "I will be arguing". Transition into content with something like "Right, now onto refutation." or "So, let's get into it."
- If tagline is present: deliver it in one breath immediately before the sign-off
- If tagline is null: end with just the sign-off, no padding
- Never repeat the motion
- The sign-off ("Proud to propose" / "Proud to oppose") closes the speech — nothing after it

SPEECH JSON:
{json.dumps(speech_json, indent=2)}"""


def _run_pass2(speech_json: dict, difficulty: str) -> str:
    prompt = _pass2_prompt(speech_json, difficulty)
    temp = 0.65 if difficulty == "advanced" else 0.6
    model = "gpt-4o" if difficulty == "advanced" else "gpt-4o-mini"

    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    return resp.choices[0].message.content.strip()


# ── Public entry point ──────────────────────────────────────────────────────

def generate_wsdc_speech(
    rag,
    motion: str,
    side: str,
    is_rebuttal: bool,
    opponent_speech: Optional[str] = None,
    difficulty: str = "intermediate",
    top_k: int = 6,
    min_score: float = 0.1,
) -> str:
    """
    Two-pass WSDC speech generation.
    Returns rendered spoken prose ready to display.
    """
    if openai_client is None:
        return "[Error: OpenAI client not initialized]"

    # Retrieve RAG context if available
    context_block = ""
    if rag is not None and hasattr(rag, "retriever") and rag.retriever:
        query = f"{motion} {opponent_speech or ''}".strip()
        hits = rag.retriever.query(query, top_k=top_k)
        hits = [(d, s) for d, s in hits if s >= min_score]
        if hits:
            blocks = [d.text for d, _ in hits]
            context_block = (
                "BACKGROUND KNOWLEDGE — use as logical grounding. "
                "Never cite as 'research' or 'studies'. "
                "Present reasoning as your own first-principles analysis.\n"
                + "\n\n".join(blocks)
            )

    try:
        speech_json = _run_pass1(
            motion=motion,
            side=side,
            is_rebuttal=is_rebuttal,
            opponent_speech=opponent_speech,
            context_block=context_block,
            difficulty=difficulty,
        )
        print(f"[WSDC] Pass 1 OK — keys: {list(speech_json.keys())}")
    except Exception as e:
        print(f"[WSDC] Pass 1 failed: {e}")
        return "[Error generating speech structure. Please try again.]"

    try:
        result = _run_pass2(speech_json, difficulty)
        print(f"[WSDC] Pass 2 OK — {len(result)} chars")
        return result
    except Exception as e:
        print(f"[WSDC] Pass 2 failed: {e}")
        return "[Error polishing speech. Please try again.]"
