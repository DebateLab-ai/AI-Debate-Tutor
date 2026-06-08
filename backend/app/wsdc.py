"""
Two-pass WSDC speech generation for intermediate and advanced difficulty.

Pass 1 — Structure: LLM outputs strict JSON covering opening, arguments,
          clashes/rebuttals, close. Low temperature; schema enforced.
          Model: Claude Sonnet 4.6 (both tiers).

Pass 2 — Render: LLM renders the JSON to spoken prose.
          Model: Haiku 4.5 for AP (any tier) and WSDC intermediate;
                 Sonnet 4.6 for WSDC advanced.
          Rationale: Pass 1 carries the argumentation; Pass 2 is delivery.
          AP already uses Haiku successfully — applying it to WSDC
          intermediate cuts ~50% of per-turn output cost with no observed
          quality drop. Advanced WSDC stays on Sonnet because tournament-
          level airtightness is the whole brand promise of that tier.

To revert to OpenAI, swap this file with wsdc_chatGPT.py.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

_anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

_SONNET = "claude-sonnet-4-6"
_HAIKU = "claude-haiku-4-5"


# ── Schema templates shown to the model ───────────────────────────────────

_FIRST_SPEAKER_SCHEMA = """\
{
  "opening": "<Rhetorical question reframing the moral stakes — OR a strategic real-world example pre-loading burden on the other team. Pick one, not both. Never repeat the motion.>",
  "arguments": [
    {
      "label": "Our first argument is [name]",
      "mechanisms": [
        "1. <Intro-level logic chain, no jargon. Each step follows from the last. A specific example may follow, but the logic must stand on its own first.>",
        "2. <same pattern>",
        "3. <same pattern — omit if only 2 mechanisms make sense>"
      ],
      "preemption": "<null OR: 'I want to quickly flag something the [opposition/proposition] might say. They cannot just come up here and say X — that is not enough because Y.'>",
      "weighing": "<null OR: why this wins on principle (e.g. moral / type-of-impact) even if they contest the pragmatics>"
    }
  ],
  "close": {
    "tagline": "<null OR one punchy sentence deliverable in a single breath. If it needs more than one sentence, set null.>",
    "sign_off": "<'Proud to propose' OR 'Proud to oppose'>"
  }
}"""

_REBUTTAL_SCHEMA = """\
{
  "roadmap": "<2-3 sentences, conversational. Start with 'In this speech...' or similar. Preview what you'll refute and whether you're adding arguments. BANNED: 'today', 'I will address the arguments on X', 'I will be arguing'.>",
  "rebuttals": [
    {
      "target": "their argument on [X]",
      "memory_jog": "<One short sentence to jog the judge's memory — NOT a re-explanation, NOT a quote. No 'as I understood it', no 'their position was'. Just the idea, plainly. E.g. 'They say mandating English boosts the economy.'>",
      "points": [
        "1. <Counter-point: logic chain first, where their premise fails and why, step by step. Example may follow to ground it.>",
        "2. <same pattern>",
        "3. <same pattern — omit if only 2 needed>"
      ]
    }
  ],
  "clashes": [
    {
      "question": "On the question of [X vs Y], which matters more?",
      "their_position": "<their claim on this question>",
      "our_response": "<turn it: show their argument actually works for our side, OR why our answer to the question wins>",
      "why_we_win": "<why winning this clash wins the round>"
    }
  ],
  "new_argument": {
    "label": "Our next argument is [name]",
    "mechanisms": ["1. <logic chain + example>", "2. <same>"],
    "weighing": "<null OR principled / type-of-impact claim>"
  },
  "close": {
    "tagline": "<null OR one punchy sentence>",
    "sign_off": "<'Proud to propose' OR 'Proud to oppose'>"
  }
}
Use rebuttals[] OR clashes[] — not both. Set the unused array to []. new_argument is optional; set null if full-sending rebuttal makes more sense."""


# ── Pass 1 ─────────────────────────────────────────────────────────────────

def _pass1_system(is_rebuttal: bool) -> list[dict]:
    """The static portion of the Pass 1 prompt — task framing, hard rules,
    and the output schema. Marked cache_control: ephemeral so repeat calls
    within ~5 minutes pay ~10% of input cost on this prefix.

    Cache key is (model, full system content), so first-speaker and rebuttal
    paths get their own cache slots — both common enough to amortize the write.
    """
    schema = _REBUTTAL_SCHEMA if is_rebuttal else _FIRST_SPEAKER_SCHEMA
    text = f"""You are generating a parliamentary debate speech. Output STRICT JSON only — no prose, no markdown, no commentary.

Hard rules:
- Never repeat the motion
- Tone: patient and composed; explain why the other side is wrong, never personal
- LOGIC > EXAMPLES. Mechanisms must be reasoning chains built from well-known facts that any layman would accept — not "evidence" cited like a research paper. Never say "research shows", "studies indicate", "a 2021 report found". Examples ground the logic; they do not replace it. A speech with three sharp logical mechanisms beats a speech with three name-drops every time.
- Accessible to a complete layman — if a concept needs intro-level explanation, give it

Output this schema (fill every field; null where marked optional):
{schema}"""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _pass1_user(
    motion: str,
    side: str,
    is_rebuttal: bool,
    opponent_speech: Optional[str],
    context_block: str,
    difficulty: str,
    format: str,
) -> str:
    sign_off = "Proud to propose" if side == "Government" else "Proud to oppose"

    if difficulty == "advanced":
        difficulty_note = (
            "ADVANCED — tournament-level. Fill every field completely and precisely. "
            "Mechanisms are airtight logical chains with no gaps. Preemptions and weighing "
            "should appear wherever they sharpen the case."
        )
    else:
        difficulty_note = (
            "INTERMEDIATE — fill every field thoughtfully. The schema is a guide; "
            "prioritise clear logic chains and real examples."
        )

    if format == "ap":
        format_note = (
            "FORMAT: American Parliamentary (AP). Judges evaluate like an 'intelligent voter' — "
            "they weigh impacts logically and are NOT persuaded by debate jargon or stylistic flourish. "
            "Mechanisms must move the intelligent voter through impact comparison, not aesthetic appeal. "
            "Skip rhetorical ornamentation; favour plain-spoken logical force."
        )
    else:
        format_note = "FORMAT: WSDC. Tournament-style worlds schools format."

    opponent_block = (
        f"\n\nOPPONENT'S SPEECH (respond ONLY to arguments actually made here):\n{opponent_speech}"
        if is_rebuttal and opponent_speech else ""
    )

    return f"""Motion: {motion}
Side: {side} ({'PROPOSE' if side == 'Government' else 'OPPOSE'} the motion)
Speaker: {'Rebuttal / later speaker' if is_rebuttal else 'First speaker'}
Sign-off: {sign_off}

{difficulty_note}

{format_note}{opponent_block}

{context_block}"""


def _run_pass1(
    motion: str,
    side: str,
    is_rebuttal: bool,
    opponent_speech: Optional[str],
    context_block: str,
    difficulty: str,
    format: str,
) -> dict:
    system = _pass1_system(is_rebuttal)
    user_msg = _pass1_user(motion, side, is_rebuttal, opponent_speech, context_block, difficulty, format)
    temp = 0.35 if difficulty == "advanced" else 0.4

    resp = _anthropic.messages.create(
        model=_SONNET,
        max_tokens=2048,
        temperature=temp,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = resp.content[0].text.strip()

    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


# ── Pass 2 ─────────────────────────────────────────────────────────────────

def _pass2_prompt(speech_json: dict, difficulty: str, format: str) -> str:
    difficulty_note = (
        "ADVANCED — well-drilled tournament speaker: airtight, authoritative, still natural. "
        "Every structural beat fully delivered; the pattern should feel natural, not robotic."
        if difficulty == "advanced"
        else "INTERMEDIATE — natural and human. The structure is a guide; the beats should "
        "land cleanly but not feel like a template being read out."
    )

    if format == "ap":
        format_note = (
            "FORMAT: American Parliamentary. The judge is an intelligent voter who weighs impacts "
            "logically and is NOT swayed by stylistic flourish or debate jargon. Strip the rhetoric. "
            "Substance over style. Be plain-spoken and direct."
        )
        word_count = "1080–1170 words"
    else:
        format_note = "FORMAT: WSDC. Tournament voice; spoken delivery."
        word_count = "1200–1300 words"

    return f"""You are a parliamentary debater rendering this JSON outline into spoken prose.
Single continuous speech — no headers, no JSON, no stage directions.

{difficulty_note}

{format_note}

Voice:
- Patient and composed throughout; explain why the other side is wrong, never attack personally
- Logic chains clear step by step — a layman must follow every link. Example after the logic, to ground it; never as a substitute.
- LOGIC > EXAMPLES. Never cite sources like a research paper. No "research shows", "studies indicate", "data suggests". Use well-known facts presented as reasoning, not citations.
- Sound spoken, not written; opening lands hard
- Roadmap: conversational. Start with "In this speech..." or similar. Banned: "today", "I will address the arguments on X", "I will be arguing". Transition into content naturally, e.g. "Right, now onto refutation."
- Sign-off ("Proud to propose"/"Proud to oppose") is the last words — nothing after
- Never repeat the motion

Signposting (mandatory — each label and each numbered item on its own new line, line breaks not paragraph breaks):
- Rebuttals: "First response — [memory_jog phrased naturally]." then "1. ...\\n2. ...\\n3. ..."
- Arguments: "Our first argument is [name]:" then numbered mechanisms on new lines
- Don't re-explain the opponent — the memory_jog is enough; the judge was in the room

Length: {word_count}. Cut filler, not substance — every sentence earns its place.

JSON:
{json.dumps(speech_json, indent=2)}"""


def _run_pass2(speech_json: dict, difficulty: str, format: str) -> str:
    prompt = _pass2_prompt(speech_json, difficulty, format)
    # WSDC advanced is the only tier we still send to Sonnet — it's the
    # "tournament-level" promise. Everything else (AP both tiers, WSDC
    # intermediate) runs on Haiku, ~3x cheaper on output tokens.
    if format == "ap" or difficulty == "intermediate":
        model = _HAIKU
    else:
        model = _SONNET

    resp = _anthropic.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0.6,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


# ── Public entry point ──────────────────────────────────────────────────────

def generate_wsdc_speech(
    rag,
    motion: str,
    side: str,
    is_rebuttal: bool,
    opponent_speech: Optional[str] = None,
    difficulty: str = "intermediate",
    format: str = "wsdc",
    top_k: int = 6,
    min_score: float = 0.1,
) -> str:
    """
    Two-pass WSDC speech generation.
    Returns rendered spoken prose ready to display.
    """
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
            format=format,
        )
        print(f"[WSDC] Pass 1 OK (format={format}) — keys: {list(speech_json.keys())}")
    except Exception as e:
        print(f"[WSDC] Pass 1 failed: {e}")
        return "[Error generating speech structure. Please try again.]"

    try:
        result = _run_pass2(speech_json, difficulty, format)
        print(f"[WSDC] Pass 2 OK (format={format}) — {len(result)} chars")
        return result
    except Exception as e:
        print(f"[WSDC] Pass 2 failed: {e}")
        return "[Error polishing speech. Please try again.]"
