"""A/B compare scoring models on the same debate transcript.

Scoring is the single highest-stakes LLM call in a debate — it produces the
numeric score and feedback that lands on the partner-facing PDF. Cheaper is
only better if quality holds up. This script lets you eyeball the trade-off.

Run from backend/:
    source venv/bin/activate
    python scripts/compare_scoring_models.py

Prints per-model:
  - cost (computed from current published pricing)
  - latency
  - the actual score numbers (overall + 3 metrics)
  - the feedback text — so you can read it and judge

Then prints a recommendation based on what scored consistently with the baseline.

A real debate transcript is hard-coded so the comparison is apples-to-apples.
Models that error (e.g. unrecognized name) are reported and skipped — no crash.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from openai import OpenAI
import anthropic


# ── Pricing (USD per 1M tokens) ─────────────────────────────────────────
# Verify against vendor pricing pages before quoting partners. Models that
# aren't in here will report cost as "n/a" but the comparison still runs.
PRICES = {
    "gpt-4o":            {"in": 2.50,  "out": 10.00},
    "gpt-4o-mini":       {"in": 0.15,  "out":  0.60},
    "gpt-5-mini":        {"in": 0.25,  "out":  2.00},   # best-guess placeholder
    "o4-mini":           {"in": 1.10,  "out":  4.40},   # best-guess placeholder
    "claude-haiku-4-5":  {"in": 1.00,  "out":  5.00},
    "claude-sonnet-4-6": {"in": 3.00,  "out": 15.00},
}


def usd(model: str, in_t: int, out_t: int) -> Optional[float]:
    p = PRICES.get(model)
    if p is None:
        return None
    return (in_t * p["in"] + out_t * p["out"]) / 1_000_000


# ── Test transcript ──────────────────────────────────────────────────────
# Pulled from a representative WSDC intermediate debate. Two-round, user-as-
# opposition, moderate quality argumentation. Realistic stress test for
# scoring nuance.

MOTION = "THW ban single-use plastics globally"
TRANSCRIPT = [
    ("assistant", 1, """In this speech I'll argue that a global ban on single-use plastics is both necessary and feasible. First, the harms of plastics are not contested — they choke marine life, microplastics enter our food chain, and waste piles up in countries that lack disposal infrastructure. Second, the only intervention proportionate to a global problem is a global ban. Voluntary measures have failed for thirty years; meanwhile, plastic production has doubled. Recycling rates remain below 10 percent worldwide. The proposition is not naive about implementation difficulties, but we say the alternative — continued inaction while a generation of children inherits poisoned oceans — is morally indefensible. Proud to propose."""),
    ("user", 1, """The proposition's case rests on the assumption that a global plastic ban is enforceable, but that misreads the realities of developing economies. In Vietnam, where I live, informal plastic supply chains employ millions; an overnight ban would devastate livelihoods before any environmental benefit materialised. Worse, the alternatives — bioplastics, glass, paper — have hidden carbon costs that often exceed the harm of well-managed plastic recycling. The solution is investment in waste infrastructure, not prohibition."""),
    ("assistant", 2, """The opposition concedes the harm — they offer no challenge to our central claim that plastics are choking ecosystems. Their case rests on transition costs, but transition costs are not impact costs; they are the price of every meaningful reform. On informal labour: a ban with a phase-in period, paired with retraining subsidies, addresses this. On bioplastics: the opposition cherry-picks the worst alternatives; modern compostables match plastic's carbon footprint at a fraction of the marine harm. Their framework asks us to accept permanent harm to avoid temporary disruption. Proud to propose."""),
    ("user", 2, """Even granting the proposition's framing, they still owe us a mechanism. None of their arguments engage with the displacement problem I raised. They simply assert that the harm of plastics outweighs the harm of unemployment — without showing the timeframe or probability. On our side, the harm is immediate and concentrated; on theirs, it is diffuse and contingent. We win this debate on weighing alone."""),
]


# ── The shared scoring prompt ────────────────────────────────────────────
# Trimmed version of main.py:compute_debate_score. Holding the prompt fixed
# means any quality difference is the model's, not the prompt's.

SYSTEM_PROMPT = """You are DebateJudgeGPT, an expert debate adjudicator.
Evaluate ONLY the HUMAN DEBATER (messages labeled 'USER'). Ignore the AI.

Score 0-10 on:
1. Content & Structure — logic, signposting, clarity
2. Engagement — direct refutation, weighing, clash
3. Strategy — collapsing, prioritizing win conditions

For each *_feedback field: include 1-2 short quotes (<=12 words) from the USER's
messages. Be specific. No generic praise.

Return ONLY a JSON object with EXACTLY these keys:
{
  "overall_score": <number 0-10>,
  "feedback": "<3-4 sentences, structured: strength, strength, weakness+drill>",
  "content_structure_score": <number 0-10>,
  "content_structure_feedback": "<2 sentences with quotes>",
  "engagement_score": <number 0-10>,
  "engagement_feedback": "<2 sentences with quotes>",
  "strategy_score": <number 0-10>,
  "strategy_feedback": "<2 sentences with quotes>",
  "weakness_type": "<one of: rebuttal, structure, weighing, evidence, strategy>"
}"""


def build_transcript_messages() -> list[dict]:
    msgs = []
    for speaker, round_no, content in TRANSCRIPT:
        role = "user" if speaker == "user" else "assistant"
        tag = f"[Round {round_no} · {speaker.upper()}]"
        msgs.append({"role": role, "content": f"{tag} {content}"})
    return msgs


USER_PROMPT = (
    f"Motion: {MOTION}\n"
    "The human debater is OPPOSITION. Evaluate them only.\n"
    "Return ONLY the JSON object — no markdown, no commentary."
)


# ── Runners ──────────────────────────────────────────────────────────────

@dataclass
class Result:
    model: str
    ok: bool
    error: Optional[str] = None
    in_tokens: int = 0
    out_tokens: int = 0
    latency_s: float = 0.0
    parsed: Optional[dict] = None
    cost: Optional[float] = None


def run_openai(model: str, client: OpenAI) -> Result:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, *build_transcript_messages(),
            {"role": "user", "content": USER_PROMPT}]
    start = time.monotonic()
    try:
        kwargs = {
            "model": model,
            "messages": msgs,
            "response_format": {"type": "json_object"},
        }
        # Reasoning models (o-series) don't accept temperature; everything else gets 0.5.
        if not model.startswith("o"):
            kwargs["temperature"] = 0.5
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        return Result(model=model, ok=False, error=f"{type(e).__name__}: {e}")
    latency = time.monotonic() - start
    raw = resp.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
    except Exception as e:
        return Result(model=model, ok=False, error=f"JSON parse failed: {e}",
                      in_tokens=resp.usage.prompt_tokens, out_tokens=resp.usage.completion_tokens,
                      latency_s=latency)
    return Result(
        model=model, ok=True, parsed=parsed,
        in_tokens=resp.usage.prompt_tokens, out_tokens=resp.usage.completion_tokens,
        latency_s=latency, cost=usd(model, resp.usage.prompt_tokens, resp.usage.completion_tokens),
    )


def run_anthropic(model: str, client: anthropic.Anthropic) -> Result:
    transcript_text = "\n\n".join(
        f"[Round {rn} · {sp.upper()}] {c}" for sp, rn, c in TRANSCRIPT
    )
    user_prompt = f"{USER_PROMPT}\n\nTRANSCRIPT:\n{transcript_text}"
    start = time.monotonic()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0.5,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as e:
        return Result(model=model, ok=False, error=f"{type(e).__name__}: {e}")
    latency = time.monotonic() - start
    raw = resp.content[0].text.strip()
    # Anthropic doesn't have JSON-mode; strip code fences if present.
    import re
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
    except Exception as e:
        return Result(model=model, ok=False, error=f"JSON parse failed: {e}",
                      in_tokens=resp.usage.input_tokens, out_tokens=resp.usage.output_tokens,
                      latency_s=latency)
    return Result(
        model=model, ok=True, parsed=parsed,
        in_tokens=resp.usage.input_tokens, out_tokens=resp.usage.output_tokens,
        latency_s=latency, cost=usd(model, resp.usage.input_tokens, resp.usage.output_tokens),
    )


# ── Report ───────────────────────────────────────────────────────────────

def print_result(r: Result) -> None:
    print(f"\n{'─' * 72}")
    print(f"  {r.model}")
    print(f"{'─' * 72}")
    if not r.ok:
        print(f"  FAILED: {r.error}")
        return
    cost_str = f"${r.cost:.4f}" if r.cost is not None else "n/a (no pricing)"
    print(f"  cost: {cost_str}   latency: {r.latency_s:.1f}s   tokens: {r.in_tokens} in / {r.out_tokens} out")
    p = r.parsed
    print(f"\n  overall:     {p.get('overall_score')}")
    print(f"  content:     {p.get('content_structure_score')}")
    print(f"  engagement:  {p.get('engagement_score')}")
    print(f"  strategy:    {p.get('strategy_score')}")
    print(f"  weakness:    {p.get('weakness_type')}")
    print(f"\n  feedback:    {p.get('feedback', '')[:400]}")
    print(f"  content fb:  {p.get('content_structure_feedback', '')[:300]}")
    print(f"  engage fb:   {p.get('engagement_feedback', '')[:300]}")
    print(f"  strategy fb: {p.get('strategy_feedback', '')[:300]}")


def summary_table(results: list[Result]) -> None:
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"{'Model':<22} {'Status':<10} {'Overall':>8} {'Cost':>10} {'Latency':>10}")
    print("-" * 72)
    for r in results:
        if not r.ok:
            print(f"{r.model:<22} {'ERROR':<10} {'-':>8} {'-':>10} {'-':>10}")
            continue
        overall = r.parsed.get("overall_score", "?")
        cost = f"${r.cost:.4f}" if r.cost is not None else "n/a"
        print(f"{r.model:<22} {'ok':<10} {overall!s:>8} {cost:>10} {r.latency_s:>8.1f}s")


def main() -> None:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    candidates = [
        ("openai", "gpt-4o"),
        ("openai", "gpt-4o-mini"),
        ("openai", "gpt-5-mini"),     # may fail — that's the test
        ("openai", "o4-mini"),         # may fail — that's the test
        ("anthropic", "claude-haiku-4-5"),
        ("anthropic", "claude-sonnet-4-6"),
    ]

    results: list[Result] = []
    for vendor, model in candidates:
        print(f"\n[Running {vendor}:{model}...]")
        try:
            if vendor == "openai":
                r = run_openai(model, openai_client)
            else:
                r = run_anthropic(model, anthropic_client)
        except Exception:
            traceback.print_exc()
            r = Result(model=model, ok=False, error="unexpected exception")
        results.append(r)
        print_result(r)

    summary_table(results)

    print("\nNote: read the feedback text above — that's what determines if a cheaper model is")
    print("actually viable. A score within ±0.5 of gpt-4o with intelligible feedback = a win.")


if __name__ == "__main__":
    main()
