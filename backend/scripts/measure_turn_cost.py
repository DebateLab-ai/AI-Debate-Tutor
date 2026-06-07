"""Empirical per-turn cost measurement.

Runs a real WSDC intermediate debate (the partner-relevant code path:
Anthropic Sonnet Pass 1 + Anthropic Haiku Pass 2, plus OpenAI GPT-4o
scoring at the end). Monkey-patches the Anthropic / OpenAI clients to
capture token usage from every call, then prints a per-call breakdown
with dollar costs computed from published list pricing.

Run from backend/:
    source venv/bin/activate
    python scripts/measure_turn_cost.py

Costs roughly $0.10–$0.20 in real API spend per run. The output is the
authoritative answer to "how much does one turn cost"; use it to compute
per-debate and monthly projections for the partner.

Pricing constants are kept at the top — verify against current vendor
pricing pages before quoting a partner, as model prices change.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ── Pricing (USD per 1M tokens) ────────────────────────────────────────────
# Source: Anthropic + OpenAI public pricing pages. Verify before quoting.
# Cache write costs 1.25x base input; cache read costs 0.10x base input.

PRICES = {
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "claude-haiku-4-5":  {"in": 1.00, "out":  5.00},
    "gpt-4o":            {"in": 2.50, "out": 10.00},
    "gpt-4o-mini":       {"in": 0.15, "out":  0.60},
}


def usd_cost(model: str, in_tokens: int, out_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float:
    p = PRICES[model]
    # Cache-read tokens are charged at 0.1x input rate; cache-write at 1.25x.
    # in_tokens excludes cache_read in Anthropic accounting, but we be explicit.
    base_in = in_tokens * p["in"] / 1_000_000
    cr_cost = cache_read * p["in"] * 0.10 / 1_000_000
    cw_cost = cache_write * p["in"] * 1.25 / 1_000_000
    out_cost = out_tokens * p["out"] / 1_000_000
    return base_in + cr_cost + cw_cost + out_cost


# ── Usage capture ─────────────────────────────────────────────────────────

@dataclass
class CallRecord:
    label: str
    model: str
    in_tokens: int = 0
    out_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @property
    def cost(self) -> float:
        return usd_cost(self.model, self.in_tokens, self.out_tokens, self.cache_read, self.cache_write)


@dataclass
class Ledger:
    calls: list[CallRecord] = field(default_factory=list)

    def add(self, rec: CallRecord) -> None:
        self.calls.append(rec)

    def total(self) -> float:
        return sum(c.cost for c in self.calls)


LEDGER = Ledger()


def _patch_anthropic() -> None:
    """Wrap _anthropic.messages.create to record token usage."""
    from app import wsdc

    original = wsdc._anthropic.messages.create

    def wrapped(**kwargs: Any) -> Any:
        resp = original(**kwargs)
        usage = resp.usage
        model = kwargs.get("model", "unknown")
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        # Label: figure out if this is Pass 1 or Pass 2 by max_tokens (Pass 1 is 2048, Pass 2 is 4096).
        label = "Anthropic Pass 1 (structure)" if kwargs.get("max_tokens") == 2048 else "Anthropic Pass 2 (render)"
        LEDGER.add(CallRecord(
            label=label,
            model=model,
            in_tokens=in_tok,
            out_tokens=out_tok,
            cache_read=cache_read,
            cache_write=cache_write,
        ))
        return resp

    wsdc._anthropic.messages.create = wrapped


def _patch_openai() -> None:
    """Wrap the OpenAI chat completions create call to record token usage."""
    from app import main as app_main

    original = app_main.client.chat.completions.create

    def wrapped(**kwargs: Any) -> Any:
        resp = original(**kwargs)
        usage = resp.usage
        model = kwargs.get("model", "unknown")
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        # Heuristic label: scoring uses response_format=json_object on gpt-4o.
        if kwargs.get("response_format") and model.startswith("gpt-4o"):
            label = f"OpenAI scoring ({model})"
        else:
            label = f"OpenAI generation ({model})"
        LEDGER.add(CallRecord(
            label=label,
            model=model,
            in_tokens=in_tok,
            out_tokens=out_tok,
        ))
        return resp

    app_main.client.chat.completions.create = wrapped


# ── Test debate runner ────────────────────────────────────────────────────

def run_debate() -> None:
    """Execute one WSDC intermediate debate: opening AI turn → user rebuttal
    → AI rebuttal → user closing → scoring. Mirrors what a real SuperJuniors
    student would trigger."""
    from app import main as app_main

    # Build an internal Debate + messages exactly like the api_v1 adapter does.
    from datetime import datetime
    from uuid import uuid4

    debate = app_main.Debate(
        id=uuid4(),
        title="THW ban single-use plastics globally",
        num_rounds=2,
        starter="assistant",   # AI opens, student rebuts — full Pass 1 + Pass 2 path
        current_round=1,
        next_speaker="assistant",
        status="active",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        mode="wsdc",
        difficulty="intermediate",
    )

    print("\n========== TURN 1: AI opening speech (Sonnet Pass 1 + Haiku Pass 2) ==========")
    ai_opening = app_main.generate_ai_turn_text(debate, [])
    print(f"[generated {len(ai_opening)} chars]")

    messages = [
        app_main.Message(
            id=uuid4(), debate_id=debate.id, round_no=1, speaker="assistant",
            content=ai_opening, created_at=datetime.utcnow(),
        ),
        app_main.Message(
            id=uuid4(), debate_id=debate.id, round_no=1, speaker="user",
            content=(
                "The proposition's case rests on the assumption that a global plastic ban is enforceable, "
                "but that misreads the realities of developing economies. In Vietnam, where I live, "
                "informal plastic supply chains employ millions; an overnight ban would devastate "
                "livelihoods before any environmental benefit materialised. Worse, the alternatives — "
                "bioplastics, glass, paper — have hidden carbon costs that often exceed the harm of "
                "well-managed plastic recycling. The solution is investment in waste infrastructure, "
                "not prohibition."
            ),
            created_at=datetime.utcnow(),
        ),
    ]
    debate.current_round = 2
    debate.next_speaker = "assistant"

    print("\n========== TURN 2: AI rebuttal speech (Sonnet Pass 1 + Haiku Pass 2) ==========")
    ai_rebuttal = app_main.generate_ai_turn_text(debate, messages)
    print(f"[generated {len(ai_rebuttal)} chars]")
    messages.append(app_main.Message(
        id=uuid4(), debate_id=debate.id, round_no=2, speaker="assistant",
        content=ai_rebuttal, created_at=datetime.utcnow(),
    ))
    messages.append(app_main.Message(
        id=uuid4(), debate_id=debate.id, round_no=2, speaker="user",
        content=(
            "Even granting the proposition's framing, they still owe us a mechanism. None of their "
            "arguments engage with the displacement problem I raised. They simply assert that the "
            "harm of plastics outweighs the harm of unemployment — without showing the timeframe or "
            "probability. On our side, the harm is immediate and concentrated; on theirs, it is "
            "diffuse and contingent. We win this debate on weighing alone."
        ),
        created_at=datetime.utcnow(),
    ))

    print("\n========== SCORING: GPT-4o judges the human ==========")
    score = app_main.compute_debate_score(debate, messages)
    print(f"[overall={score.overall}, weakness={score.weakness_type}]")


# ── Reporting ─────────────────────────────────────────────────────────────

def print_report() -> None:
    print("\n" + "=" * 72)
    print("PER-CALL BREAKDOWN")
    print("=" * 72)
    print(f"{'Call':<42} {'Model':<20} {'In':>7} {'Out':>7} {'Cost':>9}")
    print("-" * 72)
    for c in LEDGER.calls:
        in_str = f"{c.in_tokens + c.cache_read + c.cache_write}"
        print(f"{c.label:<42} {c.model:<20} {in_str:>7} {c.out_tokens:>7} ${c.cost:>7.4f}")
        if c.cache_read or c.cache_write:
            print(f"  └─ cache_read={c.cache_read}, cache_write={c.cache_write}")

    total = LEDGER.total()
    n_turns = sum(1 for c in LEDGER.calls if "Pass 1" in c.label)
    n_scoring = sum(1 for c in LEDGER.calls if "scoring" in c.label)

    print("-" * 72)
    print(f"TOTAL: ${total:.4f}")
    print(f"  AI turns: {n_turns}  →  ~${(total - sum(c.cost for c in LEDGER.calls if 'scoring' in c.label)) / max(n_turns, 1):.4f}/turn (avg)")
    if n_scoring:
        scoring_cost = sum(c.cost for c in LEDGER.calls if "scoring" in c.label)
        print(f"  Scoring:  {n_scoring}  →  ${scoring_cost:.4f}")

    if n_turns:
        per_turn = (total - sum(c.cost for c in LEDGER.calls if 'scoring' in c.label)) / n_turns
        scoring = sum(c.cost for c in LEDGER.calls if 'scoring' in c.label)
        # A typical SuperJuniors debate = ~3 AI turns + 1 scoring.
        typical_debate = per_turn * 3 + scoring
        print(f"\nPROJECTED 'typical debate' (3 AI turns + 1 scoring): ${typical_debate:.4f}")
        for monthly_debates in (100, 500, 1000, 5000):
            print(f"  At {monthly_debates:>5} debates/mo: ${typical_debate * monthly_debates:>7.2f}/mo")


def main() -> None:
    _patch_anthropic()
    _patch_openai()

    try:
        run_debate()
    except Exception:
        print("\n[!] Debate run failed mid-way. Printing whatever was captured.")
        traceback.print_exc()

    print_report()


if __name__ == "__main__":
    main()
