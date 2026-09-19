"""Versioned vendor rate card and cost math.

THIS FILE IS THE INVOICE. SuperJuniors are billed at cost pass-through, so the
numbers below are not estimates — they are what we charge. A stale rate here
produces a provably wrong bill.

Refreshing the rate card
------------------------
1. Check Anthropic's and OpenAI's public pricing pages.
2. If anything changed, add a NEW entry to RATE_CARDS with today's date as the
   version. Never edit a past version — historical cost_events rows reference
   it by `price_version` and must stay reproducible.
3. Set CURRENT_VERSION to the new key.
4. Run: python scripts/verify_pricing.py

Adding a model
--------------
A model missing from the active card raises UnknownModel at call time. Cost
capture is wrapped so this never breaks a student's debate, but the event is
recorded with cost 0.0 and `needs_repricing=True` so it can be corrected and
re-billed later. Grep the logs for [pricing] after any model change.
"""

from __future__ import annotations

from typing import Any

# Anthropic cache multipliers, applied against the model's base input rate.
CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25


class UnknownModel(Exception):
    """Raised when a model has no entry in the active rate card."""


# USD per 1,000,000 tokens. Per-image/per-minute units noted inline.
#
# Each version is immutable once cost_events rows reference it. Add, never edit.
RATE_CARDS: dict[str, dict[str, dict[str, float]]] = {
    "2026-06-07": {
        # Initial card, matching scripts/measure_turn_cost.py as measured.
        "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
        "claude-haiku-4-5":  {"in": 1.00, "out":  5.00},
        "gpt-4o":            {"in": 2.50, "out": 10.00},
        "gpt-4o-mini":       {"in": 0.15, "out":  0.60},
    },
    "2026-09-19": {
        "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
        "claude-haiku-4-5":  {"in": 1.00, "out":  5.00},
        "gpt-4o":            {"in": 2.50, "out": 10.00},
        "gpt-4o-mini":       {"in": 0.15, "out":  0.60},
        # Moderation is free at time of writing. Kept explicit so a future
        # charge is a one-line edit rather than a silently unbilled call.
        "omni-moderation-latest": {"in": 0.00, "out": 0.00},
        # Whisper bills per minute of audio, not per token. Website-only today
        # (the partner API has no audio endpoint) — see app/transcriber_processor.py.
        "whisper-1": {"per_minute": 0.006},
    },
}

CURRENT_VERSION = "2026-09-19"


def active_card(version: str | None = None) -> dict[str, dict[str, float]]:
    v = version or CURRENT_VERSION
    try:
        return RATE_CARDS[v]
    except KeyError as exc:
        raise UnknownModel(f"No rate card for version {v!r}") from exc


def token_cost_usd(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    version: str | None = None,
) -> float:
    """Vendor cost in USD for one token-billed model call.

    Anthropic reports cache_read/cache_write separately from input_tokens, so
    the three are summed rather than overlapping. OpenAI reports neither, and
    passing 0 for both degrades to a plain input/output calculation.
    """
    card = active_card(version)
    p = card.get(model)
    if p is None:
        raise UnknownModel(f"{model!r} is not in rate card {version or CURRENT_VERSION!r}")
    if "in" not in p:
        raise UnknownModel(f"{model!r} is not token-billed in this card (got keys {sorted(p)})")

    return (
        input_tokens * p["in"] / 1_000_000
        + cache_read_tokens * p["in"] * CACHE_READ_MULTIPLIER / 1_000_000
        + cache_write_tokens * p["in"] * CACHE_WRITE_MULTIPLIER / 1_000_000
        + output_tokens * p["out"] / 1_000_000
    )


def audio_cost_usd(model: str, seconds: float, version: str | None = None) -> float:
    """Vendor cost for a duration-billed model (Whisper)."""
    card = active_card(version)
    p = card.get(model)
    if p is None or "per_minute" not in p:
        raise UnknownModel(f"{model!r} is not duration-billed in rate card {version or CURRENT_VERSION!r}")
    return (seconds / 60.0) * p["per_minute"]


def extract_anthropic_usage(resp: Any) -> dict[str, int]:
    """Pull token counts off an Anthropic Messages response.

    getattr-with-default throughout: a missing cache field on an older SDK
    should read as zero, not raise inside a student's debate.
    """
    u = getattr(resp, "usage", None)
    if u is None:
        return {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}
    return {
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
        "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_write_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }


def extract_openai_usage(resp: Any) -> dict[str, int]:
    """Pull token counts off an OpenAI chat-completions response.

    OpenAI reports cached input under prompt_tokens_details.cached_tokens and
    INCLUDES it in prompt_tokens, so it is subtracted out to avoid double
    counting against the full input rate.
    """
    u = getattr(resp, "usage", None)
    if u is None:
        return {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}
    prompt = getattr(u, "prompt_tokens", 0) or 0
    details = getattr(u, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    return {
        "input_tokens": max(0, prompt - cached),
        "output_tokens": getattr(u, "completion_tokens", 0) or 0,
        "cache_read_tokens": cached,
        "cache_write_tokens": 0,
    }
