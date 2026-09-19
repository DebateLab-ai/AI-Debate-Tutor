"""Billing ledger: attribute vendor model spend to a tenant and record it.

Two jobs:

1. **Attribution.** Generation functions (wsdc.py, response.py, compute_debate_score)
   are shared by the website and the partner API and have no idea who they are
   working for. Threading tenant_id through every signature would touch a lot of
   call sites, so the partner API sets a ContextVar at the edge instead and the
   recorders read it. The website never sets it, so its rows land with
   tenant_id=None — which is what we want, not a bug.

2. **Recording.** record_* writes one cost_events row per vendor call.

NOTHING IN HERE MAY RAISE. A failure to bill must never break a student's
debate — a dropped row costs us money, a raised exception costs a lesson. Every
public function swallows and logs. That asymmetry is deliberate.

ContextVars are task-local under asyncio and thread-local under a threadpool, so
FastAPI's sync endpoints (which run in a worker thread) get correct isolation
either way. The value is reset per request by the dependency that sets it.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any, Optional

from app import pricing
from app.db import get_client


@dataclass
class BillingContext:
    tenant_id: Optional[str] = None
    debate_id: Optional[str] = None


_ctx: contextvars.ContextVar[BillingContext] = contextvars.ContextVar(
    "billing_ctx", default=BillingContext()
)


def set_context(tenant_id: Optional[str], debate_id: Optional[str] = None) -> contextvars.Token:
    """Bind the current request to a tenant. Returns a token for reset_context."""
    return _ctx.set(BillingContext(tenant_id=tenant_id, debate_id=debate_id))


def set_debate(debate_id: Optional[str]) -> None:
    """Attach a debate to the already-bound tenant, once it is known."""
    cur = _ctx.get()
    _ctx.set(BillingContext(tenant_id=cur.tenant_id, debate_id=debate_id))


def reset_context(token: contextvars.Token) -> None:
    try:
        _ctx.reset(token)
    except ValueError:
        # Token from a different context (shouldn't happen, but resetting is
        # best-effort cleanup and must not raise).
        pass


def current() -> BillingContext:
    return _ctx.get()


def _insert(row: dict[str, Any]) -> None:
    try:
        get_client().table("cost_events").insert(row).execute()
    except Exception as e:
        # Log loudly — this is lost revenue, but still not worth an exception.
        print(f"[billing] FAILED to record cost event ({row.get('model')}): {e}")


def record_model_call(
    *,
    provider: str,
    model: str,
    event_type: str,
    usage: dict[str, int],
    succeeded: bool = True,
    billable: bool = True,
    debate_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> None:
    """Record one token-billed vendor call against the current tenant."""
    try:
        ctx = _ctx.get()
        needs_repricing = False
        try:
            cost = pricing.token_cost_usd(model=model, **usage)
        except pricing.UnknownModel as e:
            # Unknown model: still record the tokens so the call can be repriced
            # and re-billed once the rate card is updated. Never drop the row.
            print(f"[pricing] {e} — recording with cost 0.0 and needs_repricing=True")
            cost = 0.0
            needs_repricing = True

        _insert({
            "tenant_id": ctx.tenant_id,
            "debate_id": debate_id or ctx.debate_id,
            "message_id": message_id,
            "event_type": event_type,
            "provider": provider,
            "model": model,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_read_tokens": usage.get("cache_read_tokens", 0),
            "cache_write_tokens": usage.get("cache_write_tokens", 0),
            "vendor_cost_usd": round(cost, 8),
            "price_version": pricing.CURRENT_VERSION,
            "billable": billable,
            "needs_repricing": needs_repricing,
            "succeeded": succeeded,
        })
    except Exception as e:
        print(f"[billing] record_model_call crashed, ignoring: {e}")


def record_anthropic(resp: Any, *, model: str, event_type: str, **kw: Any) -> None:
    try:
        usage = pricing.extract_anthropic_usage(resp)
    except Exception as e:
        print(f"[billing] could not read Anthropic usage: {e}")
        return
    record_model_call(provider="anthropic", model=model, event_type=event_type, usage=usage, **kw)


def record_openai(resp: Any, *, model: str, event_type: str, **kw: Any) -> None:
    try:
        usage = pricing.extract_openai_usage(resp)
    except Exception as e:
        print(f"[billing] could not read OpenAI usage: {e}")
        return
    record_model_call(provider="openai", model=model, event_type=event_type, usage=usage, **kw)


def record_audio(*, provider: str, model: str, seconds: float, event_type: str = "transcription") -> None:
    """Record a duration-billed call (Whisper)."""
    try:
        ctx = _ctx.get()
        needs_repricing = False
        try:
            cost = pricing.audio_cost_usd(model, seconds)
        except pricing.UnknownModel as e:
            print(f"[pricing] {e} — recording with cost 0.0 and needs_repricing=True")
            cost = 0.0
            needs_repricing = True
        _insert({
            "tenant_id": ctx.tenant_id,
            "debate_id": ctx.debate_id,
            "event_type": event_type,
            "provider": provider,
            "model": model,
            "audio_seconds": round(seconds, 2),
            "vendor_cost_usd": round(cost, 8),
            "price_version": pricing.CURRENT_VERSION,
            "needs_repricing": needs_repricing,
        })
    except Exception as e:
        print(f"[billing] record_audio crashed, ignoring: {e}")
