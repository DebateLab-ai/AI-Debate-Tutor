"""Billing ledger: attribute vendor model spend to a tenant and record it.

Two jobs:

1. **Attribution.** Generation functions (wsdc.py, response.py, compute_debate_score)
   are shared by the website and the partner API and have no idea who they are
   working for. Threading tenant_id through every signature would touch a lot of
   call sites, so the partner API sets a ContextVar at the edge instead and the
   recorders read it. The website never sets it, so its rows land with
   tenant_id=None — which is what we want, not a bug.

2. **Recording.** record_* writes one cost_events row per vendor call that we
   actually delivered.

   Partner charges are held on the request until the HTTP response is a success.
   The usage middleware commits them on 2xx and drops them otherwise, so a 4xx
   or 5xx does not add to the invoice or the spend cap. A model attempt that
   we reject and replace (Haiku JSON we cannot parse, a two-pass attempt that
   falls back) is dropped the same way, even when the request itself succeeds.
   The website has no tenant, so those rows are inserted immediately. They are
   not on a partner invoice.

NOTHING IN HERE MAY RAISE. A failure to bill must never break a student's
debate — a dropped row costs us money, a raised exception costs a lesson. Every
public function swallows and logs. That asymmetry is deliberate.

ContextVars are task-local under asyncio and thread-local under a threadpool, so
FastAPI's sync endpoints (which run in a worker thread) get correct isolation
either way. The value is reset per request by the dependency that sets it.

set_debate replaces the context object inside that worker thread. The pending
list has to be the same object on both, or charges recorded after set_debate
would be invisible to the middleware that commits them.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from app import pricing
from app.db import get_client


@dataclass
class BillingContext:
    tenant_id: Optional[str] = None
    debate_id: Optional[str] = None
    # One id for every model call a single HTTP request produces, so the
    # partner usage feed can roll a speech plus its safety check into one line.
    # None on the website, and on partner calls from before this was stamped.
    request_id: Optional[str] = None
    api_key_id: Optional[str] = None
    # Stamped on cost rows and the usage row so a request that crosses midnight
    # UTC cannot put the charge in one month and the line in the next.
    started_at: Optional[datetime] = None
    pending: list[dict[str, Any]] = field(default_factory=list)


_ctx: contextvars.ContextVar[BillingContext] = contextvars.ContextVar(
    "billing_ctx", default=BillingContext()
)


def set_context(
    tenant_id: Optional[str],
    debate_id: Optional[str] = None,
    request_id: Optional[str] = None,
    api_key_id: Optional[str] = None,
    started_at: Optional[datetime] = None,
) -> contextvars.Token:
    """Bind the current request to a tenant. Returns a token for reset_context."""
    if started_at is None and tenant_id:
        started_at = datetime.now(timezone.utc)
    return _ctx.set(BillingContext(
        tenant_id=tenant_id,
        debate_id=debate_id,
        request_id=request_id,
        api_key_id=api_key_id,
        started_at=started_at,
    ))


def set_debate(debate_id: Optional[str]) -> None:
    """Attach a debate to the already-bound tenant, once it is known.

    Copies request_id and the pending list forward. Replacing the whole
    context without them would split one HTTP call across two usage lines,
    and would hide later charges from the middleware.
    """
    cur = _ctx.get()
    _ctx.set(BillingContext(
        tenant_id=cur.tenant_id,
        debate_id=debate_id,
        request_id=cur.request_id,
        api_key_id=cur.api_key_id,
        started_at=cur.started_at,
        pending=cur.pending,
    ))


def reset_context(token: contextvars.Token) -> None:
    try:
        _ctx.reset(token)
    except ValueError:
        # Token from a different context (shouldn't happen, but resetting is
        # best-effort cleanup and must not raise).
        pass


def current() -> BillingContext:
    return _ctx.get()


def checkpoint() -> int:
    """Mark the pending list so a failed attempt can be rolled back."""
    try:
        return len(_ctx.get().pending)
    except Exception:
        return 0


def rollback(mark: int) -> None:
    """Drop partner charges recorded after `mark`. Never raises."""
    try:
        pending = _ctx.get().pending
        if mark < len(pending):
            del pending[mark:]
    except Exception as e:
        print(f"[billing] rollback failed, ignoring: {e}")


def discard_last() -> None:
    """Drop the most recent held partner charge. Never raises.

    Used when one model attempt is rejected and a fallback takes its place.
    Website rows are inserted immediately and are not on this list.
    """
    try:
        pending = _ctx.get().pending
        if pending:
            pending.pop()
    except Exception as e:
        print(f"[billing] discard_last failed, ignoring: {e}")


def commit() -> None:
    """Write held partner charges. Call this only when the HTTP request succeeded."""
    try:
        pending = _ctx.get().pending
        rows = list(pending)
        pending.clear()
        for row in rows:
            _insert(row)
    except Exception as e:
        print(f"[billing] commit failed, ignoring: {e}")


def discard() -> None:
    """Drop held partner charges. A failed HTTP request bills nothing."""
    try:
        _ctx.get().pending.clear()
    except Exception as e:
        print(f"[billing] discard failed, ignoring: {e}")


def _with_request_id(row: dict[str, Any]) -> dict[str, Any]:
    request_id = _ctx.get().request_id
    if request_id and "request_id" not in row:
        return {**row, "request_id": request_id}
    return row


def _insert(row: dict[str, Any]) -> None:
    """Write one cost_events row. Never raises.

    If the request_id column is not migrated yet, retry without it. A missing
    column must not drop the charge — the month total still has to be exact,
    and the per-call line can wait.
    """
    row = _with_request_id(row)
    try:
        get_client().table("cost_events").insert(row).execute()
    except Exception as e:
        if row.get("request_id") and "request_id" in str(e).lower():
            slim = {k: v for k, v in row.items() if k != "request_id"}
            try:
                get_client().table("cost_events").insert(slim).execute()
                return
            except Exception as e2:
                print(f"[billing] FAILED to record cost event ({row.get('model')}): {e2}")
                return
        # Log loudly — this is lost revenue, but still not worth an exception.
        print(f"[billing] FAILED to record cost event ({row.get('model')}): {e}")


def _hold_or_insert(row: dict[str, Any]) -> bool:
    """Partner rows wait for commit. Website rows are inserted now.

    Returns whether a partner row was held or a website row was inserted.
    """
    ctx = _ctx.get()
    if ctx.started_at is not None and "created_at" not in row:
        row["created_at"] = ctx.started_at.isoformat()
    if ctx.request_id and "request_id" not in row:
        row["request_id"] = ctx.request_id
    if ctx.tenant_id:
        ctx.pending.append(row)
        return True
    _insert(row)
    return True


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
) -> bool:
    """Record one token-billed vendor call against the current tenant.

    Returns True when a row was held or inserted. Callers that reject this
    attempt (bad JSON, a fallback) use that to drop just this charge.
    """
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

        row = {
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
        }
        return _hold_or_insert(row)
    except Exception as e:
        print(f"[billing] record_model_call crashed, ignoring: {e}")
        return False


def record_anthropic(resp: Any, *, model: str, event_type: str, **kw: Any) -> bool:
    try:
        usage = pricing.extract_anthropic_usage(resp)
    except Exception as e:
        print(f"[billing] could not read Anthropic usage: {e}")
        return False
    return record_model_call(provider="anthropic", model=model, event_type=event_type, usage=usage, **kw)


def record_openai(resp: Any, *, model: str, event_type: str, **kw: Any) -> bool:
    try:
        usage = pricing.extract_openai_usage(resp)
    except Exception as e:
        print(f"[billing] could not read OpenAI usage: {e}")
        return False
    return record_model_call(provider="openai", model=model, event_type=event_type, usage=usage, **kw)


def record_audio(*, provider: str, model: str, seconds: float, event_type: str = "transcription") -> bool:
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
        row = {
            "tenant_id": ctx.tenant_id,
            "debate_id": ctx.debate_id,
            "event_type": event_type,
            "provider": provider,
            "model": model,
            "audio_seconds": round(seconds, 2),
            "vendor_cost_usd": round(cost, 8),
            "price_version": pricing.CURRENT_VERSION,
            "needs_repricing": needs_repricing,
        }
        return _hold_or_insert(row)
    except Exception as e:
        print(f"[billing] record_audio crashed, ignoring: {e}")
        return False
