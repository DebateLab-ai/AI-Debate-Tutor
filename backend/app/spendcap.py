"""Per-tenant monthly spend cap and threshold alerting.

Postpaid billing has no prepaid balance to run out, so this is the only thing
standing between a bug on a partner's side and an unbounded Anthropic bill.

Two separate mechanisms:

  * enforce_spend_cap()  — HARD STOP. Raises 402 before any model call once
    month-to-date vendor cost reaches tenants.monthly_cost_limit_usd. Called at
    the top of every cost-incurring endpoint. This one MAY raise; that is the
    point.

  * check_thresholds_and_alert() — SOFT WARNING. Emails at 50/80/100% of the
    cap, once each per tenant per month. Runs as a background task after the
    response, and never raises.

A tenant with no limit set (NULL) is uncapped, which preserves existing
behaviour for anyone we haven't deliberately capped.

Month boundaries are UTC, matching cost_events.created_at.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, status

from app import alerts
from app.db import get_client

# Percentages of the cap at which to email. Override with
# SPEND_ALERT_THRESHOLDS="50,80,100".
DEFAULT_THRESHOLDS = [50, 80, 100]

# Cap lookups happen on every cost-incurring request. Cache the limit briefly so
# a burst of turns does not become a burst of tenant lookups. The MTD cost is
# NOT cached — that has to be current or the cap leaks spend.
_LIMIT_CACHE_SECONDS = 60
_limit_cache: dict[str, tuple[float, Optional[float]]] = {}


def _thresholds() -> list[int]:
    raw = os.getenv("SPEND_ALERT_THRESHOLDS", "")
    if not raw.strip():
        return list(DEFAULT_THRESHOLDS)
    try:
        return sorted({int(x) for x in raw.split(",") if x.strip()})
    except ValueError:
        print(f"[spendcap] bad SPEND_ALERT_THRESHOLDS {raw!r}; using {DEFAULT_THRESHOLDS}")
        return list(DEFAULT_THRESHOLDS)


def month_start(now: Optional[datetime] = None) -> datetime:
    n = now or datetime.now(timezone.utc)
    return n.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def period_label(now: Optional[datetime] = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m")


def month_to_date_cost(tenant_id: str) -> float:
    """Vendor spend for this tenant since the 1st of the month, in USD.

    Prefers the tenant_mtd_cost_usd() SQL function (migration 003). Falls back
    to paging rows into Python if it is absent, so a missing migration degrades
    to slow rather than broken.
    """
    since = month_start()
    db = get_client()
    try:
        res = db.rpc("tenant_mtd_cost_usd", {
            "p_tenant_id": tenant_id,
            "p_since": since.isoformat(),
        }).execute()
        if res.data is not None:
            return float(res.data)
    except Exception as e:
        print(f"[spendcap] tenant_mtd_cost_usd RPC unavailable ({e}); summing in Python")

    total = 0.0
    offset = 0
    page = 1000
    while True:
        rows = (
            db.table("cost_events")
            .select("vendor_cost_usd")
            .eq("tenant_id", tenant_id)
            .gte("created_at", since.isoformat())
            .range(offset, offset + page - 1)
            .execute()
            .data
            or []
        )
        total += sum(float(r["vendor_cost_usd"] or 0) for r in rows)
        if len(rows) < page:
            return total
        offset += page


def get_limit(tenant_id: str, *, use_cache: bool = True) -> Optional[float]:
    """The tenant's monthly cap in USD, or None if uncapped."""
    import time
    now = time.monotonic()
    if use_cache:
        hit = _limit_cache.get(tenant_id)
        if hit and now - hit[0] < _LIMIT_CACHE_SECONDS:
            return hit[1]
    try:
        res = (
            get_client()
            .table("tenants")
            .select("monthly_cost_limit_usd")
            .eq("id", tenant_id)
            .maybe_single()
            .execute()
        )
        row = res.data if res else None
        raw = (row or {}).get("monthly_cost_limit_usd")
        limit = float(raw) if raw is not None else None
    except Exception as e:
        # Fail OPEN on a lookup error. A database hiccup must not lock a partner
        # out of a lesson; the alert path still reports the real spend.
        print(f"[spendcap] limit lookup failed for {tenant_id} ({e}); treating as uncapped")
        return None
    _limit_cache[tenant_id] = (now, limit)
    return limit


def invalidate_limit_cache(tenant_id: Optional[str] = None) -> None:
    """Drop cached limits so an admin change takes effect immediately."""
    if tenant_id:
        _limit_cache.pop(tenant_id, None)
    else:
        _limit_cache.clear()


def enforce_spend_cap(tenant_id: str) -> None:
    """Raise 402 if this tenant has reached its monthly cap.

    Call BEFORE any model call. This is the hard stop; it is allowed to raise.
    """
    limit = get_limit(tenant_id)
    if limit is None:
        return
    try:
        spend = month_to_date_cost(tenant_id)
    except Exception as e:
        # Fail open for the same reason as get_limit.
        print(f"[spendcap] MTD lookup failed for {tenant_id} ({e}); allowing request")
        return
    if spend >= limit:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"Monthly spend cap reached: ${spend:.2f} of ${limit:.2f} used "
                f"for {period_label()}. Contact DebateLab to raise the limit."
            ),
        )


def check_thresholds_and_alert(tenant_id: str, tenant_name: Optional[str] = None) -> None:
    """Email once per crossed threshold per month. Background task; never raises."""
    try:
        limit = get_limit(tenant_id, use_cache=False)
        if limit is None or limit <= 0:
            return
        spend = month_to_date_cost(tenant_id)
        pct = (spend / limit) * 100
        period = period_label()

        crossed = [t for t in _thresholds() if pct >= t]
        if not crossed:
            return

        db = get_client()
        for threshold in crossed:
            # The primary key on cost_alerts_sent is the de-duplication: if this
            # insert succeeds we are the first to cross it this month, so we
            # send. If it conflicts, someone already did.
            try:
                db.table("cost_alerts_sent").insert({
                    "tenant_id": tenant_id,
                    "period": period,
                    "threshold_pct": threshold,
                    "mtd_cost_usd": round(spend, 8),
                }).execute()
            except Exception:
                continue  # already sent this month

            name = tenant_name or tenant_id
            capped = "SPENDING HALTED" if pct >= 100 else "warning"
            alerts.send_alert(
                # Name the threshold crossed, not just current usage: when two
                # fire on the same request, identical subjects are unreadable.
                subject=(
                    f"[DebateLab] {name} crossed {threshold}% of monthly spend cap "
                    f"— now at {pct:.0f}% ({capped})"
                ),
                body=(
                    f"Tenant:      {name}\n"
                    f"Tenant ID:   {tenant_id}\n"
                    f"Period:      {period} (UTC)\n"
                    f"Spend:       ${spend:.4f}\n"
                    f"Cap:         ${limit:.2f}\n"
                    f"Used:        {pct:.1f}%\n"
                    f"Threshold:   {threshold}%\n\n"
                    + (
                        "The cap has been reached. Further cost-incurring API calls from\n"
                        "this tenant are being rejected with HTTP 402 until the cap is\n"
                        "raised or the month rolls over.\n"
                        if pct >= 100 else
                        "This is a warning only. The tenant is still being served.\n"
                    )
                    + "\nRaise the cap:\n"
                    f"  PUT /internal/admin/tenants/{tenant_id}/cap  "
                    '{"monthly_cost_limit_usd": <new value or null>}\n'
                ),
            )
    except Exception as e:
        print(f"[spendcap] threshold check failed for {tenant_id}: {type(e).__name__}: {e}")
