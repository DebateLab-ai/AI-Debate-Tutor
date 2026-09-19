"""Private internal API under /internal/admin/*.

NOT part of the partner surface. Every route requires X-Admin-Token (see
app/admin_auth.py), reads across all tenants, and is excluded from the OpenAPI
schema so it never shows up in /docs for a partner who goes looking.

Read-only by design: nothing here mutates a row. If we ever add tenant/key
provisioning it belongs in a separate router with its own review, not bolted
onto the usage viewer.

These routes must never be documented in docs/ — that directory is partner-facing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from app import alerts, spendcap, usage_store
from app.db import get_client
from app.admin_auth import require_admin

router = APIRouter(
    prefix="/internal/admin",
    tags=["internal-admin"],
    dependencies=[Depends(require_admin)],
    include_in_schema=False,
)


def _guard_db(exc: Exception) -> HTTPException:
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=503, detail="Partner API database is not configured")
    return HTTPException(status_code=503, detail="Usage store is temporarily unavailable")


@router.get("/usage")
def get_usage(
    tenant_id: Optional[str] = Query(default=None, description="Filter to one tenant"),
    api_key_id: Optional[str] = Query(default=None, description="Filter to one API key"),
    endpoint: Optional[str] = Query(default=None, description='Exact match, e.g. "POST /api/v1/debates"'),
    status_min: Optional[int] = Query(default=None, ge=100, le=599),
    status_max: Optional[int] = Query(default=None, ge=100, le=599),
    errors_only: bool = Query(default=False, description="Shorthand for status_min=400"),
    since: Optional[datetime] = Query(default=None, description="ISO-8601, inclusive"),
    until: Optional[datetime] = Query(default=None, description="ISO-8601, exclusive"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Raw usage log lines, newest first."""
    if errors_only:
        status_min = max(status_min or 0, 400)
    if status_min is not None and status_max is not None and status_min > status_max:
        raise HTTPException(status_code=400, detail="status_min must be <= status_max")
    if since is not None and until is not None and since > until:
        raise HTTPException(status_code=400, detail="`since` must be earlier than `until`")

    try:
        rows = usage_store.list_usage(
            tenant_id=tenant_id,
            api_key_id=api_key_id,
            endpoint=endpoint,
            status_min=status_min,
            status_max=status_max,
            since=since,
            until=until,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        raise _guard_db(exc) from exc

    # next_offset is None on the last page so a caller can page without guessing.
    return {
        "count": len(rows),
        "limit": limit,
        "offset": offset,
        "next_offset": offset + limit if len(rows) == limit else None,
        "rows": rows,
    }


@router.get("/usage/summary")
def get_usage_summary(
    tenant_id: Optional[str] = Query(default=None, description="Filter to one tenant"),
    since: Optional[datetime] = Query(default=None, description="ISO-8601, inclusive; defaults to `days` ago"),
    until: Optional[datetime] = Query(default=None, description="ISO-8601, exclusive; defaults to now"),
    days: int = Query(default=30, ge=1, le=usage_store.SUMMARY_MAX_DAYS, description="Window size when `since` is omitted"),
) -> dict[str, Any]:
    """Request counts, error rates and latency percentiles by tenant, endpoint and day."""
    try:
        return usage_store.summarize_usage(
            tenant_id=tenant_id, since=since, until=until, default_days=days
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise _guard_db(exc) from exc


@router.get("/tenants")
def get_tenants() -> dict[str, Any]:
    """Tenants and their key counts — resolves the IDs used by the filters above.

    Key hashes are never selected, so there is no path from here to a usable key.
    """
    try:
        tenants = usage_store.list_tenants()
    except Exception as exc:
        raise _guard_db(exc) from exc
    return {"count": len(tenants), "tenants": tenants}


@router.get("/spend")
def get_spend() -> dict[str, Any]:
    """Month-to-date vendor spend per tenant, against each tenant's cap."""
    try:
        tenants = usage_store.list_tenants()
        out = []
        for t in tenants:
            tid = str(t["id"])
            spend = spendcap.month_to_date_cost(tid)
            limit = spendcap.get_limit(tid, use_cache=False)
            out.append({
                "tenant_id": tid,
                "name": t["name"],
                "mtd_cost_usd": round(spend, 6),
                "monthly_cost_limit_usd": limit,
                "pct_used": round(spend / limit * 100, 1) if limit else None,
                "capped": limit is not None and spend >= limit,
            })
    except Exception as exc:
        raise _guard_db(exc) from exc
    return {
        "period": spendcap.period_label(),
        "since": spendcap.month_start().isoformat(),
        "tenants": sorted(out, key=lambda e: -e["mtd_cost_usd"]),
    }


@router.put("/tenants/{tenant_id}/cap")
def set_tenant_cap(
    tenant_id: str,
    monthly_cost_limit_usd: Optional[float] = Body(
        default=None, embed=True,
        description="USD ceiling for month-to-date vendor spend. null removes the cap.",
    ),
) -> dict[str, Any]:
    """Set or clear a tenant's monthly spend cap."""
    if monthly_cost_limit_usd is not None and monthly_cost_limit_usd < 0:
        raise HTTPException(status_code=400, detail="monthly_cost_limit_usd must be >= 0")
    try:
        res = (
            get_client()
            .table("tenants")
            .update({"monthly_cost_limit_usd": monthly_cost_limit_usd})
            .eq("id", tenant_id)
            .execute()
        )
        if not res.data:
            raise HTTPException(status_code=404, detail="Tenant not found")
    except HTTPException:
        raise
    except Exception as exc:
        raise _guard_db(exc) from exc

    # The limit is cached for 60s in the request path; drop it so a change made
    # here takes effect on the very next call rather than up to a minute later.
    spendcap.invalidate_limit_cache(tenant_id)
    return {
        "tenant_id": tenant_id,
        "monthly_cost_limit_usd": monthly_cost_limit_usd,
        "mtd_cost_usd": round(spendcap.month_to_date_cost(tenant_id), 6),
    }


@router.get("/alerts/config")
def get_alert_config() -> dict[str, Any]:
    """Who gets alerted and whether email can actually be sent right now."""
    return {
        "recipients": alerts.recipients(),
        "smtp_configured": alerts.is_configured(),
        "thresholds_pct": spendcap._thresholds(),
        "how_to_add": (
            "Set ALERT_EMAILS='a@x.com, b@y.com' on the host, or edit "
            "DEFAULT_ALERT_RECIPIENTS in app/alerts.py."
        ),
    }


@router.post("/alerts/test")
def send_test_alert() -> dict[str, Any]:
    """Send a test email to the configured recipients. Confirms SMTP works."""
    sent = alerts.send_alert(
        subject="[DebateLab] Test alert",
        body="This is a test of the spend-cap alert path. If you received it, alerts work.",
    )
    return {
        "sent": sent,
        "recipients": alerts.recipients(),
        "smtp_configured": alerts.is_configured(),
        "note": None if sent else "SMTP not configured or send failed — check host logs.",
    }
