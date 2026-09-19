"""Read side for api_usage — internal/admin only.

app/usage.py writes one api_usage row per authenticated partner request; this
module is the only thing that reads them back. Unlike debates_store, these
functions are deliberately NOT tenant-scoped: they exist so we can see usage
across every tenant. Nothing here may be reachable from /api/v1/*.

Aggregation happens in Python rather than SQL because the Supabase client has
no GROUP BY. That is fine at current volume, but it means a summary reads every
matching row, so both the window and the row count are capped and the caller is
told when a result was truncated.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from app.db import get_client

# Hard ceiling on rows pulled for an aggregate. Above this the summary would be
# computed from a partial window and silently understate usage, so we flag it.
SUMMARY_MAX_ROWS = 50_000
SUMMARY_PAGE_SIZE = 1_000

# Longest window a single summary may cover, to bound the scan.
SUMMARY_MAX_DAYS = 92


def _resolve_window(
    since: Optional[datetime],
    until: Optional[datetime],
    default_days: int,
) -> tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    end = until or now
    start = since or (end - timedelta(days=default_days))
    if start > end:
        raise ValueError("`since` must be earlier than `until`")
    if (end - start) > timedelta(days=SUMMARY_MAX_DAYS):
        raise ValueError(f"Window may not exceed {SUMMARY_MAX_DAYS} days")
    return start, end


def list_usage(
    *,
    tenant_id: Optional[str] = None,
    api_key_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    status_min: Optional[int] = None,
    status_max: Optional[int] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Raw api_usage rows, newest first, with the owning tenant's name joined in."""
    q = (
        get_client()
        .table("api_usage")
        .select("id, tenant_id, api_key_id, endpoint, response_status, latency_ms, created_at, tenants(name)")
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
    )
    if tenant_id is not None:
        q = q.eq("tenant_id", tenant_id)
    if api_key_id is not None:
        q = q.eq("api_key_id", api_key_id)
    if endpoint is not None:
        q = q.eq("endpoint", endpoint)
    if status_min is not None:
        q = q.gte("response_status", status_min)
    if status_max is not None:
        q = q.lte("response_status", status_max)
    if since is not None:
        q = q.gte("created_at", since.isoformat())
    if until is not None:
        q = q.lt("created_at", until.isoformat())

    rows = q.execute().data or []
    for r in rows:
        tenant = r.pop("tenants", None) or {}
        r["tenant_name"] = tenant.get("name")
    return rows


def _fetch_for_summary(
    *,
    tenant_id: Optional[str],
    start: datetime,
    end: datetime,
) -> tuple[list[dict[str, Any]], bool]:
    """Page through the window up to SUMMARY_MAX_ROWS. Returns (rows, truncated)."""
    rows: list[dict[str, Any]] = []
    offset = 0
    while offset < SUMMARY_MAX_ROWS:
        page_size = min(SUMMARY_PAGE_SIZE, SUMMARY_MAX_ROWS - offset)
        q = (
            get_client()
            .table("api_usage")
            .select("tenant_id, api_key_id, endpoint, response_status, latency_ms, created_at, tenants(name)")
            .gte("created_at", start.isoformat())
            .lt("created_at", end.isoformat())
            .order("created_at", desc=False)
            .range(offset, offset + page_size - 1)
        )
        if tenant_id is not None:
            q = q.eq("tenant_id", tenant_id)

        page = q.execute().data or []
        rows.extend(page)
        if len(page) < page_size:
            return rows, False
        offset += page_size

    # We filled the ceiling exactly; there may or may not be more. Probe once so
    # `truncated` is accurate instead of a guess.
    probe = (
        get_client()
        .table("api_usage")
        .select("id")
        .gte("created_at", start.isoformat())
        .lt("created_at", end.isoformat())
        .range(SUMMARY_MAX_ROWS, SUMMARY_MAX_ROWS)
    )
    if tenant_id is not None:
        probe = probe.eq("tenant_id", tenant_id)
    return rows, bool(probe.execute().data)


def _percentile(sorted_values: list[int], pct: float) -> Optional[int]:
    if not sorted_values:
        return None
    # Nearest-rank: the smallest value at or above the requested percentile.
    idx = max(0, min(len(sorted_values) - 1, int(round(pct * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def _bucket_stats(latencies: list[int], total: int, errors: int, client_errors: int) -> dict[str, Any]:
    latencies.sort()
    return {
        "requests": total,
        "errors_5xx": errors,
        "errors_4xx": client_errors,
        "error_rate": round((errors + client_errors) / total, 4) if total else 0.0,
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "latency_ms_max": latencies[-1] if latencies else None,
    }


def summarize_usage(
    *,
    tenant_id: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    default_days: int = 30,
) -> dict[str, Any]:
    """Aggregate api_usage over a window, broken down by tenant, endpoint and day."""
    start, end = _resolve_window(since, until, default_days)
    rows, truncated = _fetch_for_summary(tenant_id=tenant_id, start=start, end=end)

    by_tenant: dict[str, dict[str, Any]] = {}
    by_endpoint: dict[str, dict[str, Any]] = {}
    by_day: dict[str, dict[str, Any]] = {}
    overall_latencies: list[int] = []
    overall_5xx = 0
    overall_4xx = 0

    def _acc(bucket: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
        return bucket.setdefault(
            key,
            {"latencies": [], "total": 0, "errors_5xx": 0, "errors_4xx": 0, "extra": {}},
        )

    for r in rows:
        status_code = r.get("response_status") or 0
        latency = r.get("latency_ms")
        is_5xx = status_code >= 500
        is_4xx = 400 <= status_code < 500

        tenant_key = str(r.get("tenant_id"))
        endpoint_key = r.get("endpoint") or "(unknown)"
        created = str(r.get("created_at") or "")
        day_key = created[:10] or "(unknown)"

        for bucket, key in ((by_tenant, tenant_key), (by_endpoint, endpoint_key), (by_day, day_key)):
            acc = _acc(bucket, key)
            acc["total"] += 1
            if is_5xx:
                acc["errors_5xx"] += 1
            if is_4xx:
                acc["errors_4xx"] += 1
            if latency is not None:
                acc["latencies"].append(latency)

        tenant_meta = (r.get("tenants") or {})
        by_tenant[tenant_key]["extra"].setdefault("tenant_name", tenant_meta.get("name"))
        by_tenant[tenant_key]["extra"].setdefault("api_key_ids", set())
        by_tenant[tenant_key]["extra"]["api_key_ids"].add(str(r.get("api_key_id")))

        overall_5xx += 1 if is_5xx else 0
        overall_4xx += 1 if is_4xx else 0
        if latency is not None:
            overall_latencies.append(latency)

    def _render(bucket: dict[str, dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
        out = []
        for key, acc in bucket.items():
            entry = {key_name: key}
            extra = acc["extra"]
            if "tenant_name" in extra:
                entry["tenant_name"] = extra["tenant_name"]
            if "api_key_ids" in extra:
                entry["active_api_keys"] = len(extra["api_key_ids"])
            entry.update(
                _bucket_stats(acc["latencies"], acc["total"], acc["errors_5xx"], acc["errors_4xx"])
            )
            out.append(entry)
        return out

    return {
        "window": {"since": start.isoformat(), "until": end.isoformat()},
        "truncated": truncated,
        "max_rows_scanned": SUMMARY_MAX_ROWS,
        "totals": _bucket_stats(overall_latencies, len(rows), overall_5xx, overall_4xx),
        "by_tenant": sorted(_render(by_tenant, "tenant_id"), key=lambda e: -e["requests"]),
        "by_endpoint": sorted(_render(by_endpoint, "endpoint"), key=lambda e: -e["requests"]),
        "by_day": sorted(_render(by_day, "date"), key=lambda e: e["date"]),
    }


def list_tenants() -> list[dict[str, Any]]:
    """Every tenant with its key count — the lookup table for the filters above."""
    tenants = (
        get_client()
        .table("tenants")
        .select("id, name, is_active, created_at")
        .order("created_at", desc=True)
        .execute()
        .data
        or []
    )
    keys = (
        get_client()
        .table("api_keys")
        .select("id, tenant_id, is_active, last_used_at, created_at")
        .execute()
        .data
        or []
    )
    by_tenant: dict[str, list[dict[str, Any]]] = {}
    for k in keys:
        by_tenant.setdefault(str(k["tenant_id"]), []).append(k)

    for t in tenants:
        owned = by_tenant.get(str(t["id"]), [])
        last_used = [k["last_used_at"] for k in owned if k.get("last_used_at")]
        t["api_keys_total"] = len(owned)
        t["api_keys_active"] = sum(1 for k in owned if k.get("is_active"))
        t["last_used_at"] = max(last_used) if last_used else None
    return tenants
