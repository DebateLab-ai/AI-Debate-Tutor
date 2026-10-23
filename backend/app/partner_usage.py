"""Partner-facing usage statement.

GET /api/v1/usage reads this. It is tenant-scoped and must stay that way.
The private admin reader in usage_store.py is a different module on purpose:
that one scans every tenant.

The invoice number is the sum of cost_events.vendor_cost_usd for the tenant
in the UTC month. Those values were priced when the model call happened
(app/pricing.py) and stored at 8 decimal places. This module adds the stored
numbers. It does not reprice them.

That sum is the same population tenant_mtd_cost_usd() uses for the spend cap:
every cost row, including safety checks and rows flagged needs_repricing.
Filtering either one out would make the statement and the cap disagree.

One student turn writes several cost rows (the speech, sometimes two model
calls, plus moderation). They share a request_id stamped in
bind_billing_context. /usage/events rolls those rows into one line. Rows from
before that column existed have a null request_id: they stay in the month
total and in amount_without_call_id_usd, and they do not appear as lines.

PostgREST on this project rejects SUM, so the rows are paged and added in
Decimal. Each stored value is recovered to 8 decimal places before the add,
so a float coming back from the client cannot drift the total.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from app.db import get_client
from app import spendcap

_PERIOD = re.compile(r"^(\d{4})-(\d{2})$")
_Q = Decimal("0.00000001")
_PAGE = 1000
# Above this we refuse to return a total. A partial sum would understate the bill.
_MAX_ROWS = 100_000

# Partner-facing buckets. Pass names stay on the internal event_type and are
# collapsed here so the response does not describe the speech pipeline.
_KIND = {
    "ai_speech": "speech",
    "ai_speech_pass1": "speech",
    "ai_speech_pass2": "speech",
    "score": "score",
    "drill_claim": "drill",
    "drill_score": "drill",
    "moderation": "moderation",
}
_KIND_ORDER = ("speech", "score", "drill", "moderation", "transcription", "other")


class InvalidPeriod(ValueError):
    """period was not YYYY-MM or all, or the month is in the future."""


class UsageUnavailable(Exception):
    """Ledger table cannot be read."""


class MissingRequestIdColumn(UsageUnavailable):
    """request_id has not been migrated. Month totals still work."""


class UsageIncomplete(Exception):
    """The window has more rows than we will silently truncate."""


def partner_kind(event_type: Optional[str]) -> str:
    return _KIND.get(event_type or "", "other")


def money(value: Any) -> Decimal:
    """Stored vendor cost, recovered to 8 decimal places.

    Postgres numeric(12,8) often arrives as a float. Decimal(float) keeps the
    binary value, and quantizing to 8 places returns the number we stored.
    Strings (when the client preserves them) are taken exactly.
    """
    if value is None or value == "":
        return Decimal(0).quantize(_Q)
    if isinstance(value, str):
        return Decimal(value).quantize(_Q)
    if isinstance(value, Decimal):
        return value.quantize(_Q)
    return Decimal(value).quantize(_Q)


def money_str(value: Decimal) -> str:
    return f"{value.quantize(_Q):.8f}"


def parse_period(
    period: Optional[str], now: Optional[datetime] = None
) -> tuple[str, Optional[datetime], datetime]:
    """Return (label, start inclusive or None, end exclusive) in UTC.

    None means the current month. "all" means every charge up to `now`, with
    no start. For a month, end is the first instant of the next month, so a
    row written at 23:59:59.999 on the last day is inside and a row at 00:00
    on the 1st of the next month is not.
    """
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    else:
        clock = clock.astimezone(timezone.utc)

    raw = "" if period is None else str(period).strip()
    if raw.lower() == "all":
        return "all", None, clock

    if not raw:
        year, month = clock.year, clock.month
    else:
        match = _PERIOD.match(raw)
        if not match:
            raise InvalidPeriod("period must be YYYY-MM, or all")
        year, month = int(match.group(1)), int(match.group(2))
        if not 1 <= month <= 12:
            raise InvalidPeriod("period must be YYYY-MM, or all")
        if (year, month) > (clock.year, clock.month):
            raise InvalidPeriod("period is in the future")

    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    return f"{year:04d}-{month:02d}", start, end


def _apply_window(query, start: Optional[datetime], end: Optional[datetime]):
    if start is not None:
        query = query.gte("created_at", start.isoformat())
    if end is not None:
        query = query.lt("created_at", end.isoformat())
    return query


def _missing_column(exc: Exception, column: str) -> bool:
    return column.lower() in str(exc).lower()


def _fetch_pages(build) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        if offset >= _MAX_ROWS:
            raise UsageIncomplete(
                "Too many usage rows to total exactly. Contact DebateLab."
            )
        page = build(offset, _PAGE).execute().data or []
        rows.extend(page)
        if len(page) < _PAGE:
            return rows
        offset += _PAGE


def _fetch_costs(
    tenant_id: str, start: Optional[datetime], end: Optional[datetime]
) -> tuple[list[dict[str, Any]], bool]:
    """Cost rows for the window. Second value is whether request_id was readable."""

    def _query(select: str, offset: int, page: int):
        query = (
            get_client()
            .table("cost_events")
            .select(select)
            .eq("tenant_id", tenant_id)
        )
        return (
            _apply_window(query, start, end)
            .order("id")
            .range(offset, offset + page - 1)
        )

    try:
        return _fetch_pages(
            lambda offset, page: _query(
                "id, event_type, vendor_cost_usd, needs_repricing, request_id",
                offset, page,
            )
        ), True
    except UsageIncomplete:
        raise
    except Exception as exc:
        if not _missing_column(exc, "request_id"):
            print(f"[partner_usage] cost read failed: {type(exc).__name__}: {exc}")
            raise UsageUnavailable("Usage data is temporarily unavailable") from exc
    try:
        rows = _fetch_pages(
            lambda offset, page: _query(
                "id, event_type, vendor_cost_usd, needs_repricing",
                offset, page,
            )
        )
    except UsageIncomplete:
        raise
    except Exception as exc:
        print(f"[partner_usage] cost read failed: {type(exc).__name__}: {exc}")
        raise UsageUnavailable("Usage data is temporarily unavailable") from exc
    for row in rows:
        row["request_id"] = None
    return rows, False


def _fetch_usage(
    tenant_id: str,
    start: Optional[datetime],
    end: Optional[datetime],
    *,
    with_request_id: bool,
) -> list[dict[str, Any]]:
    select = "id, endpoint, response_status, request_id" if with_request_id else "id, endpoint, response_status"

    def _query(offset: int, page: int):
        query = (
            get_client()
            .table("api_usage")
            .select(select)
            .eq("tenant_id", tenant_id)
        )
        return (
            _apply_window(query, start, end)
            .order("id")
            .range(offset, offset + page - 1)
        )

    try:
        return _fetch_pages(_query)
    except UsageIncomplete:
        raise
    except Exception as exc:
        if with_request_id and _missing_column(exc, "request_id"):
            raise MissingRequestIdColumn("Per-call usage lines are not available yet.") from exc
        print(f"[partner_usage] request read failed: {type(exc).__name__}: {exc}")
        raise UsageUnavailable("Usage data is temporarily unavailable") from exc


def _cap_str(tenant_id: str) -> Optional[str]:
    limit = spendcap.get_limit(tenant_id, use_cache=False)
    if limit is None:
        return None
    return f"{Decimal(str(limit)).quantize(Decimal('0.01')):.2f}"


def statement_from_rows(
    *,
    period: str,
    start: Optional[datetime],
    end: datetime,
    cap_usd: Optional[str],
    cost_rows: list[dict[str, Any]],
    usage_rows: list[dict[str, Any]],
    per_call_available: bool,
) -> dict[str, Any]:
    """Pure assembly. cost_rows must already be this tenant and this window."""
    buckets: dict[str, dict[str, Any]] = {}
    total = Decimal(0)
    unlinked = Decimal(0)
    unpriced = 0

    for row in cost_rows:
        kind = partner_kind(row.get("event_type"))
        amount = money(row.get("vendor_cost_usd"))
        bucket = buckets.setdefault(kind, {"model_calls": 0, "amount": Decimal(0)})
        bucket["model_calls"] += 1
        bucket["amount"] += amount
        total += amount
        if per_call_available and not row.get("request_id"):
            unlinked += amount
        if row.get("needs_repricing"):
            unpriced += 1

    spend_by_kind = [
        {
            "kind": kind,
            "model_calls": buckets[kind]["model_calls"],
            "amount_usd": money_str(buckets[kind]["amount"]),
        }
        for kind in _KIND_ORDER
        if kind in buckets
    ]

    by_endpoint: dict[str, dict[str, int]] = {}
    with_call_id = 0
    for row in usage_rows:
        endpoint = row.get("endpoint") or "(unknown)"
        acc = by_endpoint.setdefault(endpoint, {"count": 0, "errors": 0})
        acc["count"] += 1
        status = row.get("response_status") or 0
        if status >= 400:
            acc["errors"] += 1
        if row.get("request_id"):
            with_call_id += 1

    requests_by_endpoint = [
        {"endpoint": endpoint, "count": acc["count"], "errors": acc["errors"]}
        for endpoint, acc in sorted(by_endpoint.items(), key=lambda item: (-item[1]["count"], item[0]))
    ]

    return {
        "period": period,
        "period_start": start,
        "period_end": end,
        "currency": "USD",
        "amount_due_usd": money_str(total),
        "monthly_cap_usd": cap_usd,
        "unpriced_events": unpriced,
        "per_call_available": per_call_available,
        "amount_without_call_id_usd": money_str(unlinked) if per_call_available else None,
        "requests": len(usage_rows),
        "requests_with_call_id": with_call_id if per_call_available else 0,
        "requests_by_endpoint": requests_by_endpoint,
        "spend_by_kind": spend_by_kind,
    }


def build_statement(tenant_id: str, period: Optional[str] = None) -> dict[str, Any]:
    label, start, end = parse_period(period)
    cost_rows, per_call_available = _fetch_costs(tenant_id, start, end)
    try:
        usage_rows = _fetch_usage(tenant_id, start, end, with_request_id=per_call_available)
    except MissingRequestIdColumn:
        if not per_call_available:
            raise
        # request_id missing on api_usage only. Counts still work; lines do not.
        usage_rows = _fetch_usage(tenant_id, start, end, with_request_id=False)
        per_call_available = False
        for row in cost_rows:
            row["request_id"] = None
    return statement_from_rows(
        period=label,
        start=start,
        end=end,
        cap_usd=_cap_str(tenant_id),
        cost_rows=cost_rows,
        usage_rows=usage_rows,
        per_call_available=per_call_available,
    )


def _charges_for(tenant_id: str, request_ids: list[str]) -> list[dict[str, Any]]:
    if not request_ids:
        return []

    def _query(offset: int, page: int):
        return (
            get_client()
            .table("cost_events")
            .select("id, request_id, debate_id, event_type, vendor_cost_usd")
            .eq("tenant_id", tenant_id)
            .in_("request_id", request_ids)
            .order("id")
            .range(offset, offset + page - 1)
        )

    try:
        return _fetch_pages(_query)
    except UsageIncomplete:
        raise
    except Exception as exc:
        print(f"[partner_usage] call-line read failed: {type(exc).__name__}: {exc}")
        raise UsageUnavailable("Per-call usage lines are temporarily unavailable") from exc


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _line_from_charges(charges: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], Optional[str]]:
    buckets: dict[str, dict[str, Any]] = {}
    total = Decimal(0)
    debates: set[str] = set()
    for row in charges:
        kind = partner_kind(row.get("event_type"))
        amount = money(row.get("vendor_cost_usd"))
        bucket = buckets.setdefault(kind, {"model_calls": 0, "amount": Decimal(0)})
        bucket["model_calls"] += 1
        bucket["amount"] += amount
        total += amount
        if row.get("debate_id"):
            debates.add(str(row["debate_id"]))
    kinds = [
        {
            "kind": kind,
            "model_calls": buckets[kind]["model_calls"],
            "amount_usd": money_str(buckets[kind]["amount"]),
        }
        for kind in _KIND_ORDER
        if kind in buckets
    ]
    debate_id = debates.pop() if len(debates) == 1 else None
    return money_str(total), kinds, debate_id


def list_events(
    tenant_id: str,
    period: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """One row per HTTP call that carries a request_id, newest first.

    The amount is the sum of every model charge that call wrote. A call with
    no model charge (a create, a read, an idempotent replay) is "0.00000000".
    """
    label, start, end = parse_period(period)
    try:
        query = (
            get_client()
            .table("api_usage")
            .select("request_id, endpoint, response_status, created_at")
            .eq("tenant_id", tenant_id)
        )
        page = (
            _apply_window(query, start, end)
            .not_.is_("request_id", "null")
            .order("id", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        if _missing_column(exc, "request_id"):
            raise MissingRequestIdColumn(
                "Per-call usage lines are not available yet."
            ) from exc
        print(f"[partner_usage] event page failed: {type(exc).__name__}: {exc}")
        raise UsageUnavailable("Per-call usage lines are temporarily unavailable") from exc

    ids = [str(row["request_id"]) for row in page if row.get("request_id")]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for charge in _charges_for(tenant_id, ids):
        grouped.setdefault(str(charge["request_id"]), []).append(charge)

    events = []
    for row in page:
        request_id = str(row["request_id"])
        amount, kinds, debate_id = _line_from_charges(grouped.get(request_id, []))
        events.append({
            "request_id": request_id,
            "created_at": _parse_dt(row.get("created_at")),
            "endpoint": row.get("endpoint") or "(unknown)",
            "response_status": row.get("response_status") or 0,
            "debate_id": debate_id,
            "amount_usd": amount,
            "spend_by_kind": kinds,
        })

    return {"period": label, "limit": limit, "offset": offset, "events": events}
