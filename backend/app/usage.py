"""Partner request log.

One api_usage row per authenticated partner request that was not rejected for
auth or rate limiting. 401, 403, and 429 do not get a row. Every other status
does, including 400 and 5xx, so GET /api/v1/usage can count errors.

The row is written here, after the response status is known. Scheduling it as
a FastAPI background task from inside the endpoint drops it whenever the
endpoint raises: FastAPI only runs those tasks when the endpoint returns.

Partner model charges are committed or dropped in the same place. A success
writes the charges held during the request. Any other status drops them, so a
failed call is not on the invoice.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Optional

from app import billing
from app.db import get_client

# Auth and rate-limit rejections are not API attempts. Everything else is.
_SKIP_LOG_STATUS = {401, 403, 429}


def log_usage(
    tenant_id: str,
    api_key_id: str,
    endpoint: str,
    response_status: int,
    latency_ms: int | None = None,
    request_id: str | None = None,
    created_at: datetime | None = None,
) -> None:
    row: dict[str, Any] = {
        "tenant_id": tenant_id,
        "api_key_id": api_key_id,
        "endpoint": endpoint,
        "response_status": response_status,
        "latency_ms": latency_ms,
    }
    if request_id:
        row["request_id"] = request_id
    if created_at is not None:
        row["created_at"] = created_at.isoformat()
    try:
        get_client().table("api_usage").insert(row).execute()
    except Exception as e:
        # Column not migrated yet: keep the call count, drop only the id.
        # A failed insert here must never break the response the partner already got.
        if request_id and "request_id" in str(e).lower():
            row.pop("request_id", None)
            try:
                get_client().table("api_usage").insert(row).execute()
                return
            except Exception as e2:
                print(f"[usage] Failed to log API call: {e2}")
                return
        print(f"[usage] Failed to log API call: {e}")


def endpoint_label(scope: dict[str, Any]) -> str:
    """`POST /api/v1/debates/{id}/turns`, matching the partner docs.

    The route template uses `{debate_id}`. The published spec uses `{id}`.
    """
    route = scope.get("route")
    path = getattr(route, "path", None) or scope.get("path") or ""
    path = str(path).replace("{debate_id}", "{id}")
    method = scope.get("method") or "GET"
    return f"{method} {path}"


def attach_request_id(message: dict[str, Any], request_id: str | None) -> dict[str, Any]:
    """Put this call's usage id on the response, once.

    The value is the request_id later returned by GET /api/v1/usage/events.
    Auth and rate-limit rejections have no id, because they are not usage rows.
    """
    if message.get("type") != "http.response.start" or not request_id:
        return message
    headers = [
        (key, value)
        for key, value in (message.get("headers") or [])
        if key.lower() != b"x-request-id"
    ]
    headers.append((b"x-request-id", request_id.encode("ascii")))
    return {**message, "headers": headers}


def settle_partner_request(scope: dict[str, Any], status_code: int, started: float) -> None:
    """Commit or drop charges, then log the call. Never raises."""
    path = str(scope.get("path") or "")
    if scope.get("type") != "http":
        return
    if path != "/api/v1" and not path.startswith("/api/v1/"):
        return
    ctx = billing.current()
    if not ctx.tenant_id:
        return
    try:
        if status_code < 400:
            billing.commit()
        else:
            billing.discard()
    except Exception as e:
        print(f"[usage] charge settle failed: {e}")
        billing.discard()
    if status_code in _SKIP_LOG_STATUS or not ctx.api_key_id:
        return
    log_usage(
        tenant_id=ctx.tenant_id,
        api_key_id=ctx.api_key_id,
        endpoint=endpoint_label(scope),
        response_status=status_code,
        latency_ms=int((time.perf_counter() - started) * 1000),
        request_id=ctx.request_id,
        created_at=ctx.started_at,
    )


class PartnerUsageMiddleware:
    """ASGI middleware. Not BaseHTTPMiddleware: that one runs the app in a
    child task and would hide the billing ContextVar from this code."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path") or "")
        if path != "/api/v1" and not path.startswith("/api/v1/"):
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status = {"code": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                message = attach_request_id(message, billing.current().request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            settle_partner_request(scope, status["code"], started)
