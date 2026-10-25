from __future__ import annotations

from app.db import get_client


def log_usage(
    tenant_id: str,
    api_key_id: str,
    endpoint: str,
    response_status: int,
    latency_ms: int | None = None,
    request_id: str | None = None,
) -> None:
    row = {
        "tenant_id": tenant_id,
        "api_key_id": api_key_id,
        "endpoint": endpoint,
        "response_status": response_status,
        "latency_ms": latency_ms,
    }
    if request_id:
        row["request_id"] = request_id
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
