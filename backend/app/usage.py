from __future__ import annotations

from app.db import get_client


def log_usage(
    tenant_id: str,
    api_key_id: str,
    endpoint: str,
    response_status: int,
    latency_ms: int | None = None,
) -> None:
    try:
        get_client().table("api_usage").insert({
            "tenant_id": tenant_id,
            "api_key_id": api_key_id,
            "endpoint": endpoint,
            "response_status": response_status,
            "latency_ms": latency_ms,
        }).execute()
    except Exception as e:
        # Non-fatal — never let logging break the response
        print(f"[usage] Failed to log API call: {e}")
