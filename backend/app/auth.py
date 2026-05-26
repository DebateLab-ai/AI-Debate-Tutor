import hashlib
from dataclasses import dataclass

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.db import get_client
from app.ratelimit import enforce_rate_limit

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass
class AuthContext:
    tenant_id: str
    api_key_id: str


def verify_api_key(raw_key: str = Security(_api_key_header)) -> AuthContext:
    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header required",
        )

    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    db = get_client()
    response = (
        db.table("api_keys")
        .select("id, tenant_id, is_active, tenants(is_active)")
        .eq("key_hash", key_hash)
        .maybe_single()
        .execute()
    )

    # maybe_single() returns None (not a response object) when no row matches
    row = response.data if response else None

    if not row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    if not row["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key has been revoked",
        )
    if not row["tenants"]["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    # Throttle per key once the key is known-valid. Raises 429 if over the limit,
    # so a rejected request skips the last_used_at write below.
    enforce_rate_limit(str(row["id"]))

    # Fire-and-forget last_used_at update — failure here is non-fatal
    try:
        db.table("api_keys").update({"last_used_at": "now()"}).eq("id", row["id"]).execute()
    except Exception:
        pass

    return AuthContext(tenant_id=str(row["tenant_id"]), api_key_id=str(row["id"]))
