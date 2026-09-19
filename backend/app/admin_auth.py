"""Admin authentication for the internal-only surface under /internal/*.

Deliberately separate from app/auth.py: partner API keys live in Supabase and
are scoped to one tenant, whereas the admin token is a single shared secret held
by us and reads ACROSS tenants. Keeping the two code paths apart means a bug in
one can never widen the other.

The endpoints are disabled outright unless ADMIN_API_TOKEN is set, so a deploy
that forgets the env var exposes nothing rather than exposing everything.
"""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.ratelimit import enforce_rate_limit

_admin_token_header = APIKeyHeader(name="X-Admin-Token", auto_error=False)

# Rejected admin attempts are throttled hard against brute force. The window is
# shared by all callers (it is one shared secret, not a per-caller credential),
# so this is a global cap on guesses, not a per-client convenience limit.
_FAILED_ATTEMPT_LIMIT = int(os.getenv("ADMIN_FAILED_ATTEMPT_LIMIT", "10"))
_FAILED_ATTEMPT_WINDOW = int(os.getenv("ADMIN_FAILED_ATTEMPT_WINDOW_SECONDS", "300"))

# Refuse to run with a token short enough to be guessable.
MIN_TOKEN_LEN = 32


def require_admin(raw_token: str = Security(_admin_token_header)) -> None:
    """Gate an internal endpoint on the shared admin token.

    Raises 404 when the surface is disabled, so an unconfigured deployment looks
    like it has no admin routes at all rather than advertising a locked door.
    """
    expected = os.getenv("ADMIN_API_TOKEN") or ""

    if not expected:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not Found",
        )

    if len(expected) < MIN_TOKEN_LEN:
        # Misconfiguration on our side, not a caller error. Fail closed and say
        # so in the logs rather than accepting a weak secret.
        print(
            f"[admin] ADMIN_API_TOKEN is shorter than {MIN_TOKEN_LEN} chars; "
            "refusing to serve /internal/*. Generate one with: "
            "python -c 'import secrets; print(secrets.token_urlsafe(48))'"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin surface is misconfigured",
        )

    if not raw_token or not hmac.compare_digest(raw_token, expected):
        enforce_rate_limit(
            "__admin_failed__",
            limit=_FAILED_ATTEMPT_LIMIT,
            window=_FAILED_ATTEMPT_WINDOW,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token",
        )
