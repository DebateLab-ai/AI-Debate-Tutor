import os
import time
import threading
from collections import defaultdict, deque

from fastapi import HTTPException, status

# Per-API-key sliding-window rate limiter for the third-party API (/api/v1/*).
#
# State is in-memory and PER-PROCESS. The backend runs a single uvicorn worker
# (see backend/Procfile — no --workers flag), so this is correct for the current
# deployment. If the host is ever scaled to multiple workers/instances, each
# process would enforce its own window independently; move this to a shared
# store (e.g. Redis) at that point.

RATE_LIMIT = int(os.getenv("API_RATE_LIMIT_PER_MIN", "15"))  # requests per window
RATE_WINDOW = int(os.getenv("API_RATE_WINDOW_SECONDS", "60"))  # window length, seconds

_lock = threading.Lock()
_hits: dict[str, deque] = defaultdict(deque)


def enforce_rate_limit(
    api_key_id: str,
    limit: int = RATE_LIMIT,
    window: int = RATE_WINDOW,
) -> None:
    """Raise HTTP 429 if this API key has exceeded `limit` requests in `window` seconds.

    Sliding window: keeps the timestamps of recent requests per key, prunes those
    older than the window, and rejects once the count reaches the limit.
    """
    now = time.monotonic()
    cutoff = now - window
    with _lock:
        q = _hits[api_key_id]
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) >= limit:
            retry_after = int(q[0] + window - now) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: max {limit} requests per {window}s",
                headers={"Retry-After": str(retry_after)},
            )
        q.append(now)
