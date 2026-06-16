"""Tenant-scoped persistence for third-party API debates.

Every public function in this module REQUIRES a tenant_id. Reads that don't
match the tenant return None — the caller maps that to a 404. There is no code
path that returns a debate without verifying tenant ownership; that is the
whole point of this layer.

The frontend's in-memory DEBATES dict in main.py is intentionally untouched —
this store backs `/api/v1/*` only.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

from app.db import get_client

Speaker = Literal["user", "assistant"]
DebateStatus = Literal["active", "completed"]
Mode = Literal["casual", "wsdc", "ap"]
Difficulty = Literal["beginner", "intermediate", "advanced"]

# Metadata caps. Partners pass arbitrary key/value pairs (rendered into the PDF
# header). Cap both shape and size so a partner can't paste a full student record
# into a free-form bag. Enforced here, not in SQL, so we can return a clear 400.
METADATA_MAX_FIELDS = 10
METADATA_MAX_KEY_LEN = 50
METADATA_MAX_VALUE_LEN = 200


class MetadataError(ValueError):
    """Raised when a partner-supplied metadata bag violates the size/shape caps."""


def validate_metadata(metadata: Optional[dict[str, Any]]) -> dict[str, str]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise MetadataError("metadata must be an object")
    if len(metadata) > METADATA_MAX_FIELDS:
        raise MetadataError(f"metadata may have at most {METADATA_MAX_FIELDS} fields")
    cleaned: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise MetadataError("metadata keys must be non-empty strings")
        if len(key) > METADATA_MAX_KEY_LEN:
            raise MetadataError(f"metadata key '{key[:20]}...' exceeds {METADATA_MAX_KEY_LEN} chars")
        # Coerce simple scalars to string for stable PDF rendering. Reject nested
        # structures — partners who need structure can flatten on their side.
        if isinstance(value, (str, int, float, bool)):
            str_value = str(value)
        else:
            raise MetadataError(f"metadata['{key}'] must be a string, number, or boolean")
        if len(str_value) > METADATA_MAX_VALUE_LEN:
            raise MetadataError(f"metadata['{key}'] exceeds {METADATA_MAX_VALUE_LEN} chars")
        cleaned[key] = str_value
    return cleaned


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


# ---------- Debates ----------

def create_debate(
    *,
    tenant_id: str,
    motion: Optional[str],
    starter: Speaker,
    num_rounds: int,
    mode: Mode = "casual",
    difficulty: Difficulty = "intermediate",
    external_user_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    row = {
        "id": str(uuid4()),
        "tenant_id": tenant_id,
        "external_user_id": external_user_id,
        "motion": motion,
        "starter": starter,
        "next_speaker": starter,
        "mode": mode,
        "difficulty": difficulty,
        "num_rounds": num_rounds,
        "current_round": 1,
        "status": "active",
        "metadata": validate_metadata(metadata),
    }
    res = get_client().table("debates").insert(row).execute()
    return res.data[0]


def get_debate(*, tenant_id: str, debate_id: UUID | str) -> Optional[dict[str, Any]]:
    """Return the debate iff it belongs to tenant_id, else None.

    Tenant filter is applied in the WHERE clause — a debate owned by a different
    tenant is indistinguishable from a non-existent one.
    """
    res = (
        get_client()
        .table("debates")
        .select("*")
        .eq("id", str(debate_id))
        .eq("tenant_id", tenant_id)
        .maybe_single()
        .execute()
    )
    return res.data if res else None


def list_debates(
    *,
    tenant_id: str,
    external_user_id: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    q = (
        get_client()
        .table("debates")
        .select("*")
        .eq("tenant_id", tenant_id)
        .order("created_at", desc=True)
        .limit(min(limit, 200))
    )
    if external_user_id is not None:
        q = q.eq("external_user_id", external_user_id)
    if since is not None:
        q = q.gte("created_at", since.isoformat())
    res = q.execute()
    return res.data or []


def update_debate_state(
    *,
    tenant_id: str,
    debate_id: UUID | str,
    current_round: int,
    next_speaker: Speaker,
    status: DebateStatus,
) -> Optional[dict[str, Any]]:
    res = (
        get_client()
        .table("debates")
        .update({
            "current_round": current_round,
            "next_speaker": next_speaker,
            "status": status,
            "updated_at": _now_iso(),
        })
        .eq("id", str(debate_id))
        .eq("tenant_id", tenant_id)
        .execute()
    )
    return res.data[0] if res.data else None


# ---------- Messages ----------

def append_message(
    *,
    tenant_id: str,
    debate_id: UUID | str,
    round_no: int,
    speaker: Speaker,
    content: str,
) -> dict[str, Any]:
    row = {
        "id": str(uuid4()),
        "tenant_id": tenant_id,
        "debate_id": str(debate_id),
        "round_no": round_no,
        "speaker": speaker,
        "content": content,
    }
    res = get_client().table("messages").insert(row).execute()
    return res.data[0]


def list_messages(*, tenant_id: str, debate_id: UUID | str) -> list[dict[str, Any]]:
    res = (
        get_client()
        .table("messages")
        .select("*")
        .eq("tenant_id", tenant_id)
        .eq("debate_id", str(debate_id))
        .order("created_at", desc=False)
        .execute()
    )
    return res.data or []


def delete_message(
    *,
    tenant_id: str,
    debate_id: UUID | str,
    message_id: UUID | str,
) -> bool:
    """Remove a message scoped to tenant + debate. Used to roll back partial turns."""
    res = (
        get_client()
        .table("messages")
        .delete()
        .eq("id", str(message_id))
        .eq("tenant_id", tenant_id)
        .eq("debate_id", str(debate_id))
        .execute()
    )
    return bool(res.data)


# ---------- Scores ----------

def upsert_score(
    *,
    tenant_id: str,
    debate_id: UUID | str,
    overall: float,
    content_structure: float,
    engagement: float,
    strategy: float,
    feedback: str,
    content_structure_feedback: str,
    engagement_feedback: str,
    strategy_feedback: str,
    weakness_type: Optional[str],
) -> dict[str, Any]:
    row = {
        "debate_id": str(debate_id),
        "tenant_id": tenant_id,
        "overall": overall,
        "content_structure": content_structure,
        "engagement": engagement,
        "strategy": strategy,
        "feedback": feedback,
        "content_structure_feedback": content_structure_feedback,
        "engagement_feedback": engagement_feedback,
        "strategy_feedback": strategy_feedback,
        "weakness_type": weakness_type,
    }
    res = get_client().table("scores").upsert(row, on_conflict="debate_id").execute()
    return res.data[0]


def get_score(*, tenant_id: str, debate_id: UUID | str) -> Optional[dict[str, Any]]:
    res = (
        get_client()
        .table("scores")
        .select("*")
        .eq("debate_id", str(debate_id))
        .eq("tenant_id", tenant_id)
        .maybe_single()
        .execute()
    )
    return res.data if res else None


# ---------- Idempotency (partner POST retries) ----------

def get_idempotency_record(
    *,
    tenant_id: str,
    idempotency_key: str,
) -> Optional[dict[str, Any]]:
    res = (
        get_client()
        .table("idempotency_records")
        .select("*")
        .eq("tenant_id", tenant_id)
        .eq("idempotency_key", idempotency_key)
        .maybe_single()
        .execute()
    )
    return res.data if res else None


def save_idempotency_record(
    *,
    tenant_id: str,
    idempotency_key: str,
    endpoint: str,
    response_status: int,
    response_body: dict[str, Any],
) -> None:
    get_client().table("idempotency_records").insert({
        "tenant_id": tenant_id,
        "idempotency_key": idempotency_key,
        "endpoint": endpoint,
        "response_status": response_status,
        "response_body": response_body,
    }).execute()
