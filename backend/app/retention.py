"""Retention policy: delete debates older than 7 days.

Run via scripts/run_retention.py from a daily scheduler (GitHub Actions cron,
Railway cron, Supabase pg_cron — anything works). Messages and scores are
removed transitively via ON DELETE CASCADE on the FK.

api_usage rows are NOT deleted — those are billing/audit records and have no
PII attached to them (they reference tenant_id and api_key_id, never any
student-supplied content).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

from app.db import get_client

RETENTION_DAYS = 7


@dataclass(frozen=True)
class CleanupResult:
    cutoff: datetime
    debates_deleted: int


def delete_expired_debates(*, dry_run: bool = False) -> CleanupResult:
    """Delete every debate whose created_at is older than the retention window.

    With dry_run=True the cutoff is computed and the candidate count is
    returned, but no DELETE is issued. Use for sanity-checking before
    scheduling.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    cutoff_iso = cutoff.isoformat()

    db = get_client()

    if dry_run:
        # Count without deleting — head=True returns no rows but populates count.
        res = (
            db.table("debates")
            .select("id", count="exact")
            .lt("created_at", cutoff_iso)
            .execute()
        )
        return CleanupResult(cutoff=cutoff, debates_deleted=res.count or 0)

    # Supabase's delete() returns the deleted rows; counting len gives our metric.
    # ON DELETE CASCADE on messages.debate_id and scores.debate_id handles the
    # transitive cleanup — see backend/db/schema.sql.
    res = (
        db.table("debates")
        .delete()
        .lt("created_at", cutoff_iso)
        .execute()
    )
    deleted = len(res.data or [])
    return CleanupResult(cutoff=cutoff, debates_deleted=deleted)
