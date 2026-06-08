"""Daily retention cleanup runner.

Local:
    source venv/bin/activate
    python scripts/run_retention.py            # delete
    python scripts/run_retention.py --dry-run  # report only

Scheduled (GitHub Actions): see .github/workflows/retention-cleanup.yml.

Requires SUPABASE_URL and SUPABASE_SERVICE_KEY in the environment (or in
backend/.env locally). Exits 0 on success, 1 on any error.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app.retention import RETENTION_DAYS, delete_expired_debates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be deleted, but don't delete.")
    args = parser.parse_args()

    try:
        result = delete_expired_debates(dry_run=args.dry_run)
    except Exception:
        traceback.print_exc()
        return 1

    verb = "would delete" if args.dry_run else "deleted"
    print(
        f"[retention] cutoff={result.cutoff.isoformat()} "
        f"({RETENTION_DAYS}d): {verb} {result.debates_deleted} debate(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
