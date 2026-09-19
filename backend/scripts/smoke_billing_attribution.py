"""Does every partner model call get attributed to the right tenant?

    python3 scripts/smoke_billing_attribution.py     # free, no model spend

This exists because the failure mode is SILENT. A ContextVar set inside a sync
FastAPI dependency does not reach the endpoint body (anyio hands each threadpool
call a context copy), so cost rows land with tenant_id NULL and look like
website traffic. Everything still returns 200. The money just vanishes.

Intercepts billing._insert rather than writing to Supabase, so it runs without
the cost_events migration and costs nothing.
"""

import hashlib
import secrets
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from fastapi.testclient import TestClient

from app import billing
from app.db import get_client
from app.main import app

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
PASSED = 0
CAPTURED: list[dict] = []


def ok(m):
    global PASSED
    PASSED += 1
    print(f"{GREEN}PASS{RESET} {m}")


def fail(m):
    print(f"{RED}FAIL{RESET} {m}")
    raise SystemExit(1)


def check(c, m):
    ok(m) if c else fail(m)


billing._insert = lambda row: CAPTURED.append(row)   # noqa: E731


def provision(name):
    db = get_client()
    tid = db.table("tenants").insert({"name": name}).execute().data[0]["id"]
    raw = "sk_smoke_" + secrets.token_urlsafe(24)
    db.table("api_keys").insert({
        "tenant_id": tid, "key_hash": hashlib.sha256(raw.encode()).hexdigest(),
    }).execute()
    return tid, raw


def main():
    client = TestClient(app)
    a_id, a_key = provision(f"SMOKE Attr A {secrets.token_hex(3)}")
    b_id, b_key = provision(f"SMOKE Attr B {secrets.token_hex(3)}")

    try:
        # A blocked speech still triggers moderation, which records a cost event.
        # That gives us an attributed model call with no AI spend.
        CAPTURED.clear()
        r = client.post("/api/v1/debates", headers={"X-API-Key": a_key},
                        json={"motion": "Attribution probe", "starter": "user", "num_rounds": 1})
        check(r.status_code == 201, f"tenant A created a debate (got {r.status_code})")
        did = r.json()["id"]

        CAPTURED.clear()
        client.post(f"/api/v1/debates/{did}/turns", headers={"X-API-Key": a_key},
                    json={"content": "A perfectly ordinary argument about ocean plastic."})

        check(len(CAPTURED) > 0, f"the turn produced cost events ({len(CAPTURED)})")
        tenants = {row.get("tenant_id") for row in CAPTURED}
        check(tenants == {a_id},
              f"every cost event attributed to tenant A (saw {tenants})")
        check(None not in tenants,
              "no cost event leaked to tenant_id=NULL (the silent revenue loss)")
        mods = [r for r in CAPTURED if r["event_type"] == "moderation"]
        check(len(mods) > 0 and mods[0]["tenant_id"] == a_id,
              "moderation call attributed (it runs before AI generation)")
        tagged = [r for r in CAPTURED if r.get("debate_id") == did]
        check(len(tagged) > 0, "at least one row tagged with the debate id")

        # Second tenant must not inherit the first one's binding.
        CAPTURED.clear()
        r = client.post("/api/v1/debates", headers={"X-API-Key": b_key},
                        json={"motion": "Attribution probe B", "starter": "user", "num_rounds": 1})
        did_b = r.json()["id"]
        CAPTURED.clear()
        client.post(f"/api/v1/debates/{did_b}/turns", headers={"X-API-Key": b_key},
                    json={"content": "A different ordinary argument, from the other tenant."})
        tenants_b = {row.get("tenant_id") for row in CAPTURED}
        check(tenants_b == {b_id},
              f"tenant B's calls bill to B, not A (saw {tenants_b})")

        # The website path must stay unattributed.
        CAPTURED.clear()
        from app.safety import check_text
        check_text("An ordinary sentence with no tenant bound.")
        check(all(r.get("tenant_id") is None for r in CAPTURED),
              "an unauthenticated/website call records with tenant_id NULL")

        # Rate limiter must not be double-counted by the router-level dependency.
        from app.ratelimit import _hits
        key_hits = {k: len(v) for k, v in _hits.items() if v}
        print(f"\n  rate-limiter buckets after 4 authenticated requests: {key_hits}")
        check(all(n <= 4 for n in key_hits.values()),
              "auth dependency ran once per request (rate limiter not double-counted)")

        print(f"\n{GREEN}All {PASSED} assertions passed.{RESET}")
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        fail("unexpected exception")
    finally:
        for tid in (a_id, b_id):
            try:
                get_client().table("tenants").delete().eq("id", tid).execute()
            except Exception as e:
                print(f"[cleanup] {e}")


if __name__ == "__main__":
    main()
