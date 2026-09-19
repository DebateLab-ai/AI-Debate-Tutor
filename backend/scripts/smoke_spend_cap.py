"""Smoke test for the per-tenant spend cap and alert path.

    python3 scripts/smoke_spend_cap.py      # free — no model calls

Writes synthetic cost_events to push a disposable tenant past its cap, then
asserts the 402 fires, the alerts de-duplicate, and read-only endpoints stay
open. Captures emails instead of sending them. Cleans up in a finally block.

Requires migrations 001 and 003.
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

import os
os.environ.setdefault("API_RATE_LIMIT_PER_MIN", "1000")

from fastapi.testclient import TestClient

from app import alerts, spendcap
from app.db import get_client
from app.main import app

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
PASSED = 0
SENT: list[tuple[str, str]] = []

ADMIN_TOKEN = "smoke_" + secrets.token_urlsafe(48)
os.environ["ADMIN_API_TOKEN"] = ADMIN_TOKEN

alerts.send_alert = lambda subject, body: (SENT.append((subject, body)), True)[1]  # noqa: E731


def ok(m):
    global PASSED
    PASSED += 1
    print(f"{GREEN}PASS{RESET} {m}")


def fail(m):
    print(f"{RED}FAIL{RESET} {m}")
    raise SystemExit(1)


def check(c, m):
    ok(m) if c else fail(m)


def require_migrations():
    db = get_client()
    try:
        db.table("cost_alerts_sent").select("*").limit(1).execute()
    except Exception as e:
        print(f"{YELLOW}cost_alerts_sent missing — apply db/migrations/003_spend_cap.sql{RESET}")
        print(f"  ({str(e)[:110]})")
        raise SystemExit(2)
    try:
        db.table("tenants").select("monthly_cost_limit_usd").limit(1).execute()
    except Exception as e:
        print(f"{YELLOW}tenants.monthly_cost_limit_usd missing — apply 003_spend_cap.sql{RESET}")
        print(f"  ({str(e)[:110]})")
        raise SystemExit(2)


def provision(cap):
    db = get_client()
    tid = db.table("tenants").insert({
        "name": f"SMOKE Cap {secrets.token_hex(3)}",
        "monthly_cost_limit_usd": cap,
    }).execute().data[0]["id"]
    raw = "sk_smoke_" + secrets.token_urlsafe(24)
    db.table("api_keys").insert({
        "tenant_id": tid, "key_hash": hashlib.sha256(raw.encode()).hexdigest(),
    }).execute()
    return tid, raw


def add_cost(tenant_id, usd):
    get_client().table("cost_events").insert({
        "tenant_id": tenant_id, "event_type": "ai_speech", "provider": "anthropic",
        "model": "claude-haiku-4-5", "input_tokens": 1, "output_tokens": 1,
        "vendor_cost_usd": usd, "price_version": "smoke",
    }).execute()


def main():
    require_migrations()
    tid, key = provision(cap=1.00)
    client = TestClient(app)
    hdr = {"X-API-Key": key}
    admin = {"X-Admin-Token": ADMIN_TOKEN}

    try:
        # --- RPC / summation -------------------------------------------------
        check(abs(spendcap.month_to_date_cost(tid)) < 1e-9, "new tenant starts at $0.00 MTD")
        add_cost(tid, 0.60)
        check(abs(spendcap.month_to_date_cost(tid) - 0.60) < 1e-6,
              f"MTD reflects a written cost event (${spendcap.month_to_date_cost(tid):.4f})")

        check(spendcap.get_limit(tid, use_cache=False) == 1.00, "cap read back as $1.00")

        # --- under the cap: requests flow ------------------------------------
        # NOTE ordering: every cost-incurring endpoint now schedules a threshold
        # check as a background task, and TestClient runs those before returning.
        # So this create fires the 50% alert by itself — assert on that rather
        # than calling check_thresholds_and_alert first and finding it deduped.
        SENT.clear()
        r = client.post("/api/v1/debates", headers=hdr,
                        json={"motion": "Cap probe", "starter": "user", "num_rounds": 1})
        check(r.status_code == 201, f"under cap: create allowed (got {r.status_code})")
        did = r.json()["id"]

        # --- 50% threshold alert ---------------------------------------------
        check(len(SENT) == 1 and "50%" in SENT[0][0],
              f"50% threshold emailed once, fired by the request itself "
              f"(subjects={[s for s, _ in SENT]})")
        SENT.clear()
        spendcap.check_thresholds_and_alert(tid, "SMOKE Cap")
        check(len(SENT) == 0, "repeat check does not re-send the same threshold")

        # --- cross the cap ----------------------------------------------------
        add_cost(tid, 0.50)   # now $1.10 of $1.00
        spendcap.invalidate_limit_cache(tid)
        SENT.clear()
        spendcap.check_thresholds_and_alert(tid, "SMOKE Cap")
        subjects = " ".join(s for s, _ in SENT)
        check(len(SENT) == 2, f"exactly two new thresholds fire (got {len(SENT)})")
        check("crossed 80%" in subjects and "crossed 100%" in subjects,
              f"80% and 100% each named in their own subject (sent={[s for s,_ in SENT]})")
        check(all("now at 110%" in s for s, _ in SENT),
              "both subjects report current usage of 110%")
        check(any("SPENDING HALTED" in s for s, _ in SENT),
              "the over-cap email says spending is halted")

        # --- hard stop --------------------------------------------------------
        r = client.post(f"/api/v1/debates/{did}/turns", headers=hdr,
                        json={"content": "This should be rejected before any model call."})
        check(r.status_code == 402, f"over cap: /turns returns 402 (got {r.status_code})")
        check("cap reached" in r.json()["detail"].lower(),
              f"402 explains why: {r.json()['detail'][:70]}")

        r = client.post("/api/v1/drills/rebuttal/start", headers=hdr,
                        json={"motion": "Cap probe", "user_position": "for"})
        check(r.status_code == 402, f"over cap: drills also blocked (got {r.status_code})")

        # --- reads stay open --------------------------------------------------
        check(client.get(f"/api/v1/debates/{did}", headers=hdr).status_code == 200,
              "over cap: GET a debate still works (already paid for)")
        check(client.get("/api/v1/debates", headers=hdr).status_code == 200,
              "over cap: listing debates still works")

        # --- raising the cap unblocks immediately -----------------------------
        r = client.put(f"/internal/admin/tenants/{tid}/cap", headers=admin,
                       json={"monthly_cost_limit_usd": 50.0})
        check(r.status_code == 200, f"admin raised the cap (got {r.status_code})")
        r = client.post("/api/v1/debates", headers=hdr,
                        json={"motion": "After raise", "starter": "user", "num_rounds": 1})
        check(r.status_code == 201,
              f"raising the cap takes effect at once, no 60s cache wait (got {r.status_code})")

        # --- removing the cap --------------------------------------------------
        r = client.put(f"/internal/admin/tenants/{tid}/cap", headers=admin,
                       json={"monthly_cost_limit_usd": None})
        check(r.status_code == 200 and r.json()["monthly_cost_limit_usd"] is None,
              "cap can be cleared (null = uncapped)")
        check(spendcap.get_limit(tid, use_cache=False) is None, "uncapped tenant reads as None")

        # --- admin visibility ---------------------------------------------------
        r = client.get("/internal/admin/spend", headers=admin)
        check(r.status_code == 200, f"admin /spend returns 200 (got {r.status_code})")
        mine = [t for t in r.json()["tenants"] if t["tenant_id"] == tid]
        check(len(mine) == 1 and abs(mine[0]["mtd_cost_usd"] - 1.10) < 1e-6,
              f"admin /spend shows this tenant's MTD (${mine[0]['mtd_cost_usd'] if mine else '?'})")

        r = client.get("/internal/admin/alerts/config", headers=admin)
        check(r.status_code == 200 and "ricochandra128@gmail.com" in r.json()["recipients"],
              f"alert recipients configured: {r.json().get('recipients')}")

        # --- uncapped tenants are unaffected --------------------------------
        tid2, key2 = provision(cap=None)
        try:
            add_cost(tid2, 999.0)
            r = client.post("/api/v1/debates", headers={"X-API-Key": key2},
                            json={"motion": "Uncapped", "starter": "user", "num_rounds": 1})
            check(r.status_code == 201,
                  f"a tenant with no cap is never blocked (got {r.status_code})")
            SENT.clear()
            spendcap.check_thresholds_and_alert(tid2, "uncapped")
            check(len(SENT) == 0, "no cap means no alerts")
        finally:
            get_client().table("tenants").delete().eq("id", tid2).execute()

        print(f"\n{GREEN}All {PASSED} assertions passed.{RESET}")
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        fail("unexpected exception")
    finally:
        try:
            get_client().table("cost_events").delete().eq("tenant_id", tid).execute()
            get_client().table("tenants").delete().eq("id", tid).execute()
        except Exception as e:
            print(f"[cleanup] {e}")


if __name__ == "__main__":
    main()
