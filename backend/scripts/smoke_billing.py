"""Smoke test for the billing ledger (cost_events + pricing).

    python3 scripts/smoke_billing.py            # free: no model calls
    python3 scripts/smoke_billing.py --with-ai  # also drives a real turn (~$0.05)

Requires db/migrations/001_cost_events.sql to have been applied. Provisions a
disposable tenant, writes ledger rows directly, asserts the math and the
attribution, then cleans up.
"""

import argparse
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

from app import billing, pricing
from app.db import get_client
from app.main import app

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
PASSED = 0


def ok(m):
    global PASSED
    PASSED += 1
    print(f"{GREEN}PASS{RESET} {m}")


def fail(m):
    print(f"{RED}FAIL{RESET} {m}")
    raise SystemExit(1)


def check(cond, m):
    ok(m) if cond else fail(m)


def require_table() -> None:
    try:
        get_client().table("cost_events").select("id").limit(1).execute()
    except Exception as e:
        print(f"{YELLOW}cost_events table is missing.{RESET}")
        print("Apply db/migrations/001_cost_events.sql in the Supabase SQL editor first.")
        print(f"  ({str(e)[:120]})")
        raise SystemExit(2)


def provision() -> tuple[str, str]:
    db = get_client()
    tid = db.table("tenants").insert({"name": f"SMOKE Billing {secrets.token_hex(4)}"}).execute().data[0]["id"]
    raw = "sk_smoke_" + secrets.token_urlsafe(24)
    db.table("api_keys").insert({
        "tenant_id": tid,
        "key_hash": hashlib.sha256(raw.encode()).hexdigest(),
    }).execute()
    return tid, raw


def rows_for(tenant_id):
    return get_client().table("cost_events").select("*").eq("tenant_id", tenant_id).execute().data or []


def main(with_ai: bool) -> None:
    require_table()
    tenant_id, raw_key = provision()
    client = TestClient(app)

    try:
        # --- pricing math -------------------------------------------------
        c = pricing.token_cost_usd("claude-sonnet-4-6", 1_000_000, 0)
        check(abs(c - 3.00) < 1e-9, f"1M Sonnet input tokens = $3.00 (got ${c:.4f})")
        c = pricing.token_cost_usd("claude-haiku-4-5", 0, 1_000_000)
        check(abs(c - 5.00) < 1e-9, f"1M Haiku output tokens = $5.00 (got ${c:.4f})")
        c = pricing.token_cost_usd("claude-sonnet-4-6", 0, 0, cache_read_tokens=1_000_000)
        check(abs(c - 0.30) < 1e-9, f"cached input billed at 0.10x (got ${c:.4f})")

        # --- attribution --------------------------------------------------
        tok = billing.set_context(tenant_id=tenant_id, debate_id=None)
        billing.record_model_call(
            provider="anthropic", model="claude-haiku-4-5", event_type="score",
            usage={"input_tokens": 10_000, "output_tokens": 1_000,
                   "cache_read_tokens": 0, "cache_write_tokens": 0},
        )
        billing.reset_context(tok)

        rows = rows_for(tenant_id)
        check(len(rows) == 1, f"one ledger row written (got {len(rows)})")
        r = rows[0]
        expected = 10_000 * 1.00 / 1e6 + 1_000 * 5.00 / 1e6
        check(abs(float(r["vendor_cost_usd"]) - expected) < 1e-8,
              f"cost computed correctly (${float(r['vendor_cost_usd']):.8f} == ${expected:.8f})")
        check(r["price_version"] == pricing.CURRENT_VERSION,
              f"row stamped with rate card {r['price_version']}")
        check(r["provider"] == "anthropic" and r["model"] == "claude-haiku-4-5",
              "provider and model recorded")

        # --- unknown model is recorded, not dropped ------------------------
        tok = billing.set_context(tenant_id=tenant_id)
        billing.record_model_call(
            provider="openai", model="gpt-nonexistent-9", event_type="ai_speech",
            usage={"input_tokens": 500, "output_tokens": 100,
                   "cache_read_tokens": 0, "cache_write_tokens": 0},
        )
        billing.reset_context(tok)
        unknown = [x for x in rows_for(tenant_id) if x["model"] == "gpt-nonexistent-9"]
        check(len(unknown) == 1, "unpriced model still produces a ledger row")
        check(unknown[0]["needs_repricing"] is True,
              "unpriced row flagged needs_repricing (tokens preserved for re-billing)")
        check(unknown[0]["input_tokens"] == 500,
              "unpriced row preserved the token counts")

        # --- no tenant bound => website row, not a partner row -------------
        before = len(rows_for(tenant_id))
        billing.record_model_call(
            provider="openai", model="gpt-4o-mini", event_type="ai_speech",
            usage={"input_tokens": 100, "output_tokens": 50,
                   "cache_read_tokens": 0, "cache_write_tokens": 0},
        )
        check(len(rows_for(tenant_id)) == before,
              "unbound context does not attribute cost to any tenant")

        # --- never raises ---------------------------------------------------
        try:
            billing.record_model_call(
                provider="x", model=None, event_type="broken", usage={},
            )
            ok("a malformed record call is swallowed, never raised")
            PASSED_LOCAL = True
        except Exception as e:
            fail(f"billing raised where it must not: {e}")

        # --- end to end -----------------------------------------------------
        if with_ai:
            r = client.post("/api/v1/debates", headers={"X-API-Key": raw_key},
                            json={"motion": "This house would ban single-use plastics",
                                  "starter": "user", "num_rounds": 1,
                                  "mode": "casual", "difficulty": "beginner"})
            check(r.status_code == 201, f"debate created (got {r.status_code})")
            did = r.json()["id"]
            before = len(rows_for(tenant_id))
            r = client.post(f"/api/v1/debates/{did}/turns", headers={"X-API-Key": raw_key},
                            json={"content": "Plastic waste is choking our oceans and the harm is irreversible."})
            check(r.status_code == 200, f"turn succeeded (got {r.status_code})")
            after = rows_for(tenant_id)
            new = [x for x in after if x["id"] not in {y["id"] for y in rows_for(tenant_id)[:before]}]
            check(len(after) > before, f"real turn wrote ledger rows ({before} -> {len(after)})")
            tagged = [x for x in after if x["debate_id"] == did]
            check(len(tagged) > 0, "cost rows tagged with the debate they belong to")
            spend = sum(float(x["vendor_cost_usd"]) for x in after)
            print(f"\n  ledger total for this tenant: ${spend:.6f} across {len(after)} rows")
            for x in after:
                if x["debate_id"] == did:
                    print(f"    {x['event_type']:<16} {x['model']:<24} "
                          f"in={x['input_tokens']:<6} out={x['output_tokens']:<5} "
                          f"${float(x['vendor_cost_usd']):.6f}")

        print(f"\n{GREEN}All {PASSED} assertions passed.{RESET}")
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        fail("unexpected exception")
    finally:
        try:
            get_client().table("cost_events").delete().eq("tenant_id", tenant_id).execute()
            get_client().table("tenants").delete().eq("id", tenant_id).execute()
        except Exception as e:
            print(f"[cleanup] {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-ai", action="store_true", help="also drive a real turn (~$0.05)")
    main(ap.parse_args().with_ai)
