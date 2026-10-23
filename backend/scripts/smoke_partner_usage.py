"""Smoke test for GET /api/v1/usage.

    python3 scripts/smoke_partner_usage.py

Free. No model calls. Inserts synthetic cost rows and checks the statement
adds them exactly, hides the other tenant, and collapses pass names.

The pure checks run even when Supabase is down. The database checks need
migration 001. Per-call lines need migration 004; without it the month total
is still checked and /usage/events must return 503 rather than a short bill.
"""

import hashlib
import secrets
import sys
import traceback
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import os
os.environ.setdefault("API_RATE_LIMIT_PER_MIN", "1000")

from fastapi.testclient import TestClient

from app import partner_usage
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


def check(c, m):
    ok(m) if c else fail(m)


def pure_checks():
    """No database. Pins the addition and the partner-facing kind names."""
    check(
        partner_usage.money(float(Decimal("1.23456789"))) == Decimal("1.23456789"),
        "an 8-dp cost survives a float round-trip",
    )
    check(partner_usage.money("0.00000001") == Decimal("0.00000001"), "string money is exact")

    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    end = datetime(2026, 11, 1, tzinfo=timezone.utc)
    statement = partner_usage.statement_from_rows(
        period="2026-10",
        start=start,
        end=end,
        cap_usd=None,
        per_call_available=True,
        cost_rows=[
            {"event_type": "ai_speech_pass1", "vendor_cost_usd": "1.23456789", "request_id": "r", "needs_repricing": False},
            {"event_type": "ai_speech_pass2", "vendor_cost_usd": "0.00000001", "request_id": "r", "needs_repricing": False},
            {"event_type": "unmapped_kind", "vendor_cost_usd": "0.10000000", "request_id": None, "needs_repricing": True},
        ],
        usage_rows=[
            {"endpoint": "POST /api/v1/debates/{id}/turns", "response_status": 200, "request_id": "r"},
            {"endpoint": "POST /api/v1/debates", "response_status": 400, "request_id": None},
        ],
    )
    check(statement["amount_due_usd"] == "1.33456790", f"rows sum exactly ({statement['amount_due_usd']})")
    by_kind = {row["kind"]: row for row in statement["spend_by_kind"]}
    check("ai_speech_pass1" not in by_kind and "speech" in by_kind, "pass names collapse to speech")
    check(by_kind["speech"]["amount_usd"] == "1.23456790", "both speech passes add into one kind")
    check(by_kind["speech"]["model_calls"] == 2, "both speech passes still count")
    check(by_kind["other"]["amount_usd"] == "0.10000000", "unknown kinds keep their cost")
    check(statement["unpriced_events"] == 1, "needs_repricing is counted, not dropped")
    check(statement["amount_without_call_id_usd"] == "0.10000000", "unstamped cost is split out, still inside the total")
    check(statement["requests"] == 2 and statement["requests_with_call_id"] == 1, "request counts split stamped vs all")
    errors = {row["endpoint"]: row["errors"] for row in statement["requests_by_endpoint"]}
    check(errors["POST /api/v1/debates"] == 1, "4xx counted as errors")

    try:
        partner_usage.parse_period("2026-13")
        fail("month 13 should be rejected")
    except partner_usage.InvalidPeriod:
        ok("month 13 rejected")
    try:
        partner_usage.parse_period("2099-01")
        fail("a future month should be rejected")
    except partner_usage.InvalidPeriod:
        ok("future month rejected")

    label, window_start, window_end = partner_usage.parse_period(None, now=datetime(2026, 10, 22, tzinfo=timezone.utc))
    check(label == "2026-10" and window_start.day == 1 and window_end == datetime(2026, 11, 1, tzinfo=timezone.utc),
          "omitted period is the UTC month, end exclusive")


def require_ledger():
    try:
        get_client().table("cost_events").select("id").limit(1).execute()
    except Exception as exc:
        print(f"{YELLOW}cost_events is missing. Apply db/migrations/001_cost_events.sql{RESET}")
        print(f"  ({str(exc)[:140]})")
        raise SystemExit(2)


def request_id_ready() -> bool:
    db = get_client()
    try:
        db.table("cost_events").select("request_id").limit(1).execute()
        db.table("api_usage").select("request_id").limit(1).execute()
        return True
    except Exception as exc:
        if "request_id" in str(exc).lower():
            print(f"{YELLOW}request_id column missing — month total only. Apply 004_usage_request_id.sql for lines.{RESET}")
            return False
        raise


def provision(name):
    db = get_client()
    tid = db.table("tenants").insert({"name": name}).execute().data[0]["id"]
    raw = "sk_smoke_" + secrets.token_urlsafe(24)
    key_id = db.table("api_keys").insert({
        "tenant_id": tid,
        "key_hash": hashlib.sha256(raw.encode()).hexdigest(),
    }).execute().data[0]["id"]
    return tid, raw, key_id


def add_cost(tenant_id, event_type, usd, request_id=None, debate_id=None, needs_repricing=False):
    row = {
        "tenant_id": tenant_id,
        "event_type": event_type,
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
        "input_tokens": 1,
        "output_tokens": 1,
        "vendor_cost_usd": usd,
        "price_version": "smoke",
        "needs_repricing": needs_repricing,
    }
    if debate_id:
        row["debate_id"] = debate_id
    if request_id:
        row["request_id"] = request_id
    get_client().table("cost_events").insert(row).execute()


def add_call(tenant_id, key_id, endpoint, status, request_id=None):
    row = {
        "tenant_id": tenant_id,
        "api_key_id": key_id,
        "endpoint": endpoint,
        "response_status": status,
        "latency_ms": 10,
    }
    if request_id:
        row["request_id"] = request_id
    get_client().table("api_usage").insert(row).execute()


def database_checks():
    require_ledger()
    lines = request_id_ready()
    db = get_client()
    a_id, a_key, a_key_id = provision(f"SMOKE Usage A {secrets.token_hex(3)}")
    b_id, b_key, _b_key_id = provision(f"SMOKE Usage B {secrets.token_hex(3)}")
    debate_id = "11111111-1111-1111-1111-111111111111"
    r1 = "22222222-2222-2222-2222-222222222222"
    r2 = "33333333-3333-3333-3333-333333333333"
    client = TestClient(app)

    try:
        # 1.23456789 + 0.00000001 + 0 + 0.50 + 0.25 = 1.98456790
        if lines:
            add_cost(a_id, "ai_speech_pass1", "1.23456789", request_id=r1, debate_id=debate_id)
            add_cost(a_id, "ai_speech_pass2", "0.00000001", request_id=r1, debate_id=debate_id)
            add_cost(a_id, "moderation", "0", request_id=r1, debate_id=debate_id)
            add_cost(a_id, "score", "0.50000000", request_id=r2)
            add_call(a_id, a_key_id, "POST /api/v1/debates/{id}/turns", 200, request_id=r1)
            add_call(a_id, a_key_id, "POST /api/v1/debates/{id}/finish", 200, request_id=r2)
        else:
            add_cost(a_id, "ai_speech_pass1", "1.23456789")
            add_cost(a_id, "ai_speech_pass2", "0.00000001")
            add_cost(a_id, "moderation", "0")
            add_cost(a_id, "score", "0.50000000")
        add_cost(a_id, "ai_speech", "0.25000000")  # no request id either way
        add_call(a_id, a_key_id, "POST /api/v1/debates", 201)
        add_cost(b_id, "ai_speech", "9.00000000")

        statement = partner_usage.build_statement(a_id)
        check(statement["amount_due_usd"] == "1.98456790",
              f"tenant A amount due is the exact sum ({statement['amount_due_usd']})")
        kinds = {row["kind"]: row["amount_usd"] for row in statement["spend_by_kind"]}
        check("ai_speech_pass1" not in kinds, "statement does not expose pass names")
        check(kinds.get("speech") == "1.48456790", f"speech kind is both passes plus the unstamped speech ({kinds})")
        check(kinds.get("score") == "0.50000000", "score kind stands alone")
        check(kinds.get("moderation") == "0.00000000", "a zero-cost safety check is present and zero")
        other = partner_usage.build_statement(b_id)
        check(other["amount_due_usd"] == "9.00000000", "tenant B sees only its own charge")

        if lines:
            check(statement["per_call_available"] is True, "per-call lines available")
            check(statement["amount_without_call_id_usd"] == "0.25000000",
                  f"unstamped remainder ({statement['amount_without_call_id_usd']})")
            check(statement["requests_with_call_id"] == 2, "two stamped HTTP calls")
            page = partner_usage.list_events(a_id, limit=50, offset=0)
            by_id = {row["request_id"]: row for row in page["events"]}
            check(r1 in by_id and r2 in by_id, "both stamped calls are lines")
            check(by_id[r1]["amount_usd"] == "1.23456790",
                  f"one turn sums both speech passes ({by_id[r1]['amount_usd']})")
            check(by_id[r1]["debate_id"] == debate_id, "line carries the debate id")
            check(all(k["kind"] != "ai_speech_pass1" for k in by_id[r1]["spend_by_kind"]),
                  "line kinds do not name passes")
            check(by_id[r2]["amount_usd"] == "0.50000000", "score call is its own line")
            line_sum = sum((Decimal(row["amount_usd"]) for row in page["events"]), Decimal(0))
            check(line_sum + Decimal(statement["amount_without_call_id_usd"]) == Decimal(statement["amount_due_usd"]),
                  "event lines plus the unstamped remainder equal the amount due")
        else:
            check(statement["per_call_available"] is False, "lines reported unavailable without the column")
            check(statement["amount_without_call_id_usd"] is None, "unstamped split is omitted until the column exists")

        hdr = {"X-API-Key": a_key}
        missing = client.get("/api/v1/usage")
        check(missing.status_code == 401, f"no key is 401 (got {missing.status_code})")
        bad = client.get("/api/v1/usage", headers=hdr, params={"period": "2026-13"})
        check(bad.status_code == 400, f"bad period is 400 (got {bad.status_code})")
        future = client.get("/api/v1/usage", headers=hdr, params={"period": "2099-01"})
        check(future.status_code == 400, f"future period is 400 (got {future.status_code})")

        got = client.get("/api/v1/usage", headers=hdr)
        check(got.status_code == 200, f"usage 200 (got {got.status_code})")
        check(got.json()["amount_due_usd"] == "1.98456790", "HTTP body matches the exact sum")
        check(got.json()["currency"] == "USD", "currency is USD")
        body = str(got.json())
        check("ai_speech_pass" not in body and "claude-" not in body, "HTTP body has no pipeline or model names")

        other_http = client.get("/api/v1/usage", headers={"X-API-Key": b_key})
        check(other_http.status_code == 200 and other_http.json()["amount_due_usd"] == "9.00000000",
              "other tenant's HTTP statement is only their charge")

        events = client.get("/api/v1/usage/events", headers=hdr)
        if lines:
            check(events.status_code == 200, f"events 200 (got {events.status_code})")
            amounts = {row["request_id"]: row["amount_usd"] for row in events.json()["events"]}
            check(amounts.get(r1) == "1.23456790" and amounts.get(r2) == "0.50000000",
                  f"HTTP lines match ({amounts})")
        else:
            check(events.status_code == 503, f"events 503 until migration 004 (got {events.status_code})")

        # A capped tenant can still read the statement. Skip when migration 003
        # is not applied — the month total does not depend on the cap column.
        try:
            db.table("tenants").update({"monthly_cost_limit_usd": "0.01"}).eq("id", a_id).execute()
        except Exception as exc:
            print(f"{YELLOW}cap column missing, skipping the over-cap read ({str(exc)[:80]}){RESET}")
        else:
            capped = client.get("/api/v1/usage", headers=hdr)
            check(capped.status_code == 200 and capped.json()["amount_due_usd"] == "1.98456790",
                  "a tenant over the cap can still read what they owe")

        print(f"\n{GREEN}All {PASSED} assertions passed.{RESET}")
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        fail("unexpected exception")
    finally:
        for tid in (a_id, b_id):
            try:
                db.table("cost_events").delete().eq("tenant_id", tid).execute()
            except Exception as exc:
                print(f"[cleanup] cost_events {exc}")
            try:
                db.table("tenants").delete().eq("id", tid).execute()
            except Exception as exc:
                print(f"[cleanup] tenant {exc}")


def main():
    pure_checks()
    try:
        database_checks()
    except SystemExit as exc:
        if exc.code == 2:
            print(f"\n{YELLOW}Pure checks passed ({PASSED}). Database checks skipped.{RESET}")
            raise
        raise


if __name__ == "__main__":
    main()
