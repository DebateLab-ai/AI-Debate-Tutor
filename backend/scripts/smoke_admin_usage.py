"""Smoke test for the private /internal/admin/* usage endpoints.

Run from backend/ (requires backend/.env with SUPABASE_URL and SUPABASE_SERVICE_KEY):
    python3 scripts/smoke_admin_usage.py

Free — no AI calls. Provisions a disposable tenant + key, drives two real
/api/v1/* requests so api_usage rows exist, then asserts the admin surface
reads them back and is closed to everyone else. Cleans up in a finally block.
"""

import hashlib
import os
import secrets
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Set before app import so the admin router sees a configured token regardless
# of what the local .env holds.
ADMIN_TOKEN = "smoke_" + secrets.token_urlsafe(48)
os.environ["ADMIN_API_TOKEN"] = ADMIN_TOKEN

from fastapi.testclient import TestClient

from app.db import get_client
from app.main import app

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
PASSED = 0


def ok(msg: str) -> None:
    global PASSED
    PASSED += 1
    print(f"{GREEN}PASS{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}FAIL{RESET} {msg}")
    raise SystemExit(1)


def check(cond: bool, msg: str) -> None:
    ok(msg) if cond else fail(msg)


def provision_key(tenant_name: str) -> tuple[str, str]:
    db = get_client()
    tenant = db.table("tenants").insert({"name": tenant_name}).execute()
    tenant_id = tenant.data[0]["id"]
    raw_key = "sk_smoke_" + secrets.token_urlsafe(24)
    db.table("api_keys").insert({
        "tenant_id": tenant_id,
        "key_hash": hashlib.sha256(raw_key.encode()).hexdigest(),
    }).execute()
    return tenant_id, raw_key


def cleanup(tenant_id: str) -> None:
    # api_usage/api_keys cascade from tenants (ON DELETE CASCADE in schema.sql).
    try:
        get_client().table("tenants").delete().eq("id", tenant_id).execute()
    except Exception as e:
        print(f"[cleanup] tenant {tenant_id}: {e}")


def main() -> None:
    client = TestClient(app)
    admin = {"X-Admin-Token": ADMIN_TOKEN}
    tenant_id, raw_key = provision_key(f"SMOKE Admin Usage {secrets.token_hex(4)}")

    try:
        # --- Generate usage rows through the real partner surface -------------
        r = client.post(
            "/api/v1/debates",
            headers={"X-API-Key": raw_key},
            json={"motion": "Smoke motion", "starter": "user", "num_rounds": 1},
        )
        check(r.status_code == 201, f"partner create returned 201 (got {r.status_code})")
        client.get("/api/v1/debates", headers={"X-API-Key": raw_key})

        # usage.py writes via BackgroundTasks, which TestClient runs before the
        # response is handed back, so the rows are already committed here.

        # --- Auth gating ------------------------------------------------------
        check(client.get("/internal/admin/usage").status_code == 401,
              "no admin token -> 401")
        check(client.get("/internal/admin/usage", headers={"X-Admin-Token": "wrong"}).status_code == 401,
              "bad admin token -> 401")
        check(client.get("/internal/admin/usage", headers={"X-API-Key": raw_key}).status_code == 401,
              "a valid PARTNER key does not open the admin surface")

        # --- The admin surface is invisible to partners -----------------------
        schema = client.get("/openapi.json").json()
        check(not any(p.startswith("/internal") for p in schema.get("paths", {})),
              "/internal/* is absent from the OpenAPI schema")

        # --- Raw log read -----------------------------------------------------
        r = client.get("/internal/admin/usage", headers=admin, params={"tenant_id": tenant_id})
        check(r.status_code == 200, f"admin usage returned 200 (got {r.status_code})")
        body = r.json()
        check(body["count"] >= 2, f"logged both partner calls (count={body['count']})")
        row = body["rows"][0]
        check({"endpoint", "response_status", "created_at", "tenant_name"} <= set(row),
              "rows carry endpoint, status, timestamp and tenant name")
        check(all(str(x["tenant_id"]) == tenant_id for x in body["rows"]),
              "tenant_id filter returns only that tenant")

        # --- Filters ----------------------------------------------------------
        r = client.get("/internal/admin/usage", headers=admin,
                       params={"tenant_id": tenant_id, "endpoint": "POST /api/v1/debates"})
        check(r.status_code == 200 and all(
            x["endpoint"] == "POST /api/v1/debates" for x in r.json()["rows"]),
            "endpoint filter narrows to one route")

        r = client.get("/internal/admin/usage", headers=admin,
                       params={"tenant_id": tenant_id, "errors_only": "true"})
        check(r.status_code == 200 and all(
            x["response_status"] >= 400 for x in r.json()["rows"]),
            "errors_only returns no 2xx rows")

        r = client.get("/internal/admin/usage", headers=admin,
                       params={"tenant_id": tenant_id, "limit": 1})
        check(r.status_code == 200 and len(r.json()["rows"]) == 1 and r.json()["next_offset"] == 1,
              "limit + next_offset paginate correctly")

        check(client.get("/internal/admin/usage", headers=admin,
                         params={"status_min": 500, "status_max": 400}).status_code == 400,
              "inverted status range -> 400")

        # --- Summary ----------------------------------------------------------
        r = client.get("/internal/admin/usage/summary", headers=admin,
                       params={"tenant_id": tenant_id, "days": 1})
        check(r.status_code == 200, f"summary returned 200 (got {r.status_code})")
        s = r.json()
        check(s["totals"]["requests"] >= 2, "summary totals count the calls")
        check(s["truncated"] is False, "summary not truncated at this volume")
        check(len(s["by_endpoint"]) >= 1 and len(s["by_day"]) >= 1,
              "summary breaks down by endpoint and by day")
        check(any(t["tenant_id"] == tenant_id for t in s["by_tenant"]),
              "summary by_tenant includes the smoke tenant")
        check("latency_ms_p95" in s["totals"], "summary reports latency percentiles")

        check(client.get("/internal/admin/usage/summary", headers=admin,
                         params={"days": 999}).status_code == 422,
              "over-long window rejected by validation")

        # --- Tenants ----------------------------------------------------------
        r = client.get("/internal/admin/tenants", headers=admin)
        check(r.status_code == 200, f"tenants returned 200 (got {r.status_code})")
        mine = [t for t in r.json()["tenants"] if t["id"] == tenant_id]
        check(len(mine) == 1 and mine[0]["api_keys_total"] == 1,
              "tenant listing reports the key count")
        check(not any("key_hash" in t for t in r.json()["tenants"]),
              "tenant listing never exposes key hashes")

        # --- Disabled when unconfigured ---------------------------------------
        os.environ.pop("ADMIN_API_TOKEN")
        try:
            check(client.get("/internal/admin/usage", headers=admin).status_code == 404,
                  "unset ADMIN_API_TOKEN -> 404 (surface disabled, not just locked)")
            os.environ["ADMIN_API_TOKEN"] = "tooshort"
            check(client.get("/internal/admin/usage",
                             headers={"X-Admin-Token": "tooshort"}).status_code == 503,
                  "token below the minimum length is refused")
        finally:
            os.environ["ADMIN_API_TOKEN"] = ADMIN_TOKEN

        print(f"\n{GREEN}All {PASSED} assertions passed.{RESET}")
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        fail("unexpected exception")
    finally:
        cleanup(tenant_id)


if __name__ == "__main__":
    main()
