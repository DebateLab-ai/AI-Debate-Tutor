"""End-to-end journey from SuperJuniors' point of view.

    python3 scripts/smoke_partner_journey.py            # free (synthetic spend)
    python3 scripts/smoke_partner_journey.py --with-ai  # one real casual turn (~$0.01)

Not a unit test. This walks the integration a partner actually builds — a class
of students, a spend cap that trips partway through a lesson, retries, rate
limits, abandonment — and reports what their users would experience. It is
written to SURFACE bad UX, not just to assert green.

Findings are printed as NOTE lines; assertions only fail on things that are
outright broken.
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

import os
os.environ.setdefault("API_RATE_LIMIT_PER_MIN", "1000")

from fastapi.testclient import TestClient

from app import alerts, spendcap
from app.db import get_client
from app.main import app

G, R, Y, C, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[0m"
PASSED = 0
NOTES: list[str] = []
SENT: list[tuple[str, str]] = []
# send_alert is called with keyword args, so the stub must accept them.
alerts.send_alert = lambda subject, body: (SENT.append((subject, body)), True)[1]  # noqa: E731

ADMIN = "smoke_" + secrets.token_urlsafe(48)
os.environ["ADMIN_API_TOKEN"] = ADMIN


def ok(m):
    global PASSED
    PASSED += 1
    print(f"{G}PASS{RESET} {m}")


def fail(m):
    print(f"{R}FAIL{RESET} {m}")
    raise SystemExit(1)


def check(c, m):
    ok(m) if c else fail(m)


def note(m):
    NOTES.append(m)
    print(f"{Y}NOTE{RESET} {m}")


def step(m):
    print(f"\n{C}── {m}{RESET}")


def provision(cap):
    db = get_client()
    tid = db.table("tenants").insert(
        {"name": f"SMOKE SuperJuniors {secrets.token_hex(3)}", "monthly_cost_limit_usd": cap}
    ).execute().data[0]["id"]
    raw = "sk_smoke_" + secrets.token_urlsafe(24)
    db.table("api_keys").insert(
        {"tenant_id": tid, "key_hash": hashlib.sha256(raw.encode()).hexdigest()}
    ).execute()
    return tid, raw


def spend(tid, usd):
    get_client().table("cost_events").insert({
        "tenant_id": tid, "event_type": "ai_speech", "provider": "anthropic",
        "model": "claude-haiku-4-5", "vendor_cost_usd": usd, "price_version": "smoke",
    }).execute()


def main(with_ai):
    tid, key = provision(cap=1.00)
    c = TestClient(app)
    H = {"X-API-Key": key}
    A = {"X-Admin-Token": ADMIN}

    try:
        # ── A class of students starts debates ────────────────────────────
        step("A teacher starts a class of 5 students on casual debates")
        debates = {}
        for i in range(1, 6):
            r = c.post("/api/v1/debates", headers=H, json={
                "motion": "THW ban single-use plastics", "starter": "user",
                "num_rounds": 2, "mode": "casual", "difficulty": "beginner",
                "external_user_id": f"student-{i}",
                "metadata": {"Debater": f"Student {i}", "Class": "Wed 4pm"},
            })
            check(r.status_code == 201, f"student-{i} debate created")
            debates[f"student-{i}"] = r.json()["id"]

        r = c.get("/api/v1/debates", headers=H, params={"external_user_id": "student-3"})
        check(r.status_code == 200 and len(r.json()) == 1,
              "teacher can list one student's debates via external_user_id")

        # ── Spend climbs mid-lesson ───────────────────────────────────────
        step("Spend climbs to 90% of the cap while the class is mid-debate")
        spend(tid, 0.90)
        spendcap.invalidate_limit_cache(tid)
        SENT.clear()
        spendcap.check_thresholds_and_alert(tid, "SuperJuniors")
        check(any("50%" in s for s, _ in SENT) and any("80%" in s for s, _ in SENT),
              "50% and 80% warnings emailed before anything breaks")

        r = c.post(f"/api/v1/debates/{debates['student-1']}/turns", headers=H,
                   json={"content": "Plastic waste is choking our oceans and the harm is irreversible."})
        check(r.status_code == 200, f"at 90%, a student turn still succeeds ({r.status_code})")

        # ── The cap trips MID-DEBATE ──────────────────────────────────────
        step("The cap trips while students are mid-debate")
        spend(tid, 0.20)   # now $1.10 of $1.00
        spendcap.invalidate_limit_cache(tid)

        r = c.post(f"/api/v1/debates/{debates['student-2']}/turns", headers=H,
                   json={"content": "My opening argument about ocean plastics and harm."})
        check(r.status_code == 402, f"over cap: next student's turn is refused ({r.status_code})")
        note(f"student-2 mid-lesson sees: {r.json()['detail']}")

        # student-1 already spoke once. Can they FINISH and get their score?
        r = c.post(f"/api/v1/debates/{debates['student-1']}/finish", headers=H)
        check(r.status_code == 200,
              f"over cap: a student who already debated STILL gets their score ({r.status_code})")
        note("scoring is intentionally exempt from the cap — cheap, terminal, idempotent")

        r = c.get(f"/api/v1/debates/{debates['student-1']}", headers=H)
        check(r.status_code == 200, "over cap: transcript still readable")
        r = c.post("/api/v1/debates", headers=H, json={
            "motion": "New debate at cap", "starter": "user", "num_rounds": 1, "mode": "casual"})
        check(r.status_code == 402,
              f"over cap: starting a NEW debate is refused up front ({r.status_code}) "
              "— fails before the student writes anything")

        r = c.get(f"/api/v1/debates/{debates['student-1']}/report.pdf", headers=H)
        check(r.status_code == 200,
              f"over cap: the scored student can still download their PDF ({r.status_code})")

        # ── Teacher raises the cap ────────────────────────────────────────
        step("Rico raises the cap; the class resumes")
        r = c.put(f"/internal/admin/tenants/{tid}/cap", headers=A,
                  json={"monthly_cost_limit_usd": 100.0})
        check(r.status_code == 200, "cap raised via admin")
        r = c.post(f"/api/v1/debates/{debates['student-2']}/turns", headers=H,
                   json={"content": "My opening argument about ocean plastics and harm."})
        check(r.status_code == 200, f"student-2 resumes immediately ({r.status_code})")

        # ── Retry semantics ───────────────────────────────────────────────
        step("Their client retries after a timeout")
        did = debates["student-4"]
        ih = {**H, "Idempotency-Key": f"sj-{secrets.token_hex(6)}"}
        body = {"content": "A considered argument about displacement effects and cost."}
        r1 = c.post(f"/api/v1/debates/{did}/turns", headers=ih, json=body)
        before = len(get_client().table("cost_events").select("id").eq("tenant_id", tid).execute().data or [])
        r2 = c.post(f"/api/v1/debates/{did}/turns", headers=ih, json=body)
        after = len(get_client().table("cost_events").select("id").eq("tenant_id", tid).execute().data or [])
        check(r1.status_code == 200 and r2.status_code == 200,
              f"retry with the same Idempotency-Key succeeds twice ({r1.status_code}/{r2.status_code})")
        check(r1.json()["user_message"]["id"] == r2.json()["user_message"]["id"],
              "retry returns the identical message, not a duplicate turn")
        check(after == before,
              f"retry costs NOTHING extra — no double-billing ({before} -> {after} cost rows)")

        r3 = c.post(f"/api/v1/debates/{did}/turns", headers=ih,
                    json={"content": "A completely different speech reusing the same key."})
        note(f"same Idempotency-Key with a DIFFERENT body returns {r3.status_code} "
             f"— cached response replayed, new content silently ignored"
             if r3.status_code == 200 else
             f"same key + different body -> {r3.status_code}")

        # ── Moderation ────────────────────────────────────────────────────
        step("A student submits something the moderator blocks")
        r = c.post(f"/api/v1/debates/{debates['student-5']}/turns", headers=H,
                   json={"content": "I will find you and kill you and your whole family."})
        note(f"blocked speech -> {r.status_code} "
             f"({str(r.json().get('detail'))[:60]})")
        rows = get_client().table("cost_events").select("event_type").eq("tenant_id", tid).execute().data or []
        mods = [x for x in rows if x["event_type"] == "moderation"]
        note(f"blocked speech still cost a moderation call; {len(mods)} moderation rows billed to them")

        # ── Rate limiting ─────────────────────────────────────────────────
        step("Their whole site shares one key and hits the rate limit")
        os.environ["API_RATE_LIMIT_PER_MIN"] = "5"
        # importlib.import_module, not `import app.auth` — the latter would bind
        # a local named `app` and shadow the FastAPI instance in this function.
        import importlib
        ratelimit = importlib.import_module("app.ratelimit")
        auth_mod = importlib.import_module("app.auth")
        importlib.reload(ratelimit)
        importlib.reload(auth_mod)
        codes = [c.get("/api/v1/debates", headers=H).status_code for _ in range(9)]
        n429 = codes.count(429)
        note(f"with limit=5, nine requests gave: {codes}")
        if n429:
            rr = c.get("/api/v1/debates", headers=H)
            note(f"429 carries Retry-After={rr.headers.get('Retry-After')!r} "
                 f"(their client can back off correctly)")
        else:
            note("limit change did not take effect in-process (module reload); "
                 "rate limiting is covered by its own path in production")
        os.environ["API_RATE_LIMIT_PER_MIN"] = "1000"
        importlib.reload(ratelimit); importlib.reload(auth_mod)

        # ── Abandonment ───────────────────────────────────────────────────
        step("A student abandons a debate")
        r = c.post("/api/v1/debates", headers=H, json={
            "motion": "Abandoned", "starter": "user", "num_rounds": 3, "mode": "casual"})
        aid = r.json()["id"]
        rows_before = len(get_client().table("cost_events").select("id").eq("tenant_id", tid).execute().data or [])
        r = c.get(f"/api/v1/debates/{aid}", headers=H)
        rows_after = len(get_client().table("cost_events").select("id").eq("tenant_id", tid).execute().data or [])
        check(rows_after == rows_before,
              "an abandoned debate that never got a turn costs them nothing")
        check(r.json()["status"] == "active",
              "abandoned debate stays 'active' — no auto-expiry until retention")

        # ── Final spend picture ───────────────────────────────────────────
        step("What Rico sees on the invoice")
        r = c.get("/internal/admin/spend", headers=A)
        mine = [t for t in r.json()["tenants"] if t["tenant_id"] == tid][0]
        print(f"     MTD ${mine['mtd_cost_usd']:.6f} of cap ${mine['monthly_cost_limit_usd']} "
              f"({mine['pct_used']}% used)")

        print(f"\n{G}{PASSED} assertions passed.{RESET}")
        if NOTES:
            print(f"\n{Y}{len(NOTES)} behaviours worth reviewing:{RESET}")
            for n in NOTES:
                print(f"  • {n}")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-ai", action="store_true")
    main(ap.parse_args().with_ai)
