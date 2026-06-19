"""Phase 2 smoke test: full lifecycle through the public /api/v1/* endpoints.

Run from backend/ (requires backend/.env with SUPABASE_URL and SUPABASE_SERVICE_KEY):
    pip install -r requirements.txt
    python3 scripts/smoke_api_v1_lifecycle.py            # auth + create + list + tenant scoping (no AI calls)
    python scripts/smoke_api_v1_lifecycle.py --with-ai  # also exercises turns + scoring (calls OpenAI, costs ~$0.01)

Uses FastAPI TestClient so no uvicorn needed. Creates a disposable tenant and
API key, runs assertions, and cleans up in a finally block.
"""

import argparse
import hashlib
import secrets
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi.testclient import TestClient

from app.db import get_client
from app.main import app


GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}PASS{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}FAIL{RESET} {msg}")
    raise SystemExit(1)


def provision_key(tenant_name: str) -> tuple[str, str]:
    """Insert a tenant and api_key row, return (tenant_id, raw_key)."""
    db = get_client()
    tenant = db.table("tenants").insert({"name": tenant_name}).execute()
    tenant_id = tenant.data[0]["id"]
    raw_key = "sk_smoke_" + secrets.token_urlsafe(24)
    db.table("api_keys").insert({
        "tenant_id": tenant_id,
        "key_hash": hashlib.sha256(raw_key.encode()).hexdigest(),
    }).execute()
    return tenant_id, raw_key


def teardown_tenant(tenant_id: str) -> None:
    get_client().table("tenants").delete().eq("id", tenant_id).execute()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-ai", action="store_true", help="Exercise turns + scoring (calls OpenAI)")
    args = parser.parse_args()

    tenant_a_id, key_a = provision_key("smoke-api-A")
    tenant_b_id, key_b = provision_key("smoke-api-B")
    print(f"[setup] tenant A = {tenant_a_id}, key starts {key_a[:14]}...")
    print(f"[setup] tenant B = {tenant_b_id}, key starts {key_b[:14]}...")

    headers_a = {"X-API-Key": key_a}
    headers_b = {"X-API-Key": key_b}

    try:
        with TestClient(app) as client:
            # 1. Missing key → 401.
            r = client.post("/api/v1/debates", json={
                "motion": "x", "starter": "user", "num_rounds": 1,
            })
            if r.status_code != 401:
                fail(f"expected 401 without key, got {r.status_code}: {r.text}")
            ok("missing X-API-Key → 401")

            # 2. Create a debate as tenant A.
            create_payload = {
                "motion": "THW ban single-use plastics",
                "starter": "user",
                "num_rounds": 2,
                "mode": "casual",
                "difficulty": "intermediate",
                "external_user_id": "student-123",
                "metadata": {"Debater": "Nguyen An", "Class": "Wednesdays"},
            }
            r = client.post("/api/v1/debates", json=create_payload, headers=headers_a)
            if r.status_code != 201:
                fail(f"create failed: {r.status_code} {r.text}")
            debate = r.json()
            debate_id = debate["id"]
            if debate["metadata"].get("Debater") != "Nguyen An":
                fail(f"metadata didn't round-trip: {debate['metadata']}")
            if debate["external_user_id"] != "student-123":
                fail(f"external_user_id didn't round-trip: {debate}")
            ok(f"tenant A created debate {debate_id[:8]}... (metadata + external_user_id round-tripped)")

            # 3. Tenant B cannot read it.
            r = client.get(f"/api/v1/debates/{debate_id}", headers=headers_b)
            if r.status_code != 404:
                fail(f"SECURITY: tenant B got status {r.status_code} reading tenant A's debate")
            ok("tenant B reading tenant A's debate → 404")

            # 4. Tenant A can read it; messages and score start empty.
            r = client.get(f"/api/v1/debates/{debate_id}", headers=headers_a)
            if r.status_code != 200:
                fail(f"tenant A read failed: {r.status_code} {r.text}")
            full = r.json()
            if full["messages"] != [] or full["score"] is not None:
                fail(f"new debate should have no messages/score: {full}")
            ok("tenant A reads its debate (messages=[], score=null)")

            # 5. List scoping: A sees it, B doesn't.
            r_a = client.get("/api/v1/debates", headers=headers_a).json()
            r_b = client.get("/api/v1/debates", headers=headers_b).json()
            if not any(d["id"] == debate_id for d in r_a):
                fail("debate missing from tenant A's list")
            if any(d["id"] == debate_id for d in r_b):
                fail("SECURITY: tenant B's list contained tenant A's debate")
            ok(f"list scoped correctly (A={len(r_a)}, B={len(r_b)})")

            # 6. external_user_id filter.
            r = client.get("/api/v1/debates?external_user_id=student-123", headers=headers_a).json()
            if not any(d["id"] == debate_id for d in r):
                fail("external_user_id filter missed the debate")
            r = client.get("/api/v1/debates?external_user_id=student-nope", headers=headers_a).json()
            if any(d["id"] == debate_id for d in r):
                fail("external_user_id filter returned wrong row")
            ok("external_user_id filter works")

            # 7. Oversized metadata → 400.
            big_meta = {f"k{i}": "v" for i in range(20)}
            r = client.post(
                "/api/v1/debates",
                json={**create_payload, "metadata": big_meta},
                headers=headers_a,
            )
            if r.status_code != 400:
                fail(f"oversized metadata should 400, got {r.status_code}: {r.text}")
            ok("oversized metadata → 400")

            # 8. Cross-tenant turn rejection.
            r = client.post(
                f"/api/v1/debates/{debate_id}/turns",
                json={"content": "anything"},
                headers=headers_b,
            )
            if r.status_code != 404:
                fail(f"SECURITY: tenant B turn on tenant A's debate → {r.status_code}")
            ok("tenant B turn on tenant A's debate → 404")

            # 8b. AI-first debates require /open before /turns.
            r = client.post(
                "/api/v1/debates",
                json={**create_payload, "starter": "assistant"},
                headers=headers_a,
            )
            if r.status_code != 201:
                fail(f"assistant-starter create failed: {r.status_code} {r.text}")
            ai_first_id = r.json()["id"]
            r = client.post(
                f"/api/v1/debates/{ai_first_id}/turns",
                json={"content": "too early"},
                headers=headers_a,
            )
            if r.status_code != 409:
                fail(f"expected 409 on /turns before /open, got {r.status_code}: {r.text}")
            if "open" not in r.json().get("detail", "").lower():
                fail(f"409 detail should mention /open: {r.text}")
            ok("assistant-starter /turns before /open → 409 with /open hint")

            if not args.with_ai:
                print(f"\n{GREEN}Auth + CRUD + scoping checks passed.{RESET} Re-run with --with-ai to exercise the AI pipeline.")
                return

            # ---- AI-touching checks (cost a few cents) ----
            print("\n[AI] Submitting a turn (this hits OpenAI, ~5-15s)...")
            r = client.post(
                f"/api/v1/debates/{debate_id}/turns",
                json={"content": "Plastic waste is choking our oceans. We see this with the Great Pacific Garbage Patch. The harm is irreversible and global."},
                headers=headers_a,
            )
            if r.status_code != 200:
                fail(f"turn failed: {r.status_code} {r.text}")
            turn = r.json()
            if not turn["assistant_message"]["content"]:
                fail("AI returned empty content")
            if turn["assistant_message"]["speaker"] != "assistant":
                fail(f"assistant message mislabeled: {turn['assistant_message']}")
            ok(f"AI reply received ({len(turn['assistant_message']['content'])} chars)")

            # 9. Finish + score.
            print("[AI] Scoring (~5-10s)...")
            r = client.post(f"/api/v1/debates/{debate_id}/finish", headers=headers_a)
            if r.status_code != 200:
                fail(f"finish failed: {r.status_code} {r.text}")
            score = r.json()
            if not (0 <= score["overall"] <= 10):
                fail(f"overall score out of range: {score}")
            ok(f"score received (overall={score['overall']}, weakness={score['weakness_type']})")

            # 9b. /finish is idempotent — second call returns cached score without re-scoring.
            r2 = client.post(f"/api/v1/debates/{debate_id}/finish", headers=headers_a)
            if r2.status_code != 200:
                fail(f"second finish failed: {r2.status_code} {r2.text}")
            if r2.json()["overall"] != score["overall"]:
                fail("second /finish returned a different score (should be cached)")
            ok("/finish idempotent (cached score on repeat)")

            # 9c. Idempotency-Key on /turns dedupes retries.
            r = client.post(
                "/api/v1/debates",
                json={**create_payload, "starter": "user", "num_rounds": 1},
                headers=headers_a,
            )
            idem_debate_id = r.json()["id"]
            idem_headers = {**headers_a, "Idempotency-Key": "smoke-turn-key-1"}
            turn_body = {"content": "A short practice argument about plastic pollution and ocean health."}
            r1 = client.post(
                f"/api/v1/debates/{idem_debate_id}/turns",
                json=turn_body,
                headers=idem_headers,
            )
            r_dup = client.post(
                f"/api/v1/debates/{idem_debate_id}/turns",
                json=turn_body,
                headers=idem_headers,
            )
            if r1.status_code != 200 or r_dup.status_code != 200:
                fail(f"idempotent turn failed: {r1.status_code} / {r_dup.status_code}")
            if r1.json()["user_message"]["id"] != r_dup.json()["user_message"]["id"]:
                fail("Idempotency-Key did not return the same user_message id")
            ok("Idempotency-Key on /turns returns cached response")

            # 10. GET after finish includes the score.
            r = client.get(f"/api/v1/debates/{debate_id}", headers=headers_a).json()
            if r["score"] is None:
                fail("GET after finish didn't include score")
            if r["status"] != "completed":
                fail(f"debate status after finish should be 'completed', got {r['status']}")
            ok("GET after finish reflects completed status + score")

            # 11. PDF download — content type + non-empty body + tenant isolation.
            r = client.get(f"/api/v1/debates/{debate_id}/report.pdf", headers=headers_a)
            if r.status_code != 200:
                fail(f"PDF endpoint failed: {r.status_code} {r.text[:200]}")
            if r.headers.get("content-type") != "application/pdf":
                fail(f"PDF content-type wrong: {r.headers.get('content-type')}")
            if not r.content.startswith(b"%PDF-"):
                fail(f"PDF body doesn't start with magic bytes: {r.content[:20]!r}")
            ok(f"PDF downloaded ({len(r.content):,} bytes, type={r.headers['content-type']})")

            # 12. Tenant B can't download tenant A's PDF.
            r = client.get(f"/api/v1/debates/{debate_id}/report.pdf", headers=headers_b)
            if r.status_code != 404:
                fail(f"SECURITY: tenant B got status {r.status_code} on tenant A's PDF")
            ok("tenant B fetching tenant A's PDF → 404")

            # 13. Assistant-starter full open flow.
            print("[AI] Opening assistant-first debate...")
            r = client.post(f"/api/v1/debates/{ai_first_id}/open", headers=headers_a)
            if r.status_code != 200:
                fail(f"/open failed: {r.status_code} {r.text}")
            opened = r.json()
            if not opened["assistant_message"]["content"]:
                fail("AI opening returned empty content")
            if opened["next_speaker"] != "user":
                fail(f"after /open expected next_speaker=user, got {opened}")
            ok(f"assistant-first /open OK ({len(opened['assistant_message']['content'])} chars)")

            # 14. Rebuttal drill start + submit (stateless).
            print("[AI] Rebuttal drill...")
            r = client.post(
                "/api/v1/drills/rebuttal/start",
                headers=headers_a,
                json={
                    "motion": "THW test motion for drills",
                    "user_position": "for",
                    "weakness_type": "rebuttal",
                },
            )
            if r.status_code != 200:
                fail(f"drill /start failed: {r.status_code} {r.text}")
            drill = r.json()
            if not drill.get("claim"):
                fail("drill /start returned empty claim")
            ok(f"drill /start OK ({len(drill['claim'])} char claim)")

            r = client.post(
                "/api/v1/drills/rebuttal/submit",
                headers=headers_a,
                json={
                    "motion": "THW test motion for drills",
                    "claim": drill["claim"],
                    "claim_position": drill["claim_position"],
                    "rebuttal": (
                        "The claim assumes medical plastics cannot be replaced, but many "
                        "hospitals already use reusable sterilizable alternatives without "
                        "compromising safety — the burden is on them to prove a global ban "
                        "would collapse healthcare rather than accelerate innovation."
                    ),
                    "weakness_type": "rebuttal",
                },
            )
            if r.status_code != 200:
                fail(f"drill /submit failed: {r.status_code} {r.text}")
            scored = r.json()
            if scored.get("overall_score") is None:
                fail("drill /submit missing overall_score")
            if not scored.get("next_claim"):
                fail("drill /submit missing next_claim")
            ok(f"drill /submit OK (score={scored['overall_score']})")

            print(f"\n{GREEN}Full lifecycle passed.{RESET} Phase 2 endpoints are sound.")

    finally:
        print("\n[teardown] removing test tenants...")
        teardown_tenant(tenant_a_id)
        teardown_tenant(tenant_b_id)
        print("[teardown] done")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
