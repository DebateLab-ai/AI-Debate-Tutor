"""Phase 1 smoke test: confirm tenant scoping in debates_store.

Run from backend/:
    source venv/bin/activate
    python scripts/smoke_tenant_scoping.py

Requires SUPABASE_URL and SUPABASE_SERVICE_KEY in backend/.env. Creates two
disposable tenants, exercises the store, and deletes them on success (cascade
removes their debates/messages/scores).
"""

import sys
import traceback
from pathlib import Path

# Run as a script from backend/ so `app.*` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app.db import get_client
from app import debates_store


GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}PASS{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}FAIL{RESET} {msg}")
    raise SystemExit(1)


def main() -> None:
    db = get_client()

    # 1. Create two disposable tenants.
    tenants = db.table("tenants").insert([
        {"name": "smoke-test-A"},
        {"name": "smoke-test-B"},
    ]).execute()
    tenant_a = tenants.data[0]["id"]
    tenant_b = tenants.data[1]["id"]
    print(f"[setup] tenant A = {tenant_a}")
    print(f"[setup] tenant B = {tenant_b}")

    try:
        # 2. Tenant A creates a debate.
        debate = debates_store.create_debate(
            tenant_id=tenant_a,
            motion="THW ban single-use plastics",
            starter="user",
            num_rounds=2,
            mode="casual",
            difficulty="intermediate",
            external_user_id="student-123",
            metadata={"Debater": "Nguyen An", "Class": "Wednesdays"},
        )
        debate_id = debate["id"]
        ok(f"created debate {debate_id} for tenant A")

        # 3. Tenant A can read it.
        got_a = debates_store.get_debate(tenant_id=tenant_a, debate_id=debate_id)
        if got_a is None:
            fail("tenant A could not read its own debate")
        if got_a["metadata"].get("Debater") != "Nguyen An":
            fail(f"metadata didn't round-trip: {got_a['metadata']}")
        ok("tenant A reads its own debate (metadata round-tripped)")

        # 4. Tenant B CANNOT read it — this is the load-bearing check.
        got_b = debates_store.get_debate(tenant_id=tenant_b, debate_id=debate_id)
        if got_b is not None:
            fail(f"SECURITY: tenant B read tenant A's debate! Got: {got_b}")
        ok("tenant B cannot read tenant A's debate (returns None)")

        # 5. list_debates is also tenant-scoped.
        a_list = debates_store.list_debates(tenant_id=tenant_a)
        b_list = debates_store.list_debates(tenant_id=tenant_b)
        if not any(d["id"] == debate_id for d in a_list):
            fail("debate missing from tenant A's list")
        if any(d["id"] == debate_id for d in b_list):
            fail("SECURITY: debate appeared in tenant B's list")
        ok(f"list_debates scoped correctly (A={len(a_list)}, B={len(b_list)})")

        # 6. external_user_id filter works.
        filtered = debates_store.list_debates(tenant_id=tenant_a, external_user_id="student-123")
        if not any(d["id"] == debate_id for d in filtered):
            fail("external_user_id filter missed the debate")
        none_filtered = debates_store.list_debates(tenant_id=tenant_a, external_user_id="student-nope")
        if any(d["id"] == debate_id for d in none_filtered):
            fail("external_user_id filter returned wrong row")
        ok("external_user_id filter works")

        # 7. Messages are tenant-scoped.
        debates_store.append_message(
            tenant_id=tenant_a,
            debate_id=debate_id,
            round_no=1,
            speaker="user",
            content="Plastic waste is a global crisis.",
        )
        a_msgs = debates_store.list_messages(tenant_id=tenant_a, debate_id=debate_id)
        b_msgs = debates_store.list_messages(tenant_id=tenant_b, debate_id=debate_id)
        if len(a_msgs) != 1:
            fail(f"tenant A expected 1 message, got {len(a_msgs)}")
        if b_msgs:
            fail(f"SECURITY: tenant B saw {len(b_msgs)} of tenant A's messages")
        ok("messages scoped correctly")

        # 8. update_debate_state refuses cross-tenant writes.
        bad_update = debates_store.update_debate_state(
            tenant_id=tenant_b,
            debate_id=debate_id,
            current_round=99,
            next_speaker="assistant",
            status="completed",
        )
        if bad_update is not None:
            fail(f"SECURITY: tenant B updated tenant A's debate: {bad_update}")
        still_a = debates_store.get_debate(tenant_id=tenant_a, debate_id=debate_id)
        if still_a["current_round"] != 1 or still_a["status"] != "active":
            fail(f"tenant A's debate state was mutated: {still_a}")
        ok("update_debate_state refuses cross-tenant writes")

        # 9. Metadata validation rejects oversized bags.
        try:
            debates_store.create_debate(
                tenant_id=tenant_a,
                motion="x",
                starter="user",
                num_rounds=1,
                metadata={f"k{i}": "v" for i in range(20)},
            )
            fail("metadata cap not enforced (too many fields accepted)")
        except debates_store.MetadataError:
            ok("metadata field-count cap enforced")

        try:
            debates_store.create_debate(
                tenant_id=tenant_a,
                motion="x",
                starter="user",
                num_rounds=1,
                metadata={"nested": {"bad": "value"}},
            )
            fail("metadata cap not enforced (nested object accepted)")
        except debates_store.MetadataError:
            ok("metadata nested-object rejection enforced")

        print(f"\n{GREEN}All checks passed.{RESET} Phase 1 tenant scoping is sound.")

    finally:
        # Cascade deletes their debates/messages/scores too.
        print("\n[teardown] removing test tenants...")
        db.table("tenants").delete().eq("id", tenant_a).execute()
        db.table("tenants").delete().eq("id", tenant_b).execute()
        print("[teardown] done")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
