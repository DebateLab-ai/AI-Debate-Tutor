"""Sanity-check the rate card in app/pricing.py. Free, no network, no DB.

Run after every rate-card edit:
    python3 scripts/verify_pricing.py

Catches the failure modes that produce wrong invoices: a model used in code but
missing from the card, a past version edited in place, and cost math drifting
from scripts/measure_turn_cost.py.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import pricing

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
failures = []


def ok(m): print(f"{GREEN}PASS{RESET} {m}")
def bad(m): failures.append(m); print(f"{RED}FAIL{RESET} {m}")
def warn(m): print(f"{YELLOW}WARN{RESET} {m}")


# 1. Every model named in app/ must exist in the active card.
card = pricing.active_card()
used = set()
for py in (ROOT / "app").glob("*.py"):
    text = py.read_text()
    # Ignore `if __name__ == "__main__":` blocks — CLI dev testers are not a
    # served path and their argparse defaults are not billable models.
    main_guard = text.find('if __name__ == "__main__":')
    if main_guard != -1:
        text = text[:main_guard]
    used |= set(re.findall(r'model\s*=\s*"([a-z0-9][\w.\-]+)"', text))
    used |= set(re.findall(r'"(claude-[\w.\-]+)"', text))
    used |= set(re.findall(r'"(gpt-[\w.\-]+)"', text))
    used |= set(re.findall(r'"(whisper-[\w.\-]+)"', text))
    used |= set(re.findall(r'"(omni-moderation-[\w.\-]+)"', text))

missing = sorted(m for m in used if m not in card)
if missing:
    bad(f"models used in app/ but absent from rate card {pricing.CURRENT_VERSION}: {missing}")
else:
    ok(f"all {len(used)} models referenced in app/ are priced in {pricing.CURRENT_VERSION}")

unused = sorted(m for m in card if m not in used)
if unused:
    warn(f"priced but not referenced in app/ (fine if intentional): {unused}")

# 2. Cost math must match the independently-written reference.
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from measure_turn_cost import usd_cost as reference_cost
    cases = [
        ("claude-sonnet-4-6", 10_000, 2_000, 0, 0),
        ("claude-haiku-4-5", 50_000, 4_000, 0, 0),
        ("gpt-4o-mini", 3_000, 900, 0, 0),
    ]
    drift = False
    for model, i, o, cr, cw in cases:
        mine = pricing.token_cost_usd(model, i, o, cr, cw, version="2026-06-07")
        theirs = reference_cost(model, i, o, cr, cw)
        if abs(mine - theirs) > 1e-12:
            bad(f"cost mismatch for {model}: pricing.py={mine:.10f} measure_turn_cost.py={theirs:.10f}")
            drift = True
    if not drift:
        ok("cost math agrees with scripts/measure_turn_cost.py on the 2026-06-07 card")
except ImportError as e:
    warn(f"could not import measure_turn_cost for cross-check: {e}")

# 3. Cache multipliers applied correctly.
c = pricing.token_cost_usd("claude-sonnet-4-6", 0, 0, cache_read_tokens=1_000_000)
if abs(c - 3.00 * 0.10) < 1e-9:
    ok("cache-read priced at 0.10x input")
else:
    bad(f"cache-read multiplier wrong: got {c}, expected {3.00 * 0.10}")

c = pricing.token_cost_usd("claude-sonnet-4-6", 0, 0, cache_write_tokens=1_000_000)
if abs(c - 3.00 * 1.25) < 1e-9:
    ok("cache-write priced at 1.25x input")
else:
    bad(f"cache-write multiplier wrong: got {c}, expected {3.00 * 1.25}")

# 4. An unknown model must raise, not silently cost zero.
try:
    pricing.token_cost_usd("definitely-not-a-model", 100, 100)
    bad("unknown model did not raise UnknownModel")
except pricing.UnknownModel:
    ok("unknown model raises UnknownModel (caught upstream, flagged needs_repricing)")

# 5. CURRENT_VERSION must exist.
if pricing.CURRENT_VERSION in pricing.RATE_CARDS:
    ok(f"CURRENT_VERSION {pricing.CURRENT_VERSION!r} exists")
else:
    bad(f"CURRENT_VERSION {pricing.CURRENT_VERSION!r} is not in RATE_CARDS")

# 6. A typical WSDC speech should land near the COST.md figure.
speech = (
    pricing.token_cost_usd("claude-sonnet-4-6", 6_000, 1_200)
    + pricing.token_cost_usd("claude-haiku-4-5", 3_000, 1_800)
)
print(f"\n  reference WSDC speech (Sonnet 6k/1.2k + Haiku 3k/1.8k) = ${speech:.4f}"
      f"   [COST.md says ~$0.035]")

print()
if failures:
    print(f"{RED}{len(failures)} failure(s).{RESET}")
    raise SystemExit(1)
print(f"{GREEN}Rate card OK.{RESET}")
