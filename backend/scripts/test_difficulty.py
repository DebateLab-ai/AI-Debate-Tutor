#!/usr/bin/env python3
"""Side-by-side test of difficulty tiers against the local backend.

Creates one debate per tier with the same motion and the same opening user
argument, then triggers the AI's reply and prints them back-to-back so you
can eyeball whether beginner / intermediate / advanced actually feel different.

Run the backend first:
    cd backend
    source venv/bin/activate
    uvicorn app.main:app --reload --port 8000

Then in another terminal:
    python scripts/test_difficulty.py
    python scripts/test_difficulty.py --mode parliamentary
    python scripts/test_difficulty.py --motion "Schools should ban smartphones" --argument "..."
    python scripts/test_difficulty.py --tiers beginner,advanced   # subset only

Prerequisite: real OPENAI_API_KEY in backend/.env. Without it the safety layer
fail-closes every turn submission with a 503.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_MOTION = "Schools should ban smartphones during class hours"
DEFAULT_ARGUMENT = (
    "Smartphones are a major distraction in classrooms. Students who use them "
    "during lessons score lower on tests, retain less, and lose focus on what "
    "the teacher is saying. Schools that have implemented bans report better "
    "attention and stronger peer relationships among students."
)
ALL_TIERS = ("beginner", "intermediate", "advanced")
SEP = "=" * 76


def _post(url: str, data: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


def _summarize(text: str) -> str:
    words = len(text.split())
    # naive sentence count — good enough to spot beginner shrinkage / advanced bloat
    sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)
    return f"{words} words, ~{sentences} sentences"


def run_tier(base: str, tier: str, motion: str, argument: str, mode: str) -> None:
    print(f"\n{SEP}\n  TIER: {tier.upper():13s}  mode: {mode}\n{SEP}")
    debate = _post(
        f"{base}/v1/debates",
        {
            "title": motion,
            "num_rounds": 2,
            "starter": "user",
            "mode": mode,
            "difficulty": tier,
        },
    )
    did = debate["id"]
    print(f"  debate id: {did}")

    _post(
        f"{base}/v1/debates/{did}/turns",
        {"speaker": "user", "content": argument},
    )

    print(f"  …generating AI reply (this can take 5–15s)…\n")
    reply = _post(f"{base}/v1/debates/{did}/auto-turn", {})
    text = reply.get("content", "(empty)")
    print(f"  --- AI reply ({_summarize(text)}) ---\n")
    # indent each line so the reply is visually grouped
    for line in text.splitlines() or [text]:
        print(f"  {line}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="http://127.0.0.1:8000", help="Backend base URL (default: %(default)s)")
    p.add_argument("--motion", default=DEFAULT_MOTION, help="Debate motion")
    p.add_argument("--argument", default=DEFAULT_ARGUMENT, help="Opening user argument")
    p.add_argument("--mode", default="casual", choices=("casual", "parliamentary"))
    p.add_argument(
        "--tiers",
        default=",".join(ALL_TIERS),
        help="Comma-separated tiers to test (default: all three)",
    )
    args = p.parse_args()

    try:
        _get(f"{args.base}/v1/health")
    except (urllib.error.URLError, ConnectionRefusedError) as e:
        print(f"Backend not reachable at {args.base}: {e}", file=sys.stderr)
        print("Start it: cd backend && uvicorn app.main:app --reload --port 8000", file=sys.stderr)
        return 1

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    print(f"Motion:   {args.motion}\nArgument: {args.argument[:120]}{'…' if len(args.argument) > 120 else ''}")

    for tier in tiers:
        try:
            run_tier(args.base, tier, args.motion, args.argument, args.mode)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:400]
            print(f"  HTTP {e.code}: {body}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

    print(f"\n{SEP}\nEyeball check — do the replies feel meaningfully different?")
    print("  beginner     → longer (~7-10 sentences), peer-explaining voice, no jargon-drops")
    print("  intermediate → today's debate-coach voice (the canary; unchanged behavior)")
    print("  advanced     → named techniques: fallacies, comparative weighing, warrant-chain attacks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
