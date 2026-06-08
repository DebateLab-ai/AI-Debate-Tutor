"""Generate a sample debate report PDF for visual review.

No real API calls — synthetic data, runs in ~1 second. Use this when iterating
on the template; opening the file in Preview is faster than going through
TestClient + Supabase + an AI debate every time.

Run from backend/:
    source venv/bin/activate
    python scripts/sample_pdf.py
    open /tmp/sample_report.pdf
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pdf import render_pdf

OUT_PATH = Path("/tmp/sample_report.pdf")


def main() -> None:
    debate = {
        "motion": "THW ban single-use plastics globally",
        "mode": "wsdc",
        "difficulty": "intermediate",
        "starter": "assistant",
        "num_rounds": 2,
        "created_at": datetime.utcnow().isoformat(),
        "metadata": {
            "Debater": "Nguyen An",
            "Class": "Advanced Wednesdays",
            "Instructor": "Ms. Linh",
            "Session": "Week 4",
        },
    }

    messages = [
        {"round_no": 1, "speaker": "assistant", "content": (
            "In this speech I'll argue that a global ban on single-use plastics is both necessary "
            "and feasible. First, the harms of plastics are not contested — they choke marine "
            "life, microplastics enter our food chain, and waste piles up in countries that lack "
            "disposal infrastructure. Second, the only intervention proportionate to a global "
            "problem is a global ban. Voluntary measures have failed for thirty years; "
            "meanwhile, plastic production has doubled. Recycling rates remain below 10 percent "
            "worldwide. Proud to propose."
        )},
        {"round_no": 1, "speaker": "user", "content": (
            "The proposition's case rests on the assumption that a global plastic ban is "
            "enforceable, but that misreads the realities of developing economies. In Vietnam, "
            "where I live, informal plastic supply chains employ millions; an overnight ban "
            "would devastate livelihoods before any environmental benefit materialised. Worse, "
            "the alternatives — bioplastics, glass, paper — have hidden carbon costs that often "
            "exceed the harm of well-managed plastic recycling. The solution is investment in "
            "waste infrastructure, not prohibition."
        )},
        {"round_no": 2, "speaker": "assistant", "content": (
            "The opposition concedes the harm — they offer no challenge to our central claim "
            "that plastics are choking ecosystems. Their case rests on transition costs, but "
            "transition costs are not impact costs; they are the price of every meaningful "
            "reform. On informal labour: a ban with a phase-in period, paired with retraining "
            "subsidies, addresses this. On bioplastics: the opposition cherry-picks the worst "
            "alternatives; modern compostables match plastic's carbon footprint at a fraction "
            "of the marine harm. Proud to propose."
        )},
        {"round_no": 2, "speaker": "user", "content": (
            "Even granting the proposition's framing, they still owe us a mechanism. None of "
            "their arguments engage with the displacement problem I raised. They simply assert "
            "that the harm of plastics outweighs the harm of unemployment — without showing "
            "the timeframe or probability. On our side, the harm is immediate and concentrated; "
            "on theirs, it is diffuse and contingent. We win this debate on weighing alone."
        )},
    ]

    score = {
        "overall": 6.0,
        "content_structure": 7.0,
        "engagement": 5.0,
        "strategy": 6.0,
        "feedback": (
            "Opposition effectively identifies the enforcement gap (\"misreads the realities of "
            "developing economies\") and pivots to concrete harms, but fails to sustain momentum "
            "in Round 2. Strength: specific, localized impact analysis grounded in the Vietnam "
            "example. Weakness: the weighing claim in Round 2 (\"we win this debate on weighing "
            "alone\") lacks the comparative impact calculus needed to back it. Next drill: "
            "rebuild the Round 2 weighing using probability × magnitude × timeframe explicitly."
        ),
        "content_structure_feedback": (
            "Round 1 is well-signposted with three distinct arguments (enforcement, alternatives, "
            "infrastructure). However, Round 2 becomes abstract: \"the harm is immediate and "
            "concentrated; on theirs, it is diffuse and contingent\" asserts weighing without "
            "building the comparative framework that earns that conclusion."
        ),
        "engagement_feedback": (
            "Opposition directly challenges the \"assumption that a global plastic ban is "
            "enforceable\" and flags the \"displacement problem,\" but Round 2 engagement weakens. "
            "The rebuttal \"they still owe us a mechanism\" is valid but doesn't explain *why* "
            "the proposition's phase-in + subsidies fail — it just restates the demand."
        ),
        "strategy_feedback": (
            "Opposition collapses onto weighing in Round 2 — a smart prioritization move. "
            "However, the strategy is underdeveloped: no quantified comparison of transition harm "
            "vs. environmental harm, and no clear win condition articulated beyond assertion."
        ),
        "weakness_type": "weighing",
    }

    pdf_bytes = render_pdf(debate=debate, messages=messages, score=score)
    OUT_PATH.write_bytes(pdf_bytes)
    print(f"Sample PDF written to {OUT_PATH} ({len(pdf_bytes):,} bytes)")
    print(f"Open with: open {OUT_PATH}")


if __name__ == "__main__":
    main()
