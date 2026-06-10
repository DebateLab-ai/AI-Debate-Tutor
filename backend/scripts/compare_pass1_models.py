"""A/B compare Pass 1 (argument-structure) models in the WSDC two-pass pipeline.

Pass 1 is ~74% of per-turn cost (Sonnet 4.6 today). If Haiku 4.5 can carry the
argumentation at intermediate difficulty, per-debate cost drops ~45%. Cheaper is
only better if the arguments hold up — this script lets you eyeball that.

For each scenario (an opening speech and a rebuttal, both WSDC intermediate),
it runs the REAL production pipeline (app.wsdc.generate_wsdc_speech, including
RAG retrieval) once per candidate Pass 1 model. Pass 2 stays on Haiku exactly
as in production for intermediate. It captures per-call token usage and cost,
prints a summary table, and writes the full speeches + Pass 1 outlines to a
markdown report for side-by-side reading.

Run from backend/:
    source venv/bin/activate
    python scripts/compare_pass1_models.py

Costs roughly $0.10-$0.20 in real API spend per run.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ── Pricing (USD per 1M tokens) ────────────────────────────────────────────
# Source: Anthropic public pricing. Verify before quoting a partner.

PRICES = {
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "claude-haiku-4-5":  {"in": 1.00, "out":  5.00},
}

PASS1_CANDIDATES = ["claude-sonnet-4-6", "claude-haiku-4-5"]


def usd_cost(model: str, in_tokens: int, out_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float:
    p = PRICES[model]
    return (
        in_tokens * p["in"]
        + cache_read * p["in"] * 0.10
        + cache_write * p["in"] * 1.25
        + out_tokens * p["out"]
    ) / 1_000_000


# ── Scenarios ──────────────────────────────────────────────────────────────
# One opening (no opponent) and one rebuttal (the harder Pass 1 task: it has
# to target the opponent's actual claims). Rebuttal opponent speech reuses the
# realistic Vietnam-flavored student speech from measure_turn_cost.py.

OPPONENT_SPEECH = (
    "The proposition's case rests on the assumption that a global plastic ban is enforceable, "
    "but that misreads the realities of developing economies. In Vietnam, where I live, "
    "informal plastic supply chains employ millions; an overnight ban would devastate "
    "livelihoods before any environmental benefit materialised. Worse, the alternatives — "
    "bioplastics, glass, paper — have hidden carbon costs that often exceed the harm of "
    "well-managed plastic recycling. The solution is investment in waste infrastructure, "
    "not prohibition."
)

SCENARIOS = [
    {
        "name": "Opening speech (Government, first speaker)",
        "motion": "THBT social media does more harm than good for teenagers",
        "side": "Government",
        "is_rebuttal": False,
        "opponent_speech": None,
    },
    {
        "name": "Rebuttal speech (Government answering opposition)",
        "motion": "THW ban single-use plastics globally",
        "side": "Government",
        "is_rebuttal": True,
        "opponent_speech": OPPONENT_SPEECH,
    },
]


# ── Usage capture + Pass 1 JSON capture ────────────────────────────────────

@dataclass
class CallRecord:
    label: str
    model: str
    in_tokens: int = 0
    out_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    latency_s: float = 0.0

    @property
    def cost(self) -> float:
        return usd_cost(self.model, self.in_tokens, self.out_tokens, self.cache_read, self.cache_write)


@dataclass
class RunResult:
    scenario: str
    pass1_model: str
    ok: bool
    error: Optional[str] = None
    speech: str = ""
    pass1_json: Optional[dict] = None
    calls: list[CallRecord] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(c.cost for c in self.calls)

    @property
    def total_latency(self) -> float:
        return sum(c.latency_s for c in self.calls)


CURRENT_CALLS: list[CallRecord] = []


def _patch_anthropic() -> None:
    """Wrap wsdc's Anthropic client to record token usage + latency per call."""
    from app import wsdc

    original = wsdc._anthropic.messages.create

    def wrapped(**kwargs: Any) -> Any:
        start = time.monotonic()
        resp = original(**kwargs)
        latency = time.monotonic() - start
        usage = resp.usage
        # Pass 1 uses max_tokens=2048, Pass 2 uses 4096 (see wsdc.py).
        label = "Pass 1 (structure)" if kwargs.get("max_tokens") == 2048 else "Pass 2 (render)"
        CURRENT_CALLS.append(CallRecord(
            label=label,
            model=kwargs.get("model", "unknown"),
            in_tokens=getattr(usage, "input_tokens", 0) or 0,
            out_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            latency_s=latency,
        ))
        return resp

    wsdc._anthropic.messages.create = wrapped


LAST_PASS1_JSON: list[Optional[dict]] = [None]


def _patch_pass1_capture() -> None:
    """Wrap _run_pass1 so we can inspect the argument outline each model produced."""
    from app import wsdc

    original = wsdc._run_pass1

    def wrapped(*args: Any, **kwargs: Any) -> dict:
        result = original(*args, **kwargs)
        LAST_PASS1_JSON[0] = result
        return result

    wsdc._run_pass1 = wrapped


# ── Runner ─────────────────────────────────────────────────────────────────

def run_one(scenario: dict, pass1_model: str, rag: Any) -> RunResult:
    from app import wsdc
    from app.difficulty import get_config

    cfg = get_config("intermediate")
    CURRENT_CALLS.clear()
    LAST_PASS1_JSON[0] = None
    wsdc._SONNET = pass1_model  # Pass 1 reads this global; Pass 2 is Haiku for intermediate regardless

    try:
        speech = wsdc.generate_wsdc_speech(
            rag=rag,
            motion=scenario["motion"],
            side=scenario["side"],
            is_rebuttal=scenario["is_rebuttal"],
            opponent_speech=scenario["opponent_speech"],
            difficulty="intermediate",
            format="wsdc",
            top_k=cfg.rag_top_k,
            min_score=cfg.rag_min_score,
        )
    except Exception as e:
        traceback.print_exc()
        return RunResult(scenario=scenario["name"], pass1_model=pass1_model, ok=False,
                         error=f"{type(e).__name__}: {e}", calls=list(CURRENT_CALLS))

    ok = not speech.startswith("[Error")
    return RunResult(
        scenario=scenario["name"],
        pass1_model=pass1_model,
        ok=ok,
        error=None if ok else speech,
        speech=speech,
        pass1_json=LAST_PASS1_JSON[0],
        calls=list(CURRENT_CALLS),
    )


# ── Reporting ──────────────────────────────────────────────────────────────

def outline_summary(p1: Optional[dict]) -> str:
    """One-line structural fingerprint of the Pass 1 outline."""
    if not p1:
        return "(no Pass 1 JSON captured)"
    parts = []
    if "arguments" in p1 and p1.get("arguments"):
        args = p1["arguments"] or []
        parts.append(f"{len(args)} argument(s): " + "; ".join((a.get("label") or "?") for a in args if isinstance(a, dict)))
    if p1.get("rebuttals"):
        parts.append(f"{len(p1['rebuttals'])} rebuttal(s): " + "; ".join((r.get("target") or "?") for r in p1["rebuttals"] if isinstance(r, dict)))
    if p1.get("clashes"):
        parts.append(f"{len(p1['clashes'])} clash(es)")
    if p1.get("new_argument"):
        na = p1["new_argument"]
        if isinstance(na, dict):
            parts.append(f"new argument: {na.get('label', '?')}")
    return " | ".join(parts) if parts else "(unrecognized outline shape)"


def write_markdown_report(results: list[RunResult], path: Path) -> None:
    lines = [
        "# Pass 1 model A/B — WSDC intermediate",
        "",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}. Pass 2 is Haiku 4.5 in all runs "
        "(production config for intermediate). Only Pass 1 differs.",
        "",
        "Read the speeches blind if you can: judge argument quality, whether rebuttals target "
        "what the opponent actually said, and whether logic chains have gaps.",
        "",
    ]
    for scenario in SCENARIOS:
        lines.append(f"\n## {scenario['name']}")
        lines.append(f"\n**Motion:** {scenario['motion']}")
        if scenario["opponent_speech"]:
            lines.append(f"\n**Opponent said:** {scenario['opponent_speech']}")
        for r in [x for x in results if x.scenario == scenario["name"]]:
            lines.append(f"\n### Pass 1 = `{r.pass1_model}`")
            if not r.ok:
                lines.append(f"\n**FAILED:** {r.error}")
                continue
            lines.append(f"\n*cost ${r.total_cost:.4f} · latency {r.total_latency:.1f}s · "
                         f"outline: {outline_summary(r.pass1_json)}*")
            lines.append("\n<details><summary>Pass 1 JSON outline</summary>\n")
            lines.append("```json\n" + json.dumps(r.pass1_json, indent=2, ensure_ascii=False) + "\n```")
            lines.append("\n</details>\n")
            lines.append(r.speech)
    path.write_text("\n".join(lines))


def print_summary(results: list[RunResult]) -> None:
    print(f"\n{'=' * 78}")
    print("SUMMARY  (Pass 2 = Haiku in all runs; only Pass 1 differs)")
    print(f"{'=' * 78}")
    print(f"{'Scenario':<26} {'Pass 1 model':<20} {'Status':<7} {'Cost':>9} {'Latency':>9} {'Words':>7}")
    print("-" * 78)
    for r in results:
        if not r.ok:
            print(f"{r.scenario[:25]:<26} {r.pass1_model:<20} {'ERROR':<7} {'-':>9} {'-':>9} {'-':>7}")
            continue
        words = len(r.speech.split())
        print(f"{r.scenario[:25]:<26} {r.pass1_model:<20} {'ok':<7} ${r.total_cost:>7.4f} "
              f"{r.total_latency:>7.1f}s {words:>7}")

    by_model: dict[str, float] = {}
    for r in results:
        if r.ok:
            by_model[r.pass1_model] = by_model.get(r.pass1_model, 0.0) + r.total_cost
    if len(by_model) == 2:
        sonnet = by_model.get("claude-sonnet-4-6", 0.0)
        haiku = by_model.get("claude-haiku-4-5", 0.0)
        if sonnet and haiku:
            print("-" * 78)
            print(f"Per-turn cost with Haiku Pass 1: {100 * haiku / sonnet:.0f}% of Sonnet "
                  f"(saves {100 * (1 - haiku / sonnet):.0f}%)")


def main() -> None:
    # Import app.main first: initializes RAG once (shared across all runs).
    from app.main import get_rag_for

    _patch_anthropic()
    _patch_pass1_capture()

    rag = get_rag_for("intermediate")

    results: list[RunResult] = []
    for scenario in SCENARIOS:
        for model in PASS1_CANDIDATES:
            print(f"\n[{scenario['name']} | Pass 1 = {model} ...]")
            r = run_one(scenario, model, rag)
            status = f"ok, ${r.total_cost:.4f}, {r.total_latency:.1f}s" if r.ok else f"FAILED: {r.error}"
            print(f"  -> {status}")
            results.append(r)

    print_summary(results)

    report_path = Path(__file__).resolve().parent / f"pass1_ab_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    write_markdown_report(results, report_path)
    print(f"\nFull speeches + outlines written to:\n  {report_path}")
    print("\nRead the report before deciding — the summary table can't tell you whether the")
    print("rebuttals actually engage the opponent's claims. That's the thing to judge.")


if __name__ == "__main__":
    main()
