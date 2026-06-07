"""PDF report renderer for /api/v1/debates/{id}/report.pdf.

Two sections per partner spec:
  1. Feedback & actionable next steps — overall score, per-metric breakdown,
     judge's prose feedback, and a recommended next drill keyed off the
     score's weakness_type.
  2. Transcript — every round's speeches, labeled by speaker.

Heavy imports (weasyprint, jinja2) happen lazily inside render_pdf() so a
missing system dep (Pango/Cairo on Linux without nixpacks config) becomes a
503 at endpoint time rather than crashing the entire FastAPI process on boot.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# Maps weakness_type → (drill name, short next-step instruction).
# Surfaced as the "Recommended next drill" in the PDF's feedback section so
# the student leaves with one concrete thing to practice. The DrillSubmit
# endpoints in main.py already understand these tags — partners can deep-link
# straight into a drill on their UI if they want.
_DRILL_RECOMMENDATIONS: dict[str, tuple[str, str]] = {
    "rebuttal":  ("Rebuttal Drill",
                  "Take one of your opponent's claims and respond with quote-then-counter: quote them in your own words, identify the assumption, show why it fails step-by-step."),
    "structure": ("Structure Drill",
                  "Rebuild your last speech around explicit signposts: roadmap → premise → mechanism → weighing → conclusion. One sentence per slot, no more."),
    "weighing":  ("Weighing Drill",
                  "Pick one impact from your side and one from theirs. Compare them on probability × magnitude × timeframe in three sentences."),
    "evidence":  ("Evidence Drill",
                  "Replace one of your abstract claims with a single recognizable real-world example (Amazon, COVID-19, a named country's policy). Show the causal link."),
    "strategy":  ("Strategy Drill",
                  "Identify the one argument that wins you the round. Cut everything else to a sentence; spend the saved time strengthening the winner."),
}


def _format_iso_datetime(value: Any) -> str:
    """Render a Postgres ISO string or datetime as 'YYYY-MM-DD'."""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except ValueError:
            return value
    return ""


def _group_messages_by_round(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bucket the flat message list into [{round_no, turns: [{speaker, content, ...}]}]
    so the template can iterate cleanly without doing logic in Jinja."""
    rounds: dict[int, list[dict[str, Any]]] = {}
    for m in messages:
        rounds.setdefault(m["round_no"], []).append(m)
    return [
        {"round_no": rn, "turns": rounds[rn]}
        for rn in sorted(rounds)
    ]


def render_pdf(
    *,
    debate: dict[str, Any],
    messages: list[dict[str, Any]],
    score: dict[str, Any],
    metadata: Optional[dict[str, Any]] = None,
) -> bytes:
    """Render the report PDF and return raw bytes.

    debate:   row from `debates` (must include motion, mode, difficulty,
              starter, num_rounds, created_at).
    messages: rows from `messages` ordered by created_at ascending.
    score:    row from `scores` (must include overall + per-metric scores
              and per-metric feedback strings, plus weakness_type).
    metadata: optional override for the debate's metadata bag (defaults to
              debate['metadata']).

    Raises:
        RuntimeError if WeasyPrint or its system deps can't load. The API
        layer maps this to 503.
    """
    try:
        from weasyprint import HTML
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError as e:
        raise RuntimeError(f"PDF dependencies unavailable: {e}") from e

    templates_dir = Path(__file__).resolve().parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("debate_report.html")

    weakness = (score.get("weakness_type") or "").lower()
    drill_name, drill_instruction = _DRILL_RECOMMENDATIONS.get(
        weakness, ("Open Practice", "Pick any motion and rerun this exercise focusing on the metric you scored lowest on."),
    )

    rendered_html = template.render(
        motion=debate.get("motion") or "Untitled debate",
        mode=debate.get("mode", "casual"),
        difficulty=debate.get("difficulty", "intermediate"),
        starter=debate.get("starter", "user"),
        num_rounds=debate.get("num_rounds", 0),
        date_str=_format_iso_datetime(debate.get("created_at")),
        metadata=(metadata if metadata is not None else (debate.get("metadata") or {})),
        score=score,
        rounds=_group_messages_by_round(messages),
        drill_name=drill_name,
        drill_instruction=drill_instruction,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )

    pdf_bytes = HTML(string=rendered_html).write_pdf()
    return pdf_bytes
