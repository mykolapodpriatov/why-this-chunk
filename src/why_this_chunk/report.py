"""Rendering: rich terminal tables, Markdown export, and JSON serialization.

Three output shapes are supported throughout the CLI:

* ``rich`` — coloured terminal tables (default, human-facing);
* ``md`` — Markdown for pasting into issues/PRs;
* ``json`` — stable machine-readable dicts.

The renderers never crash on degenerate inputs (empty attributions, missing
splits, ``None`` failure classes); they degrade to clear text instead.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from rich.console import Console
from rich.table import Table

from why_this_chunk.types import (
    ContributionSplit,
    DiagnosisResult,
    Explanation,
    FixSuggestion,
)

__all__ = [
    "diagnosis_to_dict",
    "diagnosis_to_markdown",
    "explanation_to_dict",
    "explanation_to_markdown",
    "render_diagnosis",
    "render_explanation",
]


def _split_summary(split: ContributionSplit | None) -> str:
    if split is None:
        return "n/a (non-hybrid retriever)"
    return (
        f"dense={split.dense_contribution:.3f} "
        f"lexical={split.lexical_contribution:.3f} "
        f"(dominant: {split.dominant})"
    )


def render_explanation(explanation: Explanation, console: Console | None = None) -> None:
    """Render an :class:`Explanation` as a rich table to ``console``."""
    out = console or Console()
    out.rule(f"[bold]explain[/bold] — {explanation.query!r}")
    out.print(
        f"chunk [cyan]{explanation.result.chunk.id}[/cyan]  "
        f"score={explanation.result.score:.4f}  rank={explanation.result.rank}"
    )
    out.print(f"split: {_split_summary(explanation.split)}")
    if explanation.degenerate:
        out.print(
            "[yellow]degenerate attribution[/yellow]: no single "
            f"{explanation.granularity} dominates (uniform shares)"
        )

    table = Table(title=f"{explanation.granularity} attribution", show_lines=False)
    table.add_column("share", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("span", justify="right")
    table.add_column(explanation.granularity)
    for attribution in explanation.sentences:
        table.add_row(
            f"{attribution.share:.3f}",
            f"{attribution.delta:+.4f}",
            f"{attribution.span[0]}-{attribution.span[1]}",
            _truncate(attribution.sentence),
        )
    out.print(table)


def render_diagnosis(diagnosis: DiagnosisResult, console: Console | None = None) -> None:
    """Render a :class:`DiagnosisResult` as rich output to ``console``."""
    out = console or Console()
    label = diagnosis.failure_class.value if diagnosis.failure_class else "indeterminate"
    out.rule(f"[bold]diagnose[/bold] — cause: [magenta]{label}[/magenta]")

    if diagnosis.unevaluable:
        names = ", ".join(c.value for c in diagnosis.unevaluable)
        out.print(f"[dim]unevaluable branches:[/dim] {names}")

    evidence_table = Table(title="evidence")
    evidence_table.add_column("key")
    evidence_table.add_column("value")
    for key, value in diagnosis.evidence.items():
        evidence_table.add_row(str(key), str(value))
    out.print(evidence_table)

    out.print(_fix_line(diagnosis.fix))


def _fix_line(fix: FixSuggestion | None) -> str:
    if fix is None:
        return "[yellow]fix:[/yellow] no single bounded config change surfaced the chunk"
    return (
        f"[green]fix:[/green] {fix.param} {fix.from_value!r} -> {fix.to_value!r} "
        f"(cost={fix.cost}, new_rank={fix.new_rank}) — {fix.explanation}"
    )


def explanation_to_markdown(explanation: Explanation) -> str:
    """Return a Markdown rendering of an :class:`Explanation`."""
    lines: list[str] = []
    lines.append(f"## explain — `{explanation.query}`")
    lines.append("")
    lines.append(
        f"- chunk: `{explanation.result.chunk.id}`  "
        f"score: `{explanation.result.score:.4f}`  rank: `{explanation.result.rank}`"
    )
    lines.append(f"- split: {_split_summary(explanation.split)}")
    if explanation.degenerate:
        lines.append(
            f"- degenerate: no single {explanation.granularity} dominates (uniform shares)"
        )
    lines.append("")
    lines.append(f"| share | delta | span | {explanation.granularity} |")
    lines.append("| ---: | ---: | ---: | --- |")
    for attribution in explanation.sentences:
        text = _truncate(attribution.sentence).replace("|", "\\|")
        lines.append(
            f"| {attribution.share:.3f} | {attribution.delta:+.4f} | "
            f"{attribution.span[0]}-{attribution.span[1]} | {text} |"
        )
    return "\n".join(lines) + "\n"


def diagnosis_to_markdown(diagnosis: DiagnosisResult) -> str:
    """Return a Markdown rendering of a :class:`DiagnosisResult`."""
    label = diagnosis.failure_class.value if diagnosis.failure_class else "indeterminate"
    lines: list[str] = []
    lines.append("## diagnose")
    lines.append("")
    lines.append(f"- cause: **{label}**")
    if diagnosis.unevaluable:
        names = ", ".join(c.value for c in diagnosis.unevaluable)
        lines.append(f"- unevaluable: {names}")
    lines.append("")
    lines.append("| evidence | value |")
    lines.append("| --- | --- |")
    for key, value in diagnosis.evidence.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    if diagnosis.fix is None:
        lines.append("- fix: _no single bounded config change surfaced the chunk_")
    else:
        fix = diagnosis.fix
        lines.append(
            f"- fix: `{fix.param}` `{fix.from_value}` -> `{fix.to_value}` "
            f"(cost {fix.cost}, new rank {fix.new_rank}) — {fix.explanation}"
        )
    return "\n".join(lines) + "\n"


def _fix_to_dict(fix: FixSuggestion | None) -> dict[str, Any] | None:
    return asdict(fix) if fix is not None else None


def explanation_to_dict(explanation: Explanation) -> dict[str, Any]:
    """Return a JSON-serializable dict for an :class:`Explanation`."""
    return {
        "query": explanation.query,
        "granularity": explanation.granularity,
        "degenerate": explanation.degenerate,
        "result": {
            "chunk_id": explanation.result.chunk.id,
            "score": explanation.result.score,
            "rank": explanation.result.rank,
        },
        "split": asdict(explanation.split) if explanation.split is not None else None,
        "sentences": [asdict(attribution) for attribution in explanation.sentences],
    }


def diagnosis_to_dict(diagnosis: DiagnosisResult) -> dict[str, Any]:
    """Return a JSON-serializable dict for a :class:`DiagnosisResult`."""
    return {
        "failure_class": (diagnosis.failure_class.value if diagnosis.failure_class else None),
        "unevaluable": [c.value for c in diagnosis.unevaluable],
        "evidence": diagnosis.evidence,
        "fix": _fix_to_dict(diagnosis.fix),
    }


def _truncate(text: str, limit: int = 80) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
