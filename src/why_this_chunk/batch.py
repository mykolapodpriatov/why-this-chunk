"""Batch diagnosis over a queries JSON-Lines file.

Reads ``{"query": ..., "expect": ...}`` records — one per line — and runs the
same per-query path the ``diagnose`` command uses (the failure taxonomy plus the
counterfactual minimal-fix search) for each row. The rows are then aggregated
into a :class:`BatchResult`: how many queries fell into each
:class:`~why_this_chunk.types.FailureClass`, how often each fix axis was the
cheapest suggestion, and the single most common suggested fix axis.

Loading is strict and line-numbered: a malformed line raises :class:`ValueError`
carrying the offending 1-based line number, exactly like
:meth:`why_this_chunk.corpus.Corpus.from_jsonl`, so callers can surface a precise
diagnostic. Blank lines are skipped; an empty file yields an empty batch.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from why_this_chunk.config import RetrievalConfig
from why_this_chunk.counterfactual import AXIS_PRIORITY, search_fixes
from why_this_chunk.retrievers import Retriever
from why_this_chunk.taxonomy import diagnose as run_diagnose
from why_this_chunk.types import FailureClass, FixSuggestion

__all__ = [
    "NO_FAILURE_LABEL",
    "BatchQuery",
    "BatchResult",
    "BatchRow",
    "load_queries",
    "run_batch",
]

#: Label used in the aggregate for queries with no failure class (the expected
#: chunk already ranks within ``top_k`` — i.e. not currently failing).
NO_FAILURE_LABEL = "none"


@dataclass(frozen=True, slots=True)
class BatchQuery:
    """One ``(query, expected_chunk_id)`` pair read from the queries file."""

    query: str
    expect: str


@dataclass(frozen=True, slots=True)
class BatchRow:
    """The diagnosis outcome for a single batched query."""

    query: str
    expect: str
    failure_class: FailureClass | None
    fix: FixSuggestion | None


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Aggregate outcome of a batch run.

    Attributes:
        rows: One :class:`BatchRow` per input query, in input order.
        failure_counts: Count of queries per failure-class label, in the
            canonical :class:`FailureClass` order with the
            :data:`NO_FAILURE_LABEL` bucket last. Only observed labels appear.
        fix_axis_counts: Count of queries whose cheapest fix used each axis, in
            the canonical axis-priority order. Only observed axes appear.
        top_fix_axis: The most common suggested fix axis, or ``None`` when no
            query had a bounded fix. Ties break by the fixed axis priority
            (``top_k`` < ``alpha`` < ``chunk_size`` < ``rerank``).
    """

    rows: list[BatchRow]
    failure_counts: dict[str, int]
    fix_axis_counts: dict[str, int]
    top_fix_axis: str | None


def load_queries(path: str | Path) -> list[BatchQuery]:
    """Load ``(query, expect)`` pairs from a JSON-Lines file.

    Each non-blank line must be a JSON object carrying at least ``query`` and
    ``expect``. Blank lines are skipped.

    Args:
        path: Path to the ``.jsonl`` queries file.

    Returns:
        The parsed queries in file order (empty for an empty file).

    Raises:
        ValueError: If a line is not valid JSON or lacks ``query``/``expect``.
            The message is prefixed with ``path:line_number`` so the failing line
            is unambiguous.
    """
    file_path = Path(path)
    queries: list[BatchQuery] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_number}: invalid JSON ({exc.msg})") from exc
            if not isinstance(record, dict) or "query" not in record or "expect" not in record:
                raise ValueError(
                    f"{file_path}:{line_number}: each line needs 'query' and 'expect' keys"
                )
            queries.append(BatchQuery(query=str(record["query"]), expect=str(record["expect"])))
    return queries


def run_batch(
    retriever: Retriever,
    queries: list[BatchQuery],
    config: RetrievalConfig | None = None,
) -> BatchResult:
    """Diagnose every query and aggregate the failure and fix distributions.

    Args:
        retriever: The retriever under test.
        queries: The parsed ``(query, expect)`` pairs.
        config: The active configuration; defaults to :class:`RetrievalConfig`.

    Returns:
        The aggregate :class:`BatchResult`.
    """
    cfg = config or RetrievalConfig()
    rows: list[BatchRow] = []
    failure_counter: Counter[str] = Counter()
    fix_axis_counter: Counter[str] = Counter()

    for query in queries:
        diagnosis = run_diagnose(retriever, query.query, query.expect, cfg)
        fixes = search_fixes(retriever, query.query, query.expect, cfg)
        rows.append(
            BatchRow(
                query=query.query,
                expect=query.expect,
                failure_class=diagnosis.failure_class,
                fix=fixes.best,
            )
        )
        label = diagnosis.failure_class.value if diagnosis.failure_class else NO_FAILURE_LABEL
        failure_counter[label] += 1
        if fixes.best is not None:
            fix_axis_counter[fixes.best.param] += 1

    return BatchResult(
        rows=rows,
        failure_counts=_ordered_failure_counts(failure_counter),
        fix_axis_counts=_ordered_axis_counts(fix_axis_counter),
        top_fix_axis=_top_axis(fix_axis_counter),
    )


def _ordered_failure_counts(counter: Counter[str]) -> dict[str, int]:
    """Return observed failure-class counts in canonical order, ``none`` last."""
    ordered: dict[str, int] = {
        klass.value: counter[klass.value] for klass in FailureClass if counter.get(klass.value)
    }
    if counter.get(NO_FAILURE_LABEL):
        ordered[NO_FAILURE_LABEL] = counter[NO_FAILURE_LABEL]
    return ordered


def _ordered_axis_counts(counter: Counter[str]) -> dict[str, int]:
    """Return observed fix-axis counts ordered by the fixed axis priority."""
    return {
        axis: counter[axis]
        for axis in sorted(counter, key=lambda name: AXIS_PRIORITY.get(name, len(AXIS_PRIORITY)))
    }


def _top_axis(counter: Counter[str]) -> str | None:
    """Most common fix axis; ties broken by the fixed axis priority."""
    if not counter:
        return None
    return min(
        counter,
        key=lambda axis: (-counter[axis], AXIS_PRIORITY.get(axis, len(AXIS_PRIORITY))),
    )
