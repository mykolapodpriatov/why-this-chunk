"""Batch diagnosis over a queries JSON-Lines file.

Reads ``{"query": ..., "expect": ...}`` records — one per line — and runs the
same per-query path the ``diagnose`` command uses (the failure taxonomy plus the
counterfactual minimal-fix search) for each row. A row may carry ``expect_text``
or ``expect_meta`` instead of ``expect``; the CLI resolves those locators to a
chunk id before diagnosis. The rows are then aggregated into a
:class:`BatchResult`: how many queries fell into each
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
    "has_failures",
    "has_unfixable",
    "load_queries",
    "run_batch",
]

#: Label used in the aggregate for queries with no failure class (the expected
#: chunk already ranks within ``top_k`` — i.e. not currently failing).
NO_FAILURE_LABEL = "none"


@dataclass(frozen=True, slots=True)
class BatchQuery:
    """One query plus an expected-chunk locator read from the queries file.

    Exactly one of ``expect``, ``expect_text``, or ``expect_meta`` is set on a
    row as loaded from JSONL. After CLI resolution, ``expect`` is the chunk id.
    """

    query: str
    expect: str | None = None
    expect_text: str | None = None
    expect_meta: str | None = None


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
    """Load query rows from a JSON-Lines file.

    Each non-blank line must be a JSON object carrying ``query`` and exactly one
    of ``expect``, ``expect_text``, or ``expect_meta``. Blank lines are skipped.

    Args:
        path: Path to the ``.jsonl`` queries file.

    Returns:
        The parsed queries in file order (empty for an empty file).

    Raises:
        ValueError: If a line is not valid JSON or lacks a valid locator.
            The message is prefixed with ``path:line_number`` so the failing line
            is unambiguous.
    """
    file_path = Path(path)
    queries: list[BatchQuery] = []
    locators = ("expect", "expect_text", "expect_meta")
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_number}: invalid JSON ({exc.msg})") from exc
            present = [key for key in locators if isinstance(record, dict) and key in record]
            if not isinstance(record, dict) or "query" not in record or len(present) != 1:
                raise ValueError(
                    f"{file_path}:{line_number}: each line needs 'query' and exactly "
                    "one of 'expect', 'expect_text', 'expect_meta'"
                )
            queries.append(
                BatchQuery(
                    query=str(record["query"]),
                    expect=str(record["expect"]) if "expect" in record else None,
                    expect_text=(str(record["expect_text"]) if "expect_text" in record else None),
                    expect_meta=(str(record["expect_meta"]) if "expect_meta" in record else None),
                )
            )
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
        expected_id = query.expect
        if expected_id is None:
            raise ValueError("run_batch requires each query to have expect resolved to a chunk id")
        diagnosis = run_diagnose(retriever, query.query, expected_id, cfg)
        fixes = search_fixes(retriever, query.query, expected_id, cfg)
        rows.append(
            BatchRow(
                query=query.query,
                expect=expected_id,
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


def has_failures(result: BatchResult) -> bool:
    """Whether any row was classified into a failure class.

    A row has a failure when its expected chunk did not surface within ``top_k``
    (any non-``None`` :class:`~why_this_chunk.types.FailureClass`) — the signal a
    CI gate uses to fail on a RAG regression. Rows already retrieving the
    expected chunk (``failure_class is None``) never count.

    Args:
        result: The aggregate batch outcome to inspect.

    Returns:
        ``True`` if at least one row has a failure class, else ``False``.
    """
    return any(row.failure_class is not None for row in result.rows)


def has_unfixable(result: BatchResult) -> bool:
    """Whether any failing row has no bounded single-axis fix.

    A row is *unfixable* when it both has a failure class **and** the bounded
    counterfactual search found no config change (``fix is None``) that surfaces
    the expected chunk — the strictest CI signal, catching regressions no single
    documented knob can recover.

    Args:
        result: The aggregate batch outcome to inspect.

    Returns:
        ``True`` if at least one failing row lacks a fix, else ``False``.
    """
    return any(row.failure_class is not None and row.fix is None for row in result.rows)


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
