"""Unit tests for the batch loader and aggregator — fully offline."""

from __future__ import annotations

from pathlib import Path

import pytest

from why_this_chunk import HybridRetriever, RetrievalConfig
from why_this_chunk.batch import (
    BatchQuery,
    BatchResult,
    BatchRow,
    has_failures,
    has_unfixable,
    load_queries,
    run_batch,
)
from why_this_chunk.types import FailureClass, FixSuggestion


def _result(rows: list[BatchRow]) -> BatchResult:
    """Wrap rows in a minimal ``BatchResult`` for gate-predicate tests."""
    return BatchResult(rows=rows, failure_counts={}, fix_axis_counts={}, top_fix_axis=None)


def _fix() -> FixSuggestion:
    return FixSuggestion(
        param="top_k",
        from_value=1,
        to_value=2,
        cost=1,
        new_rank=1,
        explanation="raise top_k",
    )


def test_load_queries_reads_records(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(
        '{"query": "a", "expect": "x"}\n{"query": "b", "expect": "y"}\n',
        encoding="utf-8",
    )
    queries = load_queries(path)
    assert queries == [BatchQuery("a", "x"), BatchQuery("b", "y")]


def test_load_queries_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text('\n{"query": "a", "expect": "x"}\n\n', encoding="utf-8")
    assert load_queries(path) == [BatchQuery("a", "x")]


def test_load_queries_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text("", encoding="utf-8")
    assert load_queries(path) == []


def test_load_queries_malformed_json_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text('{"query": "a", "expect": "x"}\nnope\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r":2: invalid JSON"):
        load_queries(path)


def test_load_queries_missing_keys_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text('{"query": "no expect"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r":1: each line needs"):
        load_queries(path)


def test_run_batch_aggregates(hybrid: HybridRetriever) -> None:
    queries = [
        BatchQuery(query="Paris France", expect="seine"),
        BatchQuery(query="Paris France", expect="nonexistent"),
    ]
    result = run_batch(hybrid, queries, RetrievalConfig(top_k=1, alpha=0.5))
    assert len(result.rows) == 2
    # A missing chunk is always classifiable as missing_from_index.
    assert result.failure_counts["missing_from_index"] == 1
    # The missing row yields no fix; the total fix count never exceeds the rows.
    assert sum(result.fix_axis_counts.values()) <= len(result.rows)


def test_run_batch_empty_is_inert(hybrid: HybridRetriever) -> None:
    result = run_batch(hybrid, [], RetrievalConfig(top_k=1, alpha=0.5))
    assert result.rows == []
    assert result.failure_counts == {}
    assert result.fix_axis_counts == {}
    assert result.top_fix_axis is None


def test_has_failures_and_unfixable_clean_run() -> None:
    # Every row already retrieves its expected chunk: neither gate should trip.
    rows = [
        BatchRow(query="a", expect="x", failure_class=None, fix=None),
        BatchRow(query="b", expect="y", failure_class=None, fix=None),
    ]
    result = _result(rows)
    assert has_failures(result) is False
    assert has_unfixable(result) is False


def test_has_failures_true_but_fixable_is_not_unfixable() -> None:
    # A classified failure that still has a bounded fix: 'failure' trips,
    # 'unfixable' does not.
    rows = [
        BatchRow(query="a", expect="x", failure_class=None, fix=None),
        BatchRow(query="b", expect="y", failure_class=FailureClass.OUT_RANKED, fix=_fix()),
    ]
    result = _result(rows)
    assert has_failures(result) is True
    assert has_unfixable(result) is False


def test_has_unfixable_true_when_failing_row_has_no_fix() -> None:
    # A failing row with no bounded fix trips both thresholds.
    rows = [
        BatchRow(query="a", expect="x", failure_class=FailureClass.MISSING_FROM_INDEX, fix=None),
    ]
    result = _result(rows)
    assert has_failures(result) is True
    assert has_unfixable(result) is True
