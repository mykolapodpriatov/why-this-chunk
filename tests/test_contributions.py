"""Tests for the lexical-vs-dense contribution split."""

from __future__ import annotations

import pytest

from why_this_chunk import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    compute_split,
    split_for_result,
)
from why_this_chunk.types import ScoreComponents


def test_split_sums_and_dominance() -> None:
    components = ScoreComponents(dense=0.8, lexical=0.2, alpha=0.5)
    split = compute_split(components)
    assert split.dense_contribution == pytest.approx(0.4)
    assert split.lexical_contribution == pytest.approx(0.1)
    assert split.dominant == "dense"


def test_lexical_dominates() -> None:
    components = ScoreComponents(dense=0.1, lexical=0.9, alpha=0.5)
    split = compute_split(components)
    assert split.dominant == "lexical"


def test_exact_tie_resolves_to_dense() -> None:
    components = ScoreComponents(dense=0.5, lexical=0.5, alpha=0.5)
    split = compute_split(components)
    assert split.dense_contribution == split.lexical_contribution
    assert split.dominant == "dense"


def test_alpha_weights_contributions() -> None:
    components = ScoreComponents(dense=1.0, lexical=1.0, alpha=0.75)
    split = compute_split(components)
    assert split.dense_contribution == pytest.approx(0.75)
    assert split.lexical_contribution == pytest.approx(0.25)
    assert split.dominant == "dense"


def test_compute_split_rejects_non_hybrid_components() -> None:
    with pytest.raises(ValueError, match="hybrid components"):
        compute_split(ScoreComponents(dense=0.5, lexical=None, alpha=0.5))


def test_split_for_hybrid_result(dense: DenseRetriever, bm25: BM25Retriever) -> None:
    hybrid = HybridRetriever(dense, bm25, alpha=0.5)
    result = hybrid.search("Paris France", 1)[0]
    split = split_for_result(result)
    assert split is not None
    assert split.dominant in {"dense", "lexical"}


def test_split_for_non_hybrid_is_none(dense: DenseRetriever) -> None:
    result = dense.search("Paris France", 1)[0]
    assert split_for_result(result) is None


def test_split_for_bm25_is_none(bm25: BM25Retriever) -> None:
    result = bm25.search("Paris France", 1)[0]
    assert split_for_result(result) is None
