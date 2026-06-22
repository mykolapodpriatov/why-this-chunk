"""Tests for BM25/dense/hybrid retrievers and the documented normalization."""

from __future__ import annotations

import numpy as np
import pytest

from why_this_chunk import (
    BM25Retriever,
    Chunk,
    Corpus,
    DenseRetriever,
    FakeEmbedder,
    HybridRetriever,
    RetrievalConfig,
)
from why_this_chunk._scoring import DEGENERATE_NORM_VALUE, min_max_normalize


def test_bm25_ranks_lexical_overlap_first(bm25: BM25Retriever) -> None:
    results = bm25.search("Paris France capital", 3)
    assert results, "expected non-empty results"
    assert results[0].chunk.id == "paris"
    assert all(results[i].rank == i for i in range(len(results)))


def test_dense_ranks_semantically_related_first(dense: DenseRetriever) -> None:
    results = dense.search("river Seine France", 3)
    ids = [r.chunk.id for r in results]
    assert "seine" in ids


def test_search_is_deterministic(hybrid: HybridRetriever) -> None:
    first = [r.chunk.id for r in hybrid.search("Paris landmark", 5)]
    second = [r.chunk.id for r in hybrid.search("Paris landmark", 5)]
    assert first == second


def test_tie_break_is_stable_by_id() -> None:
    # Two identical matching chunks (tie) plus several non-matching ones so the
    # query terms stay rare (positive IDF). The tie must resolve by lower id.
    chunks = [
        Chunk(id="z_dup", text="alpha beta gamma"),
        Chunk(id="a_dup", text="alpha beta gamma"),
    ]
    for i in range(5):
        chunks.append(Chunk(id=f"filler{i}", text=f"delta epsilon zeta number {i}"))
    corpus = Corpus.from_chunks(chunks)
    retriever = BM25Retriever(corpus)
    results = retriever.search("alpha beta gamma", 2)
    assert [r.chunk.id for r in results] == ["a_dup", "z_dup"]


def test_min_max_normalize_basic() -> None:
    out = min_max_normalize(np.array([0.0, 5.0, 10.0]))
    assert out.tolist() == [0.0, 0.5, 1.0]


def test_min_max_normalize_degenerate_all_equal() -> None:
    out = min_max_normalize(np.array([3.0, 3.0, 3.0]))
    assert out.tolist() == [DEGENERATE_NORM_VALUE] * 3


def test_min_max_normalize_single_element() -> None:
    out = min_max_normalize(np.array([42.0]))
    assert out.tolist() == [DEGENERATE_NORM_VALUE]


def test_min_max_normalize_empty() -> None:
    out = min_max_normalize(np.array([], dtype=np.float64))
    assert out.size == 0


def test_hybrid_components_populated(hybrid: HybridRetriever) -> None:
    result = hybrid.search("Paris France capital", 1)[0]
    components = result.components
    assert components is not None
    assert components.dense is not None and 0.0 <= components.dense <= 1.0
    assert components.lexical is not None and 0.0 <= components.lexical <= 1.0
    assert components.alpha == 0.5
    assert components.dense_raw is not None
    assert components.lexical_raw is not None


def test_hybrid_alpha_extremes_match_modalities(dense: DenseRetriever, bm25: BM25Retriever) -> None:
    query = "Paris landmark iron"
    dense_first = [r.chunk.id for r in dense.search(query, 5)]
    lexical_first = [r.chunk.id for r in bm25.search(query, 5)]

    alpha_one = HybridRetriever(dense, bm25, alpha=1.0)
    alpha_zero = HybridRetriever(dense, bm25, alpha=0.0)
    assert [r.chunk.id for r in alpha_one.search(query, 5)] == dense_first
    assert [r.chunk.id for r in alpha_zero.search(query, 5)] == lexical_first


def test_hybrid_rejects_out_of_range_alpha(dense: DenseRetriever, bm25: BM25Retriever) -> None:
    with pytest.raises(ValueError, match="alpha"):
        HybridRetriever(dense, bm25, alpha=1.5)


def test_hybrid_rejects_mismatched_corpora(embedder: FakeEmbedder) -> None:
    corpus_a = Corpus.from_chunks([Chunk(id="a", text="one two")])
    corpus_b = Corpus.from_chunks([Chunk(id="b", text="three four")])
    dense = DenseRetriever(corpus_a, embedder)
    lexical = BM25Retriever(corpus_b)
    with pytest.raises(ValueError, match="same chunks"):
        HybridRetriever(dense, lexical)


def test_empty_corpus_returns_no_results(embedder: FakeEmbedder) -> None:
    corpus = Corpus.from_chunks([])
    assert BM25Retriever(corpus).search("anything", 5) == []
    assert DenseRetriever(corpus, embedder).search("anything", 5) == []


def test_empty_query_yields_zero_lexical_scores(bm25: BM25Retriever) -> None:
    scores = bm25.raw_scores("")
    assert scores.shape[0] == bm25.corpus_size
    assert np.allclose(scores, 0.0)


def test_reindex_chunk_size_requires_provenance(bm25: BM25Retriever) -> None:
    with pytest.raises(NotImplementedError, match="provenance"):
        bm25.reindex(RetrievalConfig(chunk_size=128))


def test_capability_flags(
    bm25: BM25Retriever, dense: DenseRetriever, hybrid: HybridRetriever
) -> None:
    for retriever in (bm25, dense, hybrid):
        assert retriever.supports_components is True
        assert retriever.supports_reindex is True


def test_k_caps_at_corpus_size(hybrid: HybridRetriever) -> None:
    results = hybrid.search("Paris", 999)
    assert len(results) == hybrid.corpus_size


def test_hybrid_empty_corpus_raw_scores(embedder: FakeEmbedder) -> None:
    corpus = Corpus.from_chunks([])
    dense = DenseRetriever(corpus, embedder)
    lexical = BM25Retriever(corpus)
    hybrid = HybridRetriever(dense, lexical, alpha=0.5)
    assert hybrid.raw_scores("anything").size == 0
    assert hybrid.search("anything", 3) == []


def test_hybrid_reindex_alpha_only(hybrid: HybridRetriever) -> None:
    # No chunk_size change, just alpha: must succeed without provenance.
    reindexed = hybrid.reindex(RetrievalConfig(alpha=0.8))
    assert isinstance(reindexed, HybridRetriever)
    assert reindexed.alpha == 0.8


def test_hybrid_reindex_preserves_alpha_when_unset(hybrid: HybridRetriever) -> None:
    reindexed = hybrid.reindex(RetrievalConfig(alpha=None))
    assert reindexed.alpha == hybrid.alpha


def test_hybrid_config_exposes_alpha(hybrid: HybridRetriever) -> None:
    assert hybrid.config.alpha == 0.5


def test_dense_score_text(dense: DenseRetriever) -> None:
    # Identical text to query => cosine 1.0 (unit-norm bag of words).
    assert dense.score_text("river France", "river France") == pytest.approx(1.0)


def test_bm25_score_text_empty_query(bm25: BM25Retriever) -> None:
    assert bm25.score_text("", "some text") == 0.0


def test_bm25_score_text_empty_index(embedder: FakeEmbedder) -> None:
    empty = BM25Retriever(Corpus.from_chunks([]))
    assert empty.score_text("q", "t") == 0.0


def test_bm25_all_empty_text_corpus_does_not_crash() -> None:
    """Regression: a corpus whose every chunk tokenizes to nothing (whitespace/
    punctuation only) must not raise ZeroDivisionError from BM25's IDF, and must
    return well-defined zero scores aligned to corpus order.
    """
    chunks = [
        Chunk(id="a", text="   "),
        Chunk(id="b", text="!!!"),
        Chunk(id="c", text=""),
    ]
    corpus = Corpus.from_chunks(chunks)
    retriever = BM25Retriever(corpus)  # must not raise

    scores = retriever.raw_scores("anything")
    assert scores.shape[0] == retriever.corpus_size == 3
    assert np.allclose(scores, 0.0)

    results = retriever.search("anything", 3)
    assert len(results) == 3
    assert all(r.score == 0.0 for r in results)
    # score_text against the (degenerate) index is well-defined too.
    assert retriever.score_text("anything", "some text") == 0.0


def test_dense_numpy_results_carry_components(tiny_corpus: Corpus, embedder: FakeEmbedder) -> None:
    """Regression: the numpy (non-FAISS) dense path must honor its advertised
    ``supports_components`` capability and populate ``ScoreComponents`` with the
    raw dense value, mirroring the FAISS path (which previously returned None).
    """
    retriever = DenseRetriever(tiny_corpus, embedder)
    assert retriever.backend == "numpy"
    assert retriever.supports_components is True
    results = retriever.search("river Seine France", 3)
    assert results
    for result in results:
        assert result.components is not None
        assert result.components.dense_raw is not None
        assert result.components.dense_raw == result.score
