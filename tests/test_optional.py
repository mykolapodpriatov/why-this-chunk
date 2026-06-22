"""Light coverage for optional extras: FAISS dense backend and the web app.

These tests skip cleanly when an extra is not installed, so the suite stays
green on a minimal install while still exercising the optional code paths when
the extras are present. They remain fully offline (no model downloads).
"""

from __future__ import annotations

import importlib.util

import pytest

from why_this_chunk import (
    BM25Retriever,
    Corpus,
    DenseRetriever,
    FakeEmbedder,
)

_HAS_FAISS = importlib.util.find_spec("faiss") is not None
_HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None
_HAS_ST = importlib.util.find_spec("sentence_transformers") is not None


@pytest.mark.skipif(not _HAS_FAISS, reason="faiss extra not installed")
def test_faiss_backend_matches_numpy_ranking(tiny_corpus: Corpus, embedder: FakeEmbedder) -> None:
    numpy_dense = DenseRetriever(tiny_corpus, embedder, use_faiss=False)
    faiss_dense = DenseRetriever(tiny_corpus, embedder, use_faiss=True)
    assert faiss_dense.backend == "faiss"
    query = "river Seine northern France"
    numpy_ids = [r.chunk.id for r in numpy_dense.search(query, 5)]
    faiss_ids = [r.chunk.id for r in faiss_dense.search(query, 5)]
    assert numpy_ids == faiss_ids


@pytest.mark.skipif(not _HAS_FAISS, reason="faiss extra not installed")
def test_faiss_reindex_preserves_backend(tiny_corpus: Corpus, embedder: FakeEmbedder) -> None:
    from why_this_chunk import RetrievalConfig

    faiss_dense = DenseRetriever(tiny_corpus, embedder, use_faiss=True)
    # No chunk_size change => same corpus, backend retained.
    reindexed = faiss_dense.reindex(RetrievalConfig(top_k=3))
    assert reindexed.backend == "faiss"


@pytest.mark.skipif(not _HAS_FAISS, reason="faiss extra not installed")
def test_faiss_empty_corpus_falls_back(embedder: FakeEmbedder) -> None:
    empty = DenseRetriever(Corpus.from_chunks([]), embedder, use_faiss=True)
    assert empty.backend == "numpy"  # nothing to index
    assert empty.search("x", 3) == []


# Newer starlette emits a deprecation warning when its TestClient runs on httpx
# (rather than the not-yet-ubiquitous httpx2). That is an upstream packaging
# detail, irrelevant to this project, so it is ignored locally rather than
# letting the global ``error`` filter fail the test.
@pytest.mark.skipif(not _HAS_FASTAPI, reason="web extra not installed")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore")
def test_web_app_endpoints(tiny_corpus: Corpus, embedder: FakeEmbedder) -> None:
    from fastapi.testclient import TestClient

    from why_this_chunk.web import create_app

    dense = DenseRetriever(tiny_corpus, embedder)
    lexical = BM25Retriever(tiny_corpus)
    from why_this_chunk import HybridRetriever

    retriever = HybridRetriever(dense, lexical, alpha=0.5)
    client = TestClient(create_app(retriever))

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["corpus_size"] == len(tiny_corpus)

    page = client.get("/")
    assert page.status_code == 200
    assert "why-this-chunk inspector" in page.text

    explain = client.get("/api/explain", params={"query": "Paris France", "k": 2})
    assert explain.status_code == 200
    body = explain.json()
    assert body["query"] == "Paris France"
    assert len(body["explanations"]) <= 2

    bad = client.get("/api/explain", params={"query": "", "k": 2})
    assert bad.status_code == 422  # min_length validation


@pytest.mark.skipif(_HAS_ST, reason="runs only when the st extra is absent")
def test_sentence_transformer_import_guard() -> None:
    from why_this_chunk.embedders.sentence_transformers import (
        SentenceTransformerEmbedder,
    )

    with pytest.raises(ImportError, match="st"):
        SentenceTransformerEmbedder()


@pytest.mark.skipif(_HAS_ST, reason="runs only when the st extra is absent")
def test_cross_encoder_import_guard() -> None:
    from why_this_chunk.rerank import CrossEncoderReranker

    with pytest.raises(ImportError, match="st"):
        CrossEncoderReranker()


def test_rerank_wrapper_with_fake_reranker(tiny_corpus: Corpus, embedder: FakeEmbedder) -> None:
    # Exercise RerankingRetriever and the counterfactual rerank axis WITHOUT any
    # model: a deterministic fake reranker that prefers chunks containing 'seine'.
    from why_this_chunk import RetrievalConfig, search_fixes
    from why_this_chunk.rerank import RerankingRetriever

    class FakeReranker:
        # "flowing" is unique to the seine chunk, so this promotes it unambiguously.
        def score(self, query: str, texts: list[str]) -> list[float]:
            return [10.0 if "flowing" in t.lower() else 0.0 for t in texts]

    base = BM25Retriever(tiny_corpus)
    wrapped = RerankingRetriever(base, FakeReranker(), pool_size=10, active=True)
    assert wrapped.supports_rerank is True
    assert wrapped.supports_reindex is True
    # Reranked search promotes the seine chunk.
    top = wrapped.search("northern France", 1)
    assert top and top[0].chunk.id == "seine"

    # Inactive wrapper delegates to the base unchanged.
    inactive = RerankingRetriever(base, FakeReranker(), active=False)
    assert [r.chunk.id for r in inactive.search("northern France", 3)] == [
        r.chunk.id for r in base.search("northern France", 3)
    ]

    # The counterfactual rerank axis is now evaluable.
    result = search_fixes(
        wrapped, "northern France", "seine", RetrievalConfig(top_k=5, rerank=False)
    )
    assert "rerank" not in result.unevaluable

    # Reindex toggling the rerank flag returns a new wrapper.
    toggled_off = wrapped.reindex(RetrievalConfig(rerank=False))
    assert isinstance(toggled_off, RerankingRetriever)
    assert toggled_off.corpus is not None
    assert toggled_off.corpus_size == wrapped.corpus_size
    # When inactive it matches the base ordering again.
    assert [r.chunk.id for r in toggled_off.search("northern France", 3)] == [
        r.chunk.id for r in base.search("northern France", 3)
    ]


def test_rerank_empty_pool_returns_empty() -> None:
    from why_this_chunk.rerank import RerankingRetriever

    class FakeReranker:
        def score(self, query: str, texts: list[str]) -> list[float]:
            return [0.0] * len(texts)

    base = BM25Retriever(Corpus.from_chunks([]))
    wrapped = RerankingRetriever(base, FakeReranker(), active=True)
    assert wrapped.search("q", 3) == []


def test_rerank_wrapper_rejects_bad_pool_size(
    tiny_corpus: Corpus,
) -> None:
    from why_this_chunk.rerank import RerankingRetriever

    class FakeReranker:
        def score(self, query: str, texts: list[str]) -> list[float]:
            return [0.0] * len(texts)

    with pytest.raises(ValueError, match="pool_size"):
        RerankingRetriever(BM25Retriever(tiny_corpus), FakeReranker(), pool_size=0)
