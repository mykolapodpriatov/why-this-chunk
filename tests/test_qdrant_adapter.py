"""Tests for the Qdrant adapter against a real in-process Qdrant.

Gated behind the ``qdrant`` extra: skipped cleanly via ``pytest.importorskip``
when ``qdrant_client`` is absent. Everything runs against
``QdrantClient(":memory:")`` — local mode, no server, no network — with the
deterministic :class:`FakeEmbedder`, so the suite stays reproducible and offline.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient, models

from why_this_chunk import Chunk, FakeEmbedder, RetrievalConfig, explain_chunk
from why_this_chunk.retrievers import Retriever
from why_this_chunk.retrievers.adapters.qdrant import QdrantRetriever

QUERY = "river Seine northern France"


def _upsert(
    embedder: FakeEmbedder,
    chunks: list[Chunk],
    *,
    distance: models.Distance = models.Distance.COSINE,
    payload_id: bool = True,
) -> tuple[QdrantClient, str]:
    """Index ``chunks`` in a fresh in-memory collection."""
    client = QdrantClient(":memory:")
    name = f"c-{uuid.uuid4().hex}"
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=embedder.dim, distance=distance),
    )
    vectors = (
        embedder.encode([chunk.text for chunk in chunks])
        if chunks
        else np.zeros((0, embedder.dim), dtype=np.float32)
    )
    client.upsert(
        collection_name=name,
        points=[
            models.PointStruct(
                id=index + 1,
                vector=[float(value) for value in vectors[index]],
                payload=(
                    {"text": chunk.text, "chunk_id": chunk.id, "lang": "en"}
                    if payload_id
                    else {"text": chunk.text}
                ),
            )
            for index, chunk in enumerate(chunks)
        ],
    )
    return client, name


def _retriever(embedder: FakeEmbedder, chunks: list[Chunk], **kwargs: object) -> QdrantRetriever:
    client, name = _upsert(embedder, chunks, **kwargs)  # type: ignore[arg-type]
    return QdrantRetriever(client, name, embedder, id_field="chunk_id")


# --- the retriever contract -------------------------------------------------


def test_satisfies_the_retriever_protocol(embedder: FakeEmbedder, tiny_chunks: list[Chunk]) -> None:
    """The adapter is structurally a Retriever."""
    assert isinstance(_retriever(embedder, tiny_chunks), Retriever)


def test_ranking_matches_the_builtin_dense_retriever(
    embedder: FakeEmbedder, tiny_chunks: list[Chunk], dense: object
) -> None:
    """The same corpus and embedder produce the same order as DenseRetriever."""
    adapter = _retriever(embedder, tiny_chunks)

    adapter_ids = [result.chunk.id for result in adapter.search(QUERY, 5)]
    dense_ids = [result.chunk.id for result in dense.search(QUERY, 5)]  # type: ignore[attr-defined]

    assert adapter_ids == dense_ids


def test_scores_match_cosine_similarity(embedder: FakeEmbedder, tiny_chunks: list[Chunk]) -> None:
    """Scores are true cosine similarities, comparable with score_text."""
    adapter = _retriever(embedder, tiny_chunks)

    for result in adapter.search(QUERY, 5):
        assert result.score == pytest.approx(adapter.score_text(QUERY, result.chunk.text), abs=1e-5)


def test_ranks_are_contiguous_and_scores_descend(
    embedder: FakeEmbedder, tiny_chunks: list[Chunk]
) -> None:
    """Results follow the protocol's ordering contract."""
    results = _retriever(embedder, tiny_chunks).search(QUERY, 3)

    assert [result.rank for result in results] == [0, 1, 2]
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


def test_components_are_dense_only(embedder: FakeEmbedder, tiny_chunks: list[Chunk]) -> None:
    """Every result carries a dense-only ScoreComponents."""
    adapter = _retriever(embedder, tiny_chunks)
    assert adapter.supports_components

    for result in adapter.search(QUERY, 3):
        assert result.components is not None
        assert result.components.lexical_raw is None
        assert result.components.dense_raw == pytest.approx(result.score)


def test_corpus_size_counts_the_collection(
    embedder: FakeEmbedder, tiny_chunks: list[Chunk]
) -> None:
    """corpus_size reports what the store holds."""
    assert _retriever(embedder, tiny_chunks).corpus_size == len(tiny_chunks)


def test_metadata_carries_the_remaining_payload(
    embedder: FakeEmbedder, tiny_chunks: list[Chunk]
) -> None:
    """Payload keys other than text and id become chunk metadata."""
    results = _retriever(embedder, tiny_chunks).search(QUERY, 1)

    assert results[0].chunk.metadata == {"lang": "en"}


def test_point_id_used_when_no_id_field_configured(
    embedder: FakeEmbedder, tiny_chunks: list[Chunk]
) -> None:
    """Without id_field the Qdrant point id becomes the chunk id."""
    client, name = _upsert(embedder, tiny_chunks, payload_id=False)
    adapter = QdrantRetriever(client, name, embedder)

    ids = {result.chunk.id for result in adapter.search(QUERY, 5)}

    assert ids == {"1", "2", "3", "4", "5"}


# --- degrading honestly -----------------------------------------------------


def test_reindex_is_refused_and_advertised_as_such(
    embedder: FakeEmbedder, tiny_chunks: list[Chunk]
) -> None:
    """supports_reindex is False and reindex says why."""
    adapter = _retriever(embedder, tiny_chunks)
    assert adapter.supports_reindex is False

    with pytest.raises(NotImplementedError, match="lives in the store"):
        adapter.reindex(RetrievalConfig())


def test_manhattan_collection_is_rejected_at_construction(
    embedder: FakeEmbedder, tiny_chunks: list[Chunk]
) -> None:
    """A metric that cannot become a cosine similarity fails fast."""
    client, name = _upsert(embedder, tiny_chunks, distance=models.Distance.MANHATTAN)

    with pytest.raises(ValueError, match="does not convert to a cosine similarity"):
        QdrantRetriever(client, name, embedder)


def test_euclid_collection_converts_back_to_cosine(
    embedder: FakeEmbedder, tiny_chunks: list[Chunk]
) -> None:
    """A Euclid collection yields the same scores as a Cosine one."""
    cosine = _retriever(embedder, tiny_chunks)
    client, name = _upsert(embedder, tiny_chunks, distance=models.Distance.EUCLID)
    euclid = QdrantRetriever(client, name, embedder, id_field="chunk_id")
    assert euclid.metric == "euclid"

    cosine_scores = {r.chunk.id: r.score for r in cosine.search(QUERY, 5)}
    euclid_scores = {r.chunk.id: r.score for r in euclid.search(QUERY, 5)}

    assert euclid_scores.keys() == cosine_scores.keys()
    for chunk_id, score in cosine_scores.items():
        assert euclid_scores[chunk_id] == pytest.approx(score, abs=1e-5)
    # Order agrees wherever the scores are actually distinct; the tail of this
    # corpus is a three-way tie at 0.0 that each metric reaches through
    # different float noise.
    assert [r.chunk.id for r in euclid.search(QUERY, 2)] == ["seine", "paris"]


# --- edges ------------------------------------------------------------------


def test_empty_collection_returns_no_results(embedder: FakeEmbedder) -> None:
    """An empty collection yields no hits and a zero corpus size."""
    adapter = _retriever(embedder, [])

    assert adapter.corpus_size == 0
    assert adapter.search(QUERY, 5) == []


def test_k_larger_than_the_collection(embedder: FakeEmbedder, tiny_chunks: list[Chunk]) -> None:
    """Asking for more than the store holds returns what's there."""
    results = _retriever(embedder, tiny_chunks).search(QUERY, 99)

    assert len(results) == len(tiny_chunks)


def test_non_positive_k_returns_nothing(embedder: FakeEmbedder, tiny_chunks: list[Chunk]) -> None:
    """A non-positive k short-circuits before reaching Qdrant."""
    assert _retriever(embedder, tiny_chunks).search(QUERY, 0) == []


def test_search_is_deterministic(embedder: FakeEmbedder, tiny_chunks: list[Chunk]) -> None:
    """Identical inputs yield identical results."""
    adapter = _retriever(embedder, tiny_chunks)

    assert adapter.search(QUERY, 5) == adapter.search(QUERY, 5)


def test_ties_break_by_ascending_chunk_id(embedder: FakeEmbedder) -> None:
    """Chunks with identical text are ordered by id, not by point order."""
    duplicates = [Chunk(id=cid, text="identical text") for cid in ("z", "a", "m")]
    results = _retriever(embedder, duplicates).search("identical text", 3)

    assert [result.chunk.id for result in results] == ["a", "m", "z"]


# --- the explainer runs on top of it ----------------------------------------


def test_explain_chunk_works_over_the_adapter(
    embedder: FakeEmbedder, tiny_chunks: list[Chunk]
) -> None:
    """score_text is local, so occlusion attribution needs no store round-trip."""
    adapter = _retriever(embedder, tiny_chunks)
    top = adapter.search(QUERY, 1)[0]

    explanation = explain_chunk(adapter, QUERY, top)

    assert explanation.sentences
    assert sum(item.share for item in explanation.sentences) == pytest.approx(1.0)
    # A vector store has no lexical modality, so there is no split to report.
    assert explanation.split is None
