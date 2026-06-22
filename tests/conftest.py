"""Shared, fully-offline test fixtures.

Everything here uses the deterministic :class:`FakeEmbedder` and in-memory
corpora, so the suite never touches the network and is reproducible.
"""

from __future__ import annotations

import pytest

from why_this_chunk import (
    BM25Retriever,
    Chunk,
    Corpus,
    DenseRetriever,
    FakeEmbedder,
    FixedSizeChunker,
    HybridRetriever,
    SourceDocument,
)

_TINY_CHUNKS = [
    Chunk(id="paris", text="The capital of France is Paris. It lies on the Seine."),
    Chunk(id="python", text="Python is a programming language used for data science."),
    Chunk(id="eiffel", text="The Eiffel Tower is an iron landmark located in Paris."),
    Chunk(id="banana", text="Bananas are a yellow fruit that is rich in potassium."),
    Chunk(id="seine", text="The Seine is a river flowing through northern France."),
]


@pytest.fixture
def tiny_chunks() -> list[Chunk]:
    """A small fixed list of pre-chunked chunks."""
    return list(_TINY_CHUNKS)


@pytest.fixture
def tiny_corpus(tiny_chunks: list[Chunk]) -> Corpus:
    """A pre-chunked corpus (no provenance)."""
    return Corpus.from_chunks(tiny_chunks)


@pytest.fixture
def embedder() -> FakeEmbedder:
    """A deterministic offline embedder with a fixed seed."""
    return FakeEmbedder(dim=64, seed=7)


@pytest.fixture
def bm25(tiny_corpus: Corpus) -> BM25Retriever:
    """A BM25 retriever over the tiny corpus."""
    return BM25Retriever(tiny_corpus)


@pytest.fixture
def dense(tiny_corpus: Corpus, embedder: FakeEmbedder) -> DenseRetriever:
    """A dense retriever over the tiny corpus."""
    return DenseRetriever(tiny_corpus, embedder)


@pytest.fixture
def hybrid(dense: DenseRetriever, bm25: BM25Retriever) -> HybridRetriever:
    """A hybrid retriever blending the dense and BM25 modalities."""
    return HybridRetriever(dense, bm25, alpha=0.5)


@pytest.fixture
def source_docs() -> list[SourceDocument]:
    """Source documents long enough to be split by small chunk sizes."""
    return [
        SourceDocument(
            id="doc1",
            text=(
                "Photosynthesis converts light into chemical energy. "
                "Chlorophyll absorbs sunlight in the leaves of green plants. "
                "The process releases oxygen as a byproduct into the air. "
                "Glucose produced this way feeds the rest of the organism."
            ),
        ),
        SourceDocument(
            id="doc2",
            text=(
                "The mitochondria are the powerhouse of the cell organelle. "
                "They generate adenosine triphosphate through respiration. "
                "Cellular metabolism depends on this energy currency molecule."
            ),
        ),
    ]


@pytest.fixture
def chunker() -> FixedSizeChunker:
    """A deterministic fixed-size chunker with no overlap."""
    return FixedSizeChunker(overlap=0)
