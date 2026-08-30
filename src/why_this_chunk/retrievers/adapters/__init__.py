"""Adapters for retrievers whose index lives in an external vector store.

The built-in retrievers own their corpus in-process. These adapters instead wrap
a store that already holds the chunks and their embeddings, so an existing index
can be explained without re-ingesting it.

Two consequences follow from the index living elsewhere, and both are advertised
honestly through the :class:`~why_this_chunk.retrievers.Retriever` capability
model rather than faked:

* ``supports_reindex`` is ``False``. Changing ``chunk_size`` means re-chunking
  and re-upserting in the store, which is the operator's job, not this library's.
  The ``chunk_size`` counterfactual axis is therefore reported *unevaluable*.
* ``supports_components`` is ``True``, but the split is dense-only — these are
  vector stores, and there is no lexical modality to attribute to.

Sentence-level occlusion attribution *does* work: ``score_text`` re-embeds
locally with the same embedder, so ``explain`` behaves as it does for
:class:`~why_this_chunk.retrievers.dense.DenseRetriever`.

Scores are always cosine similarities, matching ``score_text`` and the built-in
dense retriever. This relies on the :class:`~why_this_chunk.embedders.Embedder`
contract that vectors are L2-normalized: for unit vectors every distance metric
these stores offer converts back to cosine exactly.

The heavy client libraries are **import-guarded**: importing this package never
pulls in ``qdrant_client`` or ``psycopg``. Instantiating an adapter without its
backend raises :class:`MissingDependencyError`, naming the extra to install.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from why_this_chunk.config import RetrievalConfig
from why_this_chunk.embedders import Embedder
from why_this_chunk.retrievers import Retriever
from why_this_chunk.types import Chunk, ScoreComponents, ScoredChunk

__all__ = ["MissingDependencyError", "VectorStoreRetriever", "rank_store_hits", "require"]


class MissingDependencyError(ImportError):
    """Raised when an adapter's optional backend dependency is not installed."""

    def __init__(self, backend: str, extra: str) -> None:
        self.backend = backend
        self.extra = extra
        super().__init__(
            f"the {backend!r} adapter requires the optional '{extra}' extra; "
            f'install it with: pip install "why-this-chunk[{extra}]"'
        )


def require(module: str, *, backend: str, extra: str) -> object:
    """Import ``module`` or raise a clear :class:`MissingDependencyError`.

    Args:
        module: The importable backend module name.
        backend: Human-readable backend name for the error.
        extra: The pip extra that provides the dependency.

    Returns:
        The imported module object.

    Raises:
        MissingDependencyError: If the module cannot be imported.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised via adapter ctors
        raise MissingDependencyError(backend, extra) from exc


def rank_store_hits(hits: Iterable[tuple[Chunk, float]], k: int) -> list[ScoredChunk]:
    """Order ``(chunk, score)`` pairs into the top-``k`` results.

    Sorted by descending score, ties broken by ascending chunk id, with 0-based
    contiguous ranks — the same contract ``retrievers/bm25.py`` applies to the
    built-in retrievers. Re-sorting locally rather than trusting the store's
    order is what makes ties deterministic across backends.
    """
    ordered = sorted(hits, key=lambda hit: (-hit[1], hit[0].id))
    return [
        ScoredChunk(
            chunk=chunk,
            score=float(score),
            rank=rank,
            components=ScoreComponents(dense=None, dense_raw=float(score)),
        )
        for rank, (chunk, score) in enumerate(ordered[: max(0, k)])
    ]


class VectorStoreRetriever(abc.ABC):
    """Shared behaviour for adapters over an external vector store.

    Subclasses supply two things: how to count the stored chunks, and how to
    answer one nearest-neighbor query as ``(chunk, cosine_similarity)`` pairs.
    Everything else — capabilities, ranking, attribution scoring — is the same
    for every store.

    Args:
        embedder: The embedder that produced the stored vectors. Query vectors
            must come from the same model, or the similarities are meaningless.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    @property
    def embedder(self) -> Embedder:
        """The embedder used for queries and attribution."""
        return self._embedder

    @property
    def corpus_size(self) -> int:
        """Number of chunks held in the store."""
        return self._count()

    @property
    def supports_components(self) -> bool:
        """Results carry a dense-only :class:`ScoreComponents`."""
        return True

    @property
    def supports_reindex(self) -> bool:
        """Always ``False``: the index is owned by the store, not this process."""
        return False

    def reindex(self, config: RetrievalConfig) -> Retriever:
        """Not supported — see :attr:`supports_reindex`.

        Raises:
            NotImplementedError: Always. Re-chunking or re-embedding means
                rewriting the store, which this adapter deliberately will not do.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot reindex: the index lives in the store, "
            "not in this process. Re-embed and re-upsert there, then wrap the "
            "result in a new adapter."
        )

    def score_text(self, query: str, text: str) -> float:
        """Cosine similarity of ``text`` to ``query``, for occlusion attribution.

        Computed locally from the embedder, never from the store: attribution
        scores hundreds of occluded variants that were never indexed.
        """
        vectors = self._embedder.encode([query, text])
        return float(np.dot(vectors[0], vectors[1]))

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Return the top-``k`` chunks by cosine similarity (see protocol)."""
        if k <= 0:
            return []
        vector = self._embedder.encode([query])[0]
        return rank_store_hits(self._query(vector, k), k)

    @abc.abstractmethod
    def _count(self) -> int:
        """Return the number of chunks in the store."""

    @abc.abstractmethod
    def _query(self, vector: NDArray[np.float32], k: int) -> Sequence[tuple[Chunk, float]]:
        """Return up to ``k`` ``(chunk, cosine_similarity)`` pairs for ``vector``."""
