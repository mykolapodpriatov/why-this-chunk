"""Dense cosine retriever over L2-normalized embeddings.

Because the :class:`~why_this_chunk.embedders.Embedder` contract guarantees
unit-norm rows, cosine similarity is a plain dot product. The default backend is
numpy; an optional FAISS backend (``[faiss]`` extra) accelerates large corpora
while returning identical rankings.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from why_this_chunk.config import RetrievalConfig
from why_this_chunk.corpus import Corpus
from why_this_chunk.embedders import Embedder
from why_this_chunk.retrievers.bm25 import _rank, _reindex_corpus
from why_this_chunk.source import Chunker
from why_this_chunk.types import Chunk, ScoreComponents, ScoredChunk

__all__ = ["DenseRetriever"]


class DenseRetriever:
    """A cosine-similarity dense retriever.

    Args:
        corpus: The corpus to search.
        embedder: The embedder used for chunks and queries.
        chunker: Optional chunker enabling ``chunk_size`` reindexing.
        config: The active retrieval configuration.
        use_faiss: Use the FAISS backend when available. Falls back to numpy if
            FAISS is not installed. Rankings are identical either way.
    """

    def __init__(
        self,
        corpus: Corpus,
        embedder: Embedder,
        chunker: Chunker | None = None,
        config: RetrievalConfig | None = None,
        use_faiss: bool = False,
    ) -> None:
        self._corpus = corpus
        self._embedder = embedder
        self._chunker = chunker
        self._config = config or RetrievalConfig()
        self._chunks: list[Chunk] = corpus.chunks
        texts = [chunk.text for chunk in self._chunks]
        self._matrix: NDArray[np.float32] = (
            embedder.encode(texts) if texts else np.zeros((0, embedder.dim), dtype=np.float32)
        )
        # ``faiss`` ships no type stubs, so the index is intentionally ``Any``.
        self._index: Any = _build_faiss(self._matrix) if use_faiss else None

    @property
    def corpus(self) -> Corpus:
        """The corpus being searched."""
        return self._corpus

    @property
    def embedder(self) -> Embedder:
        """The embedder used for chunks and queries."""
        return self._embedder

    @property
    def config(self) -> RetrievalConfig:
        """The active retrieval configuration."""
        return self._config

    @property
    def corpus_size(self) -> int:
        """Number of chunks available."""
        return len(self._chunks)

    @property
    def supports_components(self) -> bool:
        """Dense results carry a dense-only :class:`ScoreComponents`."""
        return True

    @property
    def supports_reindex(self) -> bool:
        """Reindex is supported (``chunk_size`` also needs corpus provenance)."""
        return True

    @property
    def backend(self) -> str:
        """``"faiss"`` if the FAISS index is active, else ``"numpy"``."""
        return "faiss" if self._index is not None else "numpy"

    def raw_scores(self, query: str) -> NDArray[np.float64]:
        """Return cosine similarities for ``query`` in corpus chunk order.

        Args:
            query: The query string.

        Returns:
            A 1-D array of length ``corpus_size`` in ``[-1, 1]`` (all zeros for
            an empty corpus).
        """
        if not self._chunks:
            return np.zeros(0, dtype=np.float64)
        vector = self._embedder.encode([query])[0]
        scores: NDArray[np.float64] = (self._matrix @ vector).astype(np.float64)
        return scores

    def score_text(self, query: str, text: str) -> float:
        """Cosine similarity of ``text`` to ``query`` (for attribution)."""
        query_vec = self._embedder.encode([query])[0]
        text_vec = self._embedder.encode([text])[0]
        return float(np.dot(query_vec, text_vec))

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Return the top-``k`` chunks by cosine similarity (see protocol)."""
        if not self._chunks:
            return []
        if self._index is not None:
            return self._search_faiss(query, k)
        scores = self.raw_scores(query)
        return _rank(self._chunks, scores, k, modality="dense")

    def _search_faiss(self, query: str, k: int) -> list[ScoredChunk]:
        """FAISS-backed search; identical ranking to the numpy path."""
        assert self._index is not None
        vector = self._embedder.encode([query]).astype(np.float32)
        limit = max(0, min(k, len(self._chunks)))
        if limit == 0:
            return []
        distances, indices = self._index.search(vector, limit)
        results: list[ScoredChunk] = []
        for rank, (idx, score) in enumerate(
            zip(indices[0].tolist(), distances[0].tolist(), strict=True)
        ):
            chunk = self._chunks[idx]
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=float(score),
                    rank=rank,
                    components=ScoreComponents(dense=None, dense_raw=float(score)),
                )
            )
        return results

    def reindex(self, config: RetrievalConfig) -> DenseRetriever:
        """Return a new dense retriever under ``config`` (see protocol)."""
        corpus = _reindex_corpus(self._corpus, self._chunker, self._config, config)
        return DenseRetriever(
            corpus,
            self._embedder,
            chunker=self._chunker,
            config=config,
            use_faiss=self._index is not None,
        )


def _build_faiss(matrix: NDArray[np.float32]) -> Any:
    """Build a FAISS inner-product index, or ``None`` if FAISS is unavailable."""
    if matrix.shape[0] == 0:
        return None
    try:
        import faiss
    except ImportError:  # pragma: no cover - exercised only without extra
        return None
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(np.ascontiguousarray(matrix))
    return index
