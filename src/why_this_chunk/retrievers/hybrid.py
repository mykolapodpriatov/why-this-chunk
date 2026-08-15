"""Hybrid retriever combining a dense and a lexical modality.

The combination rule is explicit and documented:

1. Compute raw scores from each modality over the **full candidate pool** (the
   whole corpus).
2. Min-max normalize each modality's scores to ``[0, 1]``. **Degenerate
   fallback:** when ``max == min`` for a modality (identical scores, or a pool
   of size 1), every normalized score for that modality is set to ``0.5`` (see
   :data:`~why_this_chunk._scoring.DEGENERATE_NORM_VALUE`) so the formula is
   always well-defined.
3. ``score = alpha * dense_n + (1 - alpha) * lexical_n``.

Ties are broken by ascending chunk id for determinism.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from why_this_chunk._scoring import min_max_normalize
from why_this_chunk.config import RetrievalConfig
from why_this_chunk.corpus import Corpus
from why_this_chunk.retrievers.bm25 import BM25Retriever
from why_this_chunk.retrievers.dense import DenseRetriever
from why_this_chunk.types import Chunk, ScoreComponents, ScoredChunk

__all__ = ["HybridRetriever"]

#: Default mixing weight when none is supplied (equal blend).
DEFAULT_ALPHA: float = 0.5


class HybridRetriever:
    """Weighted blend of a dense and a lexical retriever.

    Args:
        dense: The dense modality. Must share the hybrid's corpus.
        lexical: The lexical modality. Must share the hybrid's corpus.
        alpha: Mixing weight in ``[0, 1]`` applied to the dense modality;
            ``1 - alpha`` is applied to the lexical modality. Defaults to
            :data:`DEFAULT_ALPHA`.

    Raises:
        ValueError: If ``alpha`` is outside ``[0, 1]`` or the two modalities do
            not cover the same chunks (by id and order).
    """

    def __init__(
        self,
        dense: DenseRetriever,
        lexical: BM25Retriever,
        alpha: float = DEFAULT_ALPHA,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        dense_ids = [c.id for c in dense.corpus.chunks]
        lexical_ids = [c.id for c in lexical.corpus.chunks]
        if dense_ids != lexical_ids:
            raise ValueError(
                "dense and lexical modalities must cover the same chunks in the same order"
            )
        self._dense = dense
        self._lexical = lexical
        self._alpha = alpha
        self._chunks: list[Chunk] = dense.corpus.chunks
        base = dense.config
        self._config = base.with_updates(alpha=alpha)

    @property
    def corpus(self) -> Corpus:
        """The shared corpus being searched."""
        return self._dense.corpus

    @property
    def alpha(self) -> float:
        """The dense mixing weight."""
        return self._alpha

    @property
    def config(self) -> RetrievalConfig:
        """The active retrieval configuration (with ``alpha`` set)."""
        return self._config

    @property
    def corpus_size(self) -> int:
        """Number of chunks available."""
        return len(self._chunks)

    @property
    def supports_components(self) -> bool:
        """Hybrid results always carry a full dense/lexical split."""
        return True

    @property
    def supports_reindex(self) -> bool:
        """Reindex is supported (handles the ``alpha`` and ``chunk_size`` axes)."""
        return True

    @property
    def backend(self) -> str:
        """The dense modality's index backend (``faiss`` or ``numpy``)."""
        return self._dense.backend

    def _combined(
        self, query: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return raw and normalized dense/lexical scores for the full pool."""
        dense_raw = self._dense.raw_scores(query)
        lexical_raw = self._lexical.raw_scores(query)
        dense_n = min_max_normalize(dense_raw)
        lexical_n = min_max_normalize(lexical_raw)
        return dense_raw, lexical_raw, dense_n, lexical_n

    def raw_scores(self, query: str) -> NDArray[np.float64]:
        """Return combined hybrid scores for ``query`` in corpus order."""
        if not self._chunks:
            return np.zeros(0, dtype=np.float64)
        _, _, dense_n, lexical_n = self._combined(query)
        return self._alpha * dense_n + (1.0 - self._alpha) * lexical_n

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Return the top-``k`` chunks by combined score (see protocol).

        Each result carries a full :class:`ScoreComponents` (raw + normalized
        dense and lexical values plus ``alpha``) so the contribution split can
        be computed downstream.
        """
        if not self._chunks:
            return []
        dense_raw, lexical_raw, dense_n, lexical_n = self._combined(query)
        combined = self._alpha * dense_n + (1.0 - self._alpha) * lexical_n
        order = sorted(
            range(len(self._chunks)),
            key=lambda i: (-float(combined[i]), self._chunks[i].id),
        )
        limit = max(0, min(k, len(order)))
        results: list[ScoredChunk] = []
        for rank, idx in enumerate(order[:limit]):
            components = ScoreComponents(
                dense=float(dense_n[idx]),
                lexical=float(lexical_n[idx]),
                alpha=self._alpha,
                dense_raw=float(dense_raw[idx]),
                lexical_raw=float(lexical_raw[idx]),
            )
            results.append(
                ScoredChunk(
                    chunk=self._chunks[idx],
                    score=float(combined[idx]),
                    rank=rank,
                    components=components,
                )
            )
        return results

    def reindex(self, config: RetrievalConfig) -> HybridRetriever:
        """Return a new hybrid retriever under ``config`` (see protocol).

        Handles both the ``alpha`` axis (re-blend) and the ``chunk_size`` axis
        (delegated to the underlying modalities, which require provenance).
        """
        new_alpha = config.alpha if config.alpha is not None else self._alpha
        dense = self._dense.reindex(config)
        lexical = self._lexical.reindex(config)
        return HybridRetriever(dense, lexical, alpha=new_alpha)
