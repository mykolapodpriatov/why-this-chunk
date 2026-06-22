"""Optional cross-encoder reranking (``[st]`` extra).

A :class:`CrossEncoderReranker` re-scores ``(query, chunk)`` pairs with a
cross-encoder model, and :class:`RerankingRetriever` wraps any base retriever to
apply that reranking to a widened candidate pool. The wrapper advertises a
``supports_rerank`` property so the counterfactual ``rerank`` axis becomes
evaluable; without the wrapper that axis is reported unevaluable.

This module is import-guarded: constructing :class:`CrossEncoderReranker`
without ``sentence-transformers`` raises a clear :class:`ImportError`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from why_this_chunk.config import RetrievalConfig
from why_this_chunk.corpus import Corpus
from why_this_chunk.retrievers import Retriever
from why_this_chunk.types import ScoredChunk

__all__ = ["CrossEncoderReranker", "Reranker", "RerankingRetriever"]


@runtime_checkable
class Reranker(Protocol):
    """Re-scores ``(query, text)`` pairs; higher is more relevant."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per text in ``texts``."""
        ...


class CrossEncoderReranker:
    """A cross-encoder reranker backed by ``sentence-transformers``.

    Args:
        model_name: A ``CrossEncoder`` model id.
        device: Optional torch device string.

    Raises:
        ImportError: If the ``[st]`` extra is not installed.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - only without extra
            raise ImportError(
                "CrossEncoderReranker requires the optional 'st' extra. "
                "Install it with: pip install 'why-this-chunk[st]'"
            ) from exc
        self._model = CrossEncoder(model_name, device=device)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Score each text against ``query`` with the cross-encoder."""
        if not texts:
            return []
        pairs = [[query, text] for text in texts]
        scores = self._model.predict(pairs)
        return [float(value) for value in scores]


class RerankingRetriever:
    """Wraps a base retriever to rerank a widened candidate pool.

    The base retriever's top ``pool_size`` candidates are re-scored by the
    reranker; ties broken by ascending chunk id for determinism.

    Args:
        base: The retriever providing the initial candidate pool.
        reranker: The reranker used to re-score candidates.
        pool_size: How many base candidates to rerank (capped at corpus size).
        active: Whether reranking is applied. When ``False`` this delegates to
            the base retriever unchanged (used to model ``rerank=False``).

    Raises:
        ValueError: If ``pool_size`` is not positive.
    """

    def __init__(
        self,
        base: Retriever,
        reranker: Reranker,
        pool_size: int = 50,
        active: bool = True,
    ) -> None:
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")
        self._base = base
        self._reranker = reranker
        self._pool_size = pool_size
        self._active = active

    @property
    def corpus(self) -> Corpus | None:
        """The base retriever's corpus, if it exposes one."""
        base_corpus = getattr(self._base, "corpus", None)
        return base_corpus if isinstance(base_corpus, Corpus) else None

    @property
    def corpus_size(self) -> int:
        """Number of chunks available (from the base retriever)."""
        return self._base.corpus_size

    @property
    def supports_components(self) -> bool:
        """Components are not produced by the reranking layer."""
        return False

    @property
    def supports_reindex(self) -> bool:
        """Reindex is supported when the base retriever supports it."""
        return self._base.supports_reindex

    @property
    def supports_rerank(self) -> bool:
        """Advertises a configured reranker for the counterfactual axis."""
        return True

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Return the top-``k`` results, reranked when active."""
        if not self._active:
            return self._base.search(query, k)
        pool = self._base.search(query, min(self._pool_size, self._base.corpus_size))
        if not pool:
            return []
        scores = self._reranker.score(query, [scored.chunk.text for scored in pool])
        order = sorted(
            range(len(pool)),
            key=lambda i: (-scores[i], pool[i].chunk.id),
        )
        limit = max(0, min(k, len(order)))
        results: list[ScoredChunk] = []
        for rank, idx in enumerate(order[:limit]):
            scored = pool[idx]
            results.append(
                ScoredChunk(
                    chunk=scored.chunk,
                    score=float(scores[idx]),
                    rank=rank,
                    components=scored.components,
                )
            )
        return results

    def reindex(self, config: RetrievalConfig) -> RerankingRetriever:
        """Return a new reranking retriever under ``config``.

        Toggles the ``active`` flag from ``config.rerank`` and reindexes the
        base retriever when its config axes (e.g. ``chunk_size``) change.
        """
        base = self._base.reindex(config) if self._base.supports_reindex else self._base
        return RerankingRetriever(
            base,
            self._reranker,
            pool_size=self._pool_size,
            active=config.rerank,
        )
