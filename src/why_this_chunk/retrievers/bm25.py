"""Lexical retriever backed by ``rank_bm25``.

Scores are Okapi BM25 over word-character tokens (the same tokenization the
:class:`~why_this_chunk.embedders.fake.FakeEmbedder` uses). The retriever can
re-chunk under a new :class:`~why_this_chunk.config.RetrievalConfig` when the
corpus carries provenance, enabling the ``chunk_size`` axis.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi

from why_this_chunk._text import tokenize
from why_this_chunk.config import RetrievalConfig
from why_this_chunk.corpus import Corpus
from why_this_chunk.source import Chunker
from why_this_chunk.types import Chunk, ScoreComponents, ScoredChunk

__all__ = ["BM25Retriever"]


class BM25Retriever:
    """An Okapi BM25 lexical retriever.

    Args:
        corpus: The corpus to search.
        chunker: Optional chunker enabling ``chunk_size`` reindexing. Required
            (together with corpus provenance) for the ``chunk_size`` axis.
        config: The active retrieval configuration.
    """

    def __init__(
        self,
        corpus: Corpus,
        chunker: Chunker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._corpus = corpus
        self._chunker = chunker
        self._config = config or RetrievalConfig()
        self._chunks: list[Chunk] = corpus.chunks
        tokenized = [tokenize(chunk.text) for chunk in self._chunks]
        # ``BM25Okapi`` divides by the total token count when computing IDF, so it
        # raises ``ZeroDivisionError`` not only for zero chunks but also for a
        # corpus where *every* chunk tokenizes to nothing (e.g. all-whitespace or
        # punctuation-only text). Treat both as the empty sentinel and short-
        # circuit scoring to well-defined zeros.
        self._empty = not any(tokenized)
        self._bm25 = None if self._empty else BM25Okapi(tokenized)

    @property
    def corpus(self) -> Corpus:
        """The corpus being searched."""
        return self._corpus

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
        """BM25 results carry a lexical-only :class:`ScoreComponents`."""
        return True

    @property
    def supports_reindex(self) -> bool:
        """Reindex is supported (``chunk_size`` also needs corpus provenance)."""
        return True

    def raw_scores(self, query: str) -> NDArray[np.float64]:
        """Return BM25 scores for ``query`` aligned to corpus chunk order.

        Args:
            query: The query string.

        Returns:
            A 1-D array of length ``corpus_size`` (all zeros for an empty
            corpus or empty query). A corpus with chunks but no tokens (all
            empty/whitespace text) yields a length-``corpus_size`` zero vector,
            preserving alignment with corpus order.
        """
        if self._empty or self._bm25 is None:
            return np.zeros(len(self._chunks), dtype=np.float64)
        tokens = tokenize(query)
        if not tokens:
            return np.zeros(len(self._chunks), dtype=np.float64)
        return np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)

    def score_text(self, query: str, text: str) -> float:
        """Score an arbitrary ``text`` against ``query`` (for attribution).

        Used by occlusion attribution to re-score modified chunk text without
        rebuilding the index. The score reuses the corpus' IDF statistics.

        Args:
            query: The query string.
            text: The candidate text.

        Returns:
            The BM25 score of ``text`` for ``query`` (0.0 when the index or
            query is empty).
        """
        if self._empty or self._bm25 is None:
            return 0.0
        tokens = tokenize(query)
        if not tokens:
            return 0.0
        doc_tokens = tokenize(text)
        return _score_single(self._bm25, tokens, doc_tokens)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Return the top-``k`` chunks by BM25 score (see protocol)."""
        scores = self.raw_scores(query)
        return _rank(self._chunks, scores, k, modality="lexical")

    def reindex(self, config: RetrievalConfig) -> BM25Retriever:
        """Return a new BM25 retriever under ``config``.

        Re-chunks the corpus when ``config.chunk_size`` differs and provenance
        plus a chunker are available; otherwise reuses the current corpus.

        Raises:
            NotImplementedError: If ``chunk_size`` changed but the corpus lacks
                provenance or no chunker is configured (the caller is expected
                to gate this via capability/provenance checks).
        """
        corpus = _reindex_corpus(self._corpus, self._chunker, self._config, config)
        return BM25Retriever(corpus, chunker=self._chunker, config=config)


def _score_single(bm25: BM25Okapi, query_tokens: list[str], doc_tokens: list[str]) -> float:
    """BM25 score of one ad-hoc document using the index' IDF and avgdl."""
    if not doc_tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in doc_tokens:
        counts[token] = counts.get(token, 0) + 1
    doc_len = len(doc_tokens)
    score = 0.0
    k1 = bm25.k1
    b = bm25.b
    avgdl = bm25.avgdl
    for token in query_tokens:
        if token not in counts:
            continue
        idf = bm25.idf.get(token, 0.0)
        freq = counts[token]
        denom = freq + k1 * (1.0 - b + b * doc_len / avgdl)
        score += idf * (freq * (k1 + 1.0)) / denom
    return float(score)


def _rank(
    chunks: list[Chunk],
    scores: NDArray[np.float64],
    k: int,
    *,
    modality: Literal["lexical", "dense"],
) -> list[ScoredChunk]:
    """Order chunks by descending score, breaking ties by ascending id.

    Every result carries a single-modality :class:`ScoreComponents` recording its
    raw score under ``modality``, so a standalone retriever honors its advertised
    ``supports_components`` capability on both the lexical and dense paths.
    """
    if not chunks or scores.size == 0:
        return []
    order = sorted(
        range(len(chunks)),
        key=lambda i: (-float(scores[i]), chunks[i].id),
    )
    limit = max(0, min(k, len(order)))
    results: list[ScoredChunk] = []
    for rank, idx in enumerate(order[:limit]):
        value = float(scores[idx])
        components = (
            ScoreComponents(lexical=None, lexical_raw=value)
            if modality == "lexical"
            else ScoreComponents(dense=None, dense_raw=value)
        )
        results.append(
            ScoredChunk(chunk=chunks[idx], score=value, rank=rank, components=components)
        )
    return results


def _reindex_corpus(
    corpus: Corpus,
    chunker: Chunker | None,
    current: RetrievalConfig,
    new: RetrievalConfig,
) -> Corpus:
    """Re-chunk the corpus for ``new`` if ``chunk_size`` changed, else reuse."""
    if new.chunk_size == current.chunk_size:
        return corpus
    if not corpus.has_provenance or chunker is None:
        raise NotImplementedError(
            "chunk_size reindexing requires a corpus built with provenance "
            "(SourceDocuments + Chunker)"
        )
    return Corpus.from_sources(corpus.source_documents, chunker, new.chunk_size)
