"""Tests for the failure taxonomy: one corpus per class, decision order, and
the explicit out_ranked vs embedding_blind_spot collision.

The realistic classes use the built-in retrievers with :class:`FakeEmbedder`.
The precise ``out_ranked``/``embedding_blind_spot`` boundary uses a small,
fully-deterministic in-process retriever stub so rank and modality scores can be
pinned exactly — still entirely offline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from why_this_chunk import (
    BM25Retriever,
    Chunk,
    Corpus,
    DenseRetriever,
    FakeEmbedder,
    FixedSizeChunker,
    HybridRetriever,
    RetrievalConfig,
    SourceDocument,
    diagnose,
    large_k,
)
from why_this_chunk.config import RetrievalConfig as Config
from why_this_chunk.taxonomy import (
    DEFAULT_DENSE_THRESHOLD,
    DEFAULT_LEXICAL_THRESHOLD,
)
from why_this_chunk.types import FailureClass, ScoreComponents, ScoredChunk


class _ScriptedHybrid:
    """A deterministic hybrid-style retriever with hand-set raw scores.

    Implements the retriever protocol plus the ``corpus`` and modality-score
    introspection the taxonomy uses, so ranks and dense/lexical evidence are
    fully controllable. Mirrors how the built-in hybrid exposes ``_dense`` and
    ``_lexical`` via small inner shims.
    """

    def __init__(
        self,
        corpus: Corpus,
        dense_raw: dict[str, float],
        lexical_raw: dict[str, float],
        alpha: float = 0.5,
    ) -> None:
        self._corpus = corpus
        self._ids = [c.id for c in corpus.chunks]
        self._dense_raw = np.array([dense_raw[i] for i in self._ids], dtype=np.float64)
        self._lexical_raw = np.array([lexical_raw[i] for i in self._ids], dtype=np.float64)
        self._alpha = alpha
        self._dense = _Modality(self._dense_raw)
        self._lexical = _Modality(self._lexical_raw)

    @property
    def corpus(self) -> Corpus:
        return self._corpus

    @property
    def corpus_size(self) -> int:
        return len(self._ids)

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def supports_components(self) -> bool:
        return True

    @property
    def supports_reindex(self) -> bool:
        return False

    def _combined(self) -> NDArray[np.float64]:
        from why_this_chunk._scoring import min_max_normalize

        dn = min_max_normalize(self._dense_raw)
        ln = min_max_normalize(self._lexical_raw)
        return self._alpha * dn + (1.0 - self._alpha) * ln

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        combined = self._combined()
        order = sorted(
            range(len(self._ids)),
            key=lambda i: (-float(combined[i]), self._ids[i]),
        )
        chunks = self._corpus.chunks
        limit = max(0, min(k, len(order)))
        return [
            ScoredChunk(
                chunk=chunks[idx],
                score=float(combined[idx]),
                rank=rank,
                components=ScoreComponents(alpha=self._alpha),
            )
            for rank, idx in enumerate(order[:limit])
        ]

    def reindex(self, config: RetrievalConfig) -> _ScriptedHybrid:  # pragma: no cover
        raise NotImplementedError


class _Modality:
    """Shim exposing ``raw_scores`` over a fixed score vector."""

    def __init__(self, raw: NDArray[np.float64]) -> None:
        self._raw = raw

    def raw_scores(self, query: str) -> NDArray[np.float64]:
        return self._raw


#: The small target text region (character offsets into the stub source doc)
#: that the ``lost_to_chunking`` stubs model: it must be *fully contained* by a
#: covering window at ``good_size`` and split across windows otherwise.
_TARGET_SPAN: tuple[int, int] = (100, 150)


class _ReindexableCorpusRetriever:
    """Deterministic retriever whose ranking depends on the chunk_size config.

    It models a corpus that splits a small target region (``_TARGET_SPAN``)
    across chunks at the current ``chunk_size`` (so no covering chunk ranks
    within top_k) but reunites it under a window that *fully contains* it at a
    specific ``good_size`` (where that covering window lands at rank 0). This
    exercises the ``lost_to_chunking`` branch and the ``chunk_size`` axis,
    honoring full-span containment, without depending on BM25/embedding numerics.

    The named ``expected`` chunk carries ``_TARGET_SPAN`` as its span so the
    diagnosis maps it back to that region; the covering window returned at
    ``good_size`` spans ``(0, good_size)`` which contains ``_TARGET_SPAN``.
    """

    def __init__(self, good_size: int, current_size: int = 512) -> None:
        self._good_size = good_size
        self._config = RetrievalConfig(chunk_size=current_size)
        self._doc = SourceDocument(id="doc", text="x" * 2048)
        self._expected = Chunk(
            id="target",
            text="x" * (_TARGET_SPAN[1] - _TARGET_SPAN[0]),
            source_document_id="doc",
            span=_TARGET_SPAN,
        )
        self._corpus = self._build(current_size)

    def _build(self, size: int) -> Corpus:
        # The named target chunk is always present (membership holds); the rest
        # of the corpus is the chunker's fixed windows over the doc.
        chunker = FixedSizeChunker(overlap=0)
        windows = chunker.chunk([self._doc], size)
        return Corpus([self._expected, *windows], source_documents=[self._doc])

    @property
    def corpus(self) -> Corpus:
        return self._corpus

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)

    @property
    def supports_components(self) -> bool:
        return False

    @property
    def supports_reindex(self) -> bool:
        return True

    @property
    def expected_id(self) -> str:
        return self._expected.id

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        # Only when chunked at good_size does a window fully containing the
        # target region surface (at rank 0); otherwise return an unrelated later
        # window so the target region is absent from the top results.
        if self._config.chunk_size == self._good_size:
            covering = Chunk(
                id=f"doc::{self._good_size}::cover",
                text="x" * self._good_size,
                source_document_id="doc",
                span=(0, self._good_size),
            )
            head = [covering]
        else:
            windows = [c for c in self._corpus.chunks if c.id != self._expected.id]
            head = windows[1:2] if len(windows) > 1 else []
        return [
            ScoredChunk(chunk=c, score=1.0 - rank, rank=rank) for rank, c in enumerate(head[:k])
        ]

    def reindex(self, config: RetrievalConfig) -> _ReindexableCorpusRetriever:
        clone = _ReindexableCorpusRetriever(self._good_size, current_size=config.chunk_size)
        return clone


class _SplitAcrossWindowsRetriever:
    """Provenance-carrying retriever whose re-chunk windows only *overlap* the
    target span (never fully contain it).

    Models the case the containment rule guards against: at every swept size the
    surfacing window covers only part of ``_TARGET_SPAN`` (the rest spills into
    an adjacent window), so the expected text is never surfaced intact and the
    failure is NOT ``lost_to_chunking``.
    """

    def __init__(self, current_size: int = 512) -> None:
        self._config = RetrievalConfig(chunk_size=current_size)
        self._doc = SourceDocument(id="doc", text="x" * 2048)
        self._expected = Chunk(
            id="target",
            text="x" * (_TARGET_SPAN[1] - _TARGET_SPAN[0]),
            source_document_id="doc",
            span=_TARGET_SPAN,
        )
        self._corpus = Corpus([self._expected], source_documents=[self._doc])

    @property
    def corpus(self) -> Corpus:
        return self._corpus

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)

    @property
    def supports_components(self) -> bool:
        return False

    @property
    def supports_reindex(self) -> bool:
        return True

    @property
    def expected_id(self) -> str:
        return self._expected.id

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        # Always surface a window that starts inside the target span and ends
        # before it does: it overlaps the target but never contains it.
        exp_start, exp_end = _TARGET_SPAN
        partial = Chunk(
            id=f"doc::{self._config.chunk_size}::partial",
            text="x" * ((exp_end - 10) - (exp_start + 10)),
            source_document_id="doc",
            span=(exp_start + 10, exp_end - 10),
        )
        return [ScoredChunk(chunk=partial, score=1.0, rank=0)][:k]

    def reindex(self, config: RetrievalConfig) -> _SplitAcrossWindowsRetriever:
        return _SplitAcrossWindowsRetriever(current_size=config.chunk_size)


def test_split_across_windows_is_not_lost_to_chunking() -> None:
    """Regression: when the expected span is split across re-chunked windows (no
    single window fully contains it), the failure must NOT be classified
    ``lost_to_chunking`` — overlap alone is insufficient under the containment
    rule.
    """
    retriever = _SplitAcrossWindowsRetriever(current_size=512)
    expected = retriever.expected_id
    result = diagnose(retriever, "q", expected, Config(top_k=1, chunk_size=512))
    assert result.failure_class is not FailureClass.LOST_TO_CHUNKING
    assert "rechunk_size" not in result.evidence


def test_lost_to_chunking_positive_via_reindex() -> None:
    """With provenance + reindex, a window *fully containing* the target text
    that surfaces at a swept size is classified ``lost_to_chunking`` (branch 2),
    ahead of branches 3/4.
    """
    # good_size=256 yields a covering window (0, 256) that fully contains the
    # target region (100, 150); the smaller 128 window does not.
    retriever = _ReindexableCorpusRetriever(good_size=256, current_size=512)
    expected = retriever.expected_id
    # Under the current 512 config the target region is not in top_k.
    assert all(s.chunk.id != expected for s in retriever.search("q", 1))
    result = diagnose(retriever, "q", expected, Config(top_k=1, chunk_size=512))
    assert result.failure_class is FailureClass.LOST_TO_CHUNKING
    assert FailureClass.LOST_TO_CHUNKING not in result.unevaluable
    assert result.evidence["rechunk_size"] == 256


def test_large_k_formula() -> None:
    assert large_k(top_k=2, corpus_size=100) == 20
    assert large_k(top_k=5, corpus_size=8) == 8  # capped at corpus size


def test_missing_from_index(hybrid: HybridRetriever) -> None:
    result = diagnose(hybrid, "Paris", "does-not-exist", Config(top_k=2))
    assert result.failure_class is FailureClass.MISSING_FROM_INDEX
    assert "not present" in str(result.evidence["note"])


def test_out_ranked_via_real_retriever() -> None:
    # 'target' shares query terms but many near-duplicate distractors outrank it
    # at a small top_k while it remains within large_k.
    chunks = [Chunk(id="target", text="quantum entanglement spooky action distance")]
    for i in range(8):
        chunks.append(Chunk(id=f"distractor{i}", text="quantum entanglement spooky action remote"))
    corpus = Corpus.from_chunks(chunks)
    retriever = BM25Retriever(corpus)
    result = diagnose(retriever, "quantum entanglement spooky action", "target", Config(top_k=1))
    assert result.failure_class in {FailureClass.OUT_RANKED, FailureClass.EMBEDDING_BLIND_SPOT}
    # With small top_k and a large_k covering the whole corpus, target appears
    # within large_k => out_ranked.
    assert result.evidence["rank_at_large_k"] is not None
    assert result.failure_class is FailureClass.OUT_RANKED


def test_out_ranked_vs_blind_spot_collision() -> None:
    """The documented collision: a chunk at rank top_k+2 (inside large_k) with a
    sub-threshold dense score and high lexical score must be ``out_ranked`` —
    branch 3 fires on large_k membership, NOT ``embedding_blind_spot``.
    """
    top_k = 2
    # Build a corpus of 10 chunks. 'target' must land at rank top_k+2 = 4.
    ids = [f"c{i}" for i in range(10)]
    chunks = [Chunk(id=i, text=f"text {i}") for i in ids]
    corpus = Corpus.from_chunks(chunks)

    target = "c9"
    # Give 4 chunks higher combined score than target, so target is rank 4.
    # Target has LOW dense (sub-threshold) but HIGH lexical (above threshold).
    dense_raw = dict.fromkeys(ids, 0.0)
    lexical_raw = dict.fromkeys(ids, 0.0)
    # Four clear winners with high dense+lexical.
    for winner in ("c0", "c1", "c2", "c3"):
        dense_raw[winner] = 1.0
        lexical_raw[winner] = 1.0
    # Target: dense_n will be low, lexical_n high.
    dense_raw[target] = 0.05
    lexical_raw[target] = 0.9
    # Remaining chunks below target.
    for low in ("c4", "c5", "c6", "c7", "c8"):
        dense_raw[low] = 0.0
        lexical_raw[low] = 0.1

    retriever = _ScriptedHybrid(corpus, dense_raw, lexical_raw, alpha=0.5)

    lk = large_k(top_k, corpus.__len__())
    # Sanity: target is within large_k and at rank >= top_k.
    ranked = [s.chunk.id for s in retriever.search("q", lk)]
    target_rank = ranked.index(target)
    assert target_rank >= top_k
    assert target_rank < lk

    result = diagnose(
        retriever,
        "q",
        target,
        Config(top_k=top_k, alpha=0.5),
        dense_threshold=DEFAULT_DENSE_THRESHOLD,
        lexical_threshold=DEFAULT_LEXICAL_THRESHOLD,
    )
    # Despite sub-threshold dense + high lexical (which is the blind-spot
    # *evidence* pattern), membership in large_k forces out_ranked.
    assert result.failure_class is FailureClass.OUT_RANKED
    assert result.evidence["rank_at_large_k"] == target_rank


def test_embedding_blind_spot_outside_large_k() -> None:
    """A present chunk ranked below large_k => embedding_blind_spot (terminal)."""
    top_k = 1
    # large_k = min(10*1, corpus_size). Make corpus large and push target last.
    ids = [f"c{i}" for i in range(30)]
    chunks = [Chunk(id=i, text=f"text {i}") for i in ids]
    corpus = Corpus.from_chunks(chunks)
    target = "c29"

    dense_raw = dict.fromkeys(ids, 1.0)
    lexical_raw = dict.fromkeys(ids, 1.0)
    # Target is the single worst on both modalities, far below large_k=10.
    dense_raw[target] = -1.0
    lexical_raw[target] = 0.0
    # Make the first 10 strictly the best so target sits at the very bottom.
    retriever = _ScriptedHybrid(corpus, dense_raw, lexical_raw, alpha=0.5)

    lk = large_k(top_k, len(corpus))
    ranked = [s.chunk.id for s in retriever.search("q", lk)]
    assert target not in ranked  # outside large_k

    result = diagnose(retriever, "q", target, Config(top_k=top_k, alpha=0.5))
    assert result.failure_class is FailureClass.EMBEDDING_BLIND_SPOT
    assert result.evidence["rank_at_large_k"] is None


def test_blind_spot_evidence_string_dense_misses_lexical() -> None:
    """When dense<threshold and lexical>threshold, evidence names the dense miss."""
    top_k = 1
    ids = [f"c{i}" for i in range(30)]
    chunks = [Chunk(id=i, text=f"text {i}") for i in ids]
    corpus = Corpus.from_chunks(chunks)
    target = "c29"

    dense_raw = dict.fromkeys(ids, 1.0)
    lexical_raw = dict.fromkeys(ids, 0.0)
    dense_raw[target] = 0.0  # normalized low
    lexical_raw[target] = 100.0  # normalized high (it's the lexical max)
    # But many others must outrank on combined to keep it outside large_k:
    for other in ids:
        if other != target:
            lexical_raw[other] = 0.0
            dense_raw[other] = 1.0
    retriever = _ScriptedHybrid(corpus, dense_raw, lexical_raw, alpha=0.9)

    result = diagnose(retriever, "q", target, Config(top_k=top_k, alpha=0.9))
    assert result.failure_class is FailureClass.EMBEDDING_BLIND_SPOT
    assert "dense model misses" in str(result.evidence["note"])


def test_lost_to_chunking_with_provenance() -> None:
    # A phrase that, at a coarse chunk_size, is split across chunks but at a
    # finer size lands intact and retrievable.
    text = (
        "Intro filler sentence number one to pad the document length. "
        "The secret passphrase is hummingbird velocity matrix. "
        "More trailing filler to extend well beyond a small window size here."
    )
    docs = [SourceDocument(id="doc", text=text)]
    chunker = FixedSizeChunker(overlap=0)
    # Build coarse so the passphrase is whole in one big chunk but competes
    # against filler; finer chunking isolates it.
    corpus = Corpus.from_sources(docs, chunker, chunk_size=1024)
    embedder = FakeEmbedder(seed=5)
    dense = DenseRetriever(corpus, embedder, chunker=chunker, config=Config(chunk_size=1024))
    lexical = BM25Retriever(corpus, chunker=chunker, config=Config(chunk_size=1024))
    hybrid = HybridRetriever(dense, lexical, alpha=0.5)

    # The expected chunk is the whole-doc chunk (since chunk_size=1024 covers it).
    expected_id = corpus.chunks[0].id
    result = diagnose(
        hybrid,
        "secret passphrase hummingbird velocity matrix",
        expected_id,
        Config(top_k=1, chunk_size=1024, alpha=0.5),
    )
    # lost_to_chunking must NOT be in unevaluable (provenance is present).
    assert FailureClass.LOST_TO_CHUNKING not in result.unevaluable


def test_no_provenance_marks_lost_to_chunking_unevaluable(
    hybrid: HybridRetriever,
) -> None:
    """Without provenance, lost_to_chunking is unevaluable but the result is
    still labeled by branches 3/4.
    """
    # 'eiffel' is present and retrievable; with a tiny top_k it may be out_ranked.
    result = diagnose(hybrid, "iron landmark Paris", "seine", Config(top_k=1, alpha=0.5))
    assert FailureClass.LOST_TO_CHUNKING in result.unevaluable
    assert result.failure_class is not None  # still labeled


def test_decision_order_missing_precedes_chunking(
    source_docs: list[SourceDocument], chunker: FixedSizeChunker
) -> None:
    # Even with provenance, a truly-absent id is missing_from_index (branch 1),
    # never lost_to_chunking.
    corpus = Corpus.from_sources(source_docs, chunker, chunk_size=128)
    embedder = FakeEmbedder(seed=2)
    dense = DenseRetriever(corpus, embedder, chunker=chunker, config=Config(chunk_size=128))
    lexical = BM25Retriever(corpus, chunker=chunker, config=Config(chunk_size=128))
    hybrid = HybridRetriever(dense, lexical, alpha=0.5)
    result = diagnose(hybrid, "photosynthesis oxygen", "absent-id", Config(top_k=2, alpha=0.5))
    assert result.failure_class is FailureClass.MISSING_FROM_INDEX


def test_evidence_records_thresholds(hybrid: HybridRetriever) -> None:
    result = diagnose(hybrid, "Paris", "seine", Config(top_k=1, alpha=0.5))
    assert result.evidence["dense_threshold"] == DEFAULT_DENSE_THRESHOLD
    assert result.evidence["lexical_threshold"] == DEFAULT_LEXICAL_THRESHOLD
    assert "large_k" in result.evidence


def test_diagnose_standalone_dense_populates_dense_score(
    tiny_corpus: Corpus, embedder: FakeEmbedder
) -> None:
    retriever = DenseRetriever(tiny_corpus, embedder)
    result = diagnose(retriever, "river Seine France", "banana", Config(top_k=1))
    # Dense modality is detected; dense_score present, lexical_score absent.
    assert result.evidence["dense_score"] is not None
    assert result.evidence["lexical_score"] is None


def test_diagnose_standalone_bm25_populates_lexical_score(
    tiny_corpus: Corpus,
) -> None:
    retriever = BM25Retriever(tiny_corpus)
    result = diagnose(retriever, "Paris France capital", "banana", Config(top_k=1))
    # Lexical modality detected via the component probe.
    assert result.evidence["lexical_score"] is not None
    assert result.evidence["dense_score"] is None


def test_diagnose_chunk_already_within_top_k(hybrid: HybridRetriever) -> None:
    # 'paris' is the top result for this query; diagnosing it is NOT a failure
    # (rank < top_k), so failure_class is None — out_ranked is a failure cause
    # and must never label a chunk that is currently retrieved within top_k.
    top = hybrid.search("capital of France Paris", 1)[0]
    result = diagnose(hybrid, "capital of France Paris", top.chunk.id, Config(top_k=3, alpha=0.5))
    assert result.failure_class is None
    assert "not currently failing" in str(result.evidence["note"])
    assert "retrieved within top_k" in str(result.evidence["note"])


def test_diagnose_retriever_without_corpus_marks_missing_unevaluable() -> None:
    # A minimal retriever exposing no corpus: membership cannot be tested.
    from why_this_chunk.types import ScoredChunk as _SC

    class NoCorpus:
        @property
        def corpus_size(self) -> int:
            return 0

        @property
        def supports_components(self) -> bool:
            return False

        @property
        def supports_reindex(self) -> bool:
            return False

        def search(self, query: str, k: int) -> list[_SC]:
            return []

        def reindex(self, config: RetrievalConfig):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    result = diagnose(NoCorpus(), "q", "anything", Config(top_k=1))
    assert FailureClass.MISSING_FROM_INDEX in result.unevaluable
    assert FailureClass.LOST_TO_CHUNKING in result.unevaluable


def test_within_top_k_chunk_is_not_a_failure(hybrid: HybridRetriever) -> None:
    """Regression: a chunk at rank 0 with top_k=3 (rank < top_k) is not failing,
    so ``failure_class`` is None — ``out_ranked`` (a failure cause) must never be
    attached to a chunk already retrieved within top_k.
    """
    top = hybrid.search("capital of France Paris", 1)[0]
    assert top.rank == 0  # the expected chunk is the rank-0 result
    result = diagnose(hybrid, "capital of France Paris", top.chunk.id, Config(top_k=3, alpha=0.5))
    assert result.failure_class is None
    assert result.failure_class is not FailureClass.OUT_RANKED
    assert result.evidence["rank_at_large_k"] == 0
    assert "not currently failing" in str(result.evidence["note"])


class _NoCorpusEmpty:
    """A retriever exposing no ``corpus`` and returning no results.

    Membership is unevaluable (no corpus) and the expected chunk is absent from
    ``large_k`` (empty search), so presence cannot be established.
    """

    @property
    def corpus_size(self) -> int:
        return 0

    @property
    def supports_components(self) -> bool:
        return False

    @property
    def supports_reindex(self) -> bool:
        return False

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        return []

    def reindex(self, config: RetrievalConfig) -> _NoCorpusEmpty:  # pragma: no cover
        raise NotImplementedError


def test_indeterminate_when_membership_unevaluable_and_absent_from_large_k() -> None:
    """Regression: when membership is unevaluable (no corpus) AND the chunk is
    absent from ``large_k`` (rank None), presence cannot be established, so the
    terminal ``embedding_blind_spot`` verdict is unsound — ``failure_class`` must
    be None (indeterminate), keeping ``missing_from_index`` in ``unevaluable``.
    """
    result = diagnose(_NoCorpusEmpty(), "q", "anything", Config(top_k=2))
    assert result.evidence["rank_at_large_k"] is None
    assert FailureClass.MISSING_FROM_INDEX in result.unevaluable
    assert result.failure_class is None
    assert result.failure_class is not FailureClass.EMBEDDING_BLIND_SPOT
    assert "indeterminate" in str(result.evidence["note"])
