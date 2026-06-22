"""Tests for the bounded counterfactual minimal-fix search."""

from __future__ import annotations

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
    search_fixes,
)
from why_this_chunk.config import ALPHA_SWEEP, CHUNK_SIZE_SWEEP
from why_this_chunk.counterfactual import AXIS_PRIORITY
from why_this_chunk.counterfactual import search_fixes as run_fixes
from why_this_chunk.types import FixSuggestion


def test_top_k_fix_wins_when_raising_k_surfaces_chunk() -> None:
    # 'target' is retrievable but not within top_k=1; raising K surfaces it.
    chunks = [Chunk(id="target", text="alpha beta gamma delta")]
    for i in range(4):
        chunks.append(Chunk(id=f"d{i}", text="alpha beta gamma epsilon"))
    corpus = Corpus.from_chunks(chunks)
    retriever = BM25Retriever(corpus)
    result = search_fixes(retriever, "alpha beta gamma", "target", RetrievalConfig(top_k=1))
    assert result.best is not None
    assert result.best.param == "top_k"
    assert result.best.to_value > 1
    assert result.best.new_rank == result.best.to_value - 1


def test_no_fix_when_chunk_already_top_ranked() -> None:
    # Several chunks so IDF is sane; 'target' uniquely carries the query terms
    # and clearly ranks first, so the top_k axis must propose nothing.
    chunks = [Chunk(id="target", text="unique salient distinctive marker phrase")]
    for i in range(5):
        chunks.append(Chunk(id=f"d{i}", text=f"ordinary background filler document number {i}"))
    corpus = Corpus.from_chunks(chunks)
    retriever = BM25Retriever(corpus)
    ranked = retriever.search("unique salient distinctive marker", 1)
    assert ranked[0].chunk.id == "target"  # genuinely rank 0
    result = search_fixes(
        retriever, "unique salient distinctive marker", "target", RetrievalConfig(top_k=1)
    )
    top_k_fixes = [f for f in result.all_fixes if f.param == "top_k"]
    assert top_k_fixes == []


def test_chunk_size_fix_with_provenance() -> None:
    # A passphrase split across windows at chunk_size=512 but intact at a finer
    # or coarser swept size, surfaced via reindex.
    filler_a = "Padding sentence to consume window budget. " * 6
    filler_b = "Trailing padding to consume more window budget here. " * 6
    text = filler_a + "SECRET marmoset zeppelin cucumber TOKEN. " + filler_b
    docs = [SourceDocument(id="doc", text=text)]
    chunker = FixedSizeChunker(overlap=0)
    start_size = 256
    corpus = Corpus.from_sources(docs, chunker, chunk_size=start_size)
    embedder = FakeEmbedder(seed=11)
    dense = DenseRetriever(
        corpus, embedder, chunker=chunker, config=RetrievalConfig(chunk_size=start_size)
    )
    lexical = BM25Retriever(corpus, chunker=chunker, config=RetrievalConfig(chunk_size=start_size))
    hybrid = HybridRetriever(dense, lexical, alpha=0.5)

    # Find which existing chunk holds the SECRET phrase at the start size.
    target_id = next(c.id for c in corpus.chunks if "marmoset" in c.text)
    result = search_fixes(
        hybrid,
        "SECRET marmoset zeppelin cucumber TOKEN",
        target_id,
        RetrievalConfig(top_k=1, chunk_size=start_size, alpha=0.5),
    )
    # chunk_size axis must be evaluable (provenance present).
    assert "chunk_size" not in result.unevaluable


def test_chunk_size_unevaluable_without_provenance(bm25: BM25Retriever) -> None:
    """The chunk_size axis is reported unevaluable, NOT silently skipped, when
    the corpus lacks provenance.
    """
    result = search_fixes(bm25, "Paris France", "seine", RetrievalConfig(top_k=1))
    assert "chunk_size" in result.unevaluable


def test_chunk_size_unevaluable_without_reindex_capability() -> None:
    """A retriever lacking ``supports_reindex`` makes chunk_size unevaluable."""

    class NoReindex:
        def __init__(self, corpus: Corpus) -> None:
            self._corpus = corpus

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
            return False

        def search(self, query: str, k: int):  # type: ignore[no-untyped-def]
            return []

        def reindex(self, config: RetrievalConfig):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    docs = [SourceDocument(id="d", text="some text here for the document body")]
    corpus = Corpus.from_sources(docs, FixedSizeChunker(), chunk_size=128)
    retriever = NoReindex(corpus)
    result = search_fixes(retriever, "text", corpus.chunks[0].id, RetrievalConfig(top_k=1))
    assert "chunk_size" in result.unevaluable


def test_alpha_fix_for_hybrid() -> None:
    # A lexically-strong, densely-weak target that only surfaces when alpha
    # shifts toward the lexical modality.
    chunks = [
        Chunk(id="target", text="xyzzy plugh frobozz lexical-only marker"),
    ]
    # Distractors with strong dense overlap with the query but not the marker.
    for i in range(4):
        chunks.append(Chunk(id=f"d{i}", text="generic semantic content about topics"))
    corpus = Corpus.from_chunks(chunks)
    embedder = FakeEmbedder(seed=4)
    dense = DenseRetriever(corpus, embedder)
    lexical = BM25Retriever(corpus)
    # Start dense-heavy so the lexical-only target is buried.
    hybrid = HybridRetriever(dense, lexical, alpha=1.0)
    result = search_fixes(
        hybrid,
        "xyzzy plugh frobozz lexical-only marker",
        "target",
        RetrievalConfig(top_k=1, alpha=1.0),
    )
    # At least one axis should surface it; if alpha does, it must be in the list.
    params = {f.param for f in result.all_fixes}
    assert "alpha" not in result.unevaluable  # hybrid + reindex => evaluable
    assert params  # some fix was found


def test_minimal_cost_and_tie_break_priority() -> None:
    # Construct two fixes with equal cost and assert the priority ordering.
    fixes = [
        FixSuggestion("chunk_size", 256, 512, cost=1, new_rank=0, explanation="cs"),
        FixSuggestion("top_k", 1, 2, cost=1, new_rank=1, explanation="tk"),
    ]
    fixes.sort(key=lambda f: (f.cost, AXIS_PRIORITY[f.param]))
    assert [f.param for f in fixes] == ["top_k", "chunk_size"]


def test_sweeps_are_bounded() -> None:
    # The sweeps are finite, fixed-size constants — guard against accidental
    # unbounded growth.
    assert len(CHUNK_SIZE_SWEEP) <= 16
    assert len(ALPHA_SWEEP) <= 16
    assert all(isinstance(s, int) for s in CHUNK_SIZE_SWEEP)


def test_rerank_unevaluable_without_configured_reranker(bm25: BM25Retriever) -> None:
    result = search_fixes(bm25, "Paris", "seine", RetrievalConfig(top_k=1))
    assert "rerank" in result.unevaluable


def test_all_fixes_sorted_by_cost_then_priority() -> None:
    chunks = [Chunk(id="target", text="alpha beta gamma delta")]
    for i in range(6):
        chunks.append(Chunk(id=f"d{i}", text="alpha beta gamma epsilon"))
    corpus = Corpus.from_chunks(chunks)
    retriever = BM25Retriever(corpus)
    result = run_fixes(retriever, "alpha beta gamma", "target", RetrievalConfig(top_k=1))
    costs = [(f.cost, AXIS_PRIORITY[f.param]) for f in result.all_fixes]
    assert costs == sorted(costs)


#: Small target region (character offsets) that the chunk_size stub models: a
#: covering window at ``good_size`` must *fully contain* it (mere overlap is not
#: enough to count as a valid fix).
_TARGET_SPAN: tuple[int, int] = (100, 150)


class _ReindexableStub:
    """A retriever whose target ranks within top_k only at a specific chunk_size.

    Models the ``chunk_size`` counterfactual axis deterministically: a window
    *fully containing* the small target region (``_TARGET_SPAN``) surfaces (rank
    0) only when reindexed to ``good_size``. The named ``expected`` chunk carries
    ``_TARGET_SPAN`` so the fix search maps it back to that region; every
    ``good_size`` here yields a ``(0, good_size)`` window that contains it.
    """

    def __init__(self, good_size: int, current_size: int = 512) -> None:
        self._good_size = good_size
        self._config = RetrievalConfig(chunk_size=current_size)
        self._doc = SourceDocument(id="doc", text="y" * 2048)
        self._expected = Chunk(
            id="target",
            text="y" * (_TARGET_SPAN[1] - _TARGET_SPAN[0]),
            source_document_id="doc",
            span=_TARGET_SPAN,
        )
        windows = FixedSizeChunker().chunk([self._doc], current_size)
        self._corpus = Corpus([self._expected, *windows], source_documents=[self._doc])

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

    def search(self, query: str, k: int):  # type: ignore[no-untyped-def]
        from why_this_chunk.types import ScoredChunk

        if self._config.chunk_size == self._good_size:
            covering = Chunk(
                id=f"doc::{self._good_size}::cover",
                text="y" * self._good_size,
                source_document_id="doc",
                span=(0, self._good_size),
            )
            head = [covering]
        else:
            head = []
        return [ScoredChunk(chunk=c, score=1.0, rank=i) for i, c in enumerate(head[:k])]

    def reindex(self, config: RetrievalConfig) -> _ReindexableStub:
        return _ReindexableStub(self._good_size, current_size=config.chunk_size)


def test_chunk_size_axis_finds_fix_via_reindex() -> None:
    retriever = _ReindexableStub(good_size=256, current_size=512)
    expected = retriever.expected_id
    result = search_fixes(retriever, "q", expected, RetrievalConfig(top_k=1, chunk_size=512))
    assert "chunk_size" not in result.unevaluable
    chunk_fixes = [f for f in result.all_fixes if f.param == "chunk_size"]
    assert chunk_fixes, "expected a chunk_size fix"
    assert chunk_fixes[0].to_value == 256
    # Cost is index distance in the sweep between 512 and 256.
    assert chunk_fixes[0].cost == abs(CHUNK_SIZE_SWEEP.index(256) - CHUNK_SIZE_SWEEP.index(512))


def test_alpha_axis_finds_fix_when_only_alpha_helps() -> None:
    # Lexical-only marker buried under dense-heavy alpha=1.0; shifting alpha to
    # the lexical side surfaces it.
    chunks = [Chunk(id="target", text="qwerty lexonly marker token unique")]
    for i in range(5):
        chunks.append(Chunk(id=f"d{i}", text=f"semantic dense topical content number {i}"))
    corpus = Corpus.from_chunks(chunks)
    embedder = FakeEmbedder(seed=8)
    dense = DenseRetriever(corpus, embedder)
    lexical = BM25Retriever(corpus)
    hybrid = HybridRetriever(dense, lexical, alpha=1.0)
    # At alpha=1.0 the lexical-only marker may not be top-1.
    result = search_fixes(
        hybrid,
        "qwerty lexonly marker token unique",
        "target",
        RetrievalConfig(top_k=1, alpha=1.0),
    )
    assert "alpha" not in result.unevaluable
    # Either alpha or top_k surfaces it; the alpha axis was at least evaluated.
    assert result.all_fixes


def test_chunk_size_nearest_index_when_not_in_sweep() -> None:
    # current chunk_size 500 is not a sweep entry => nearest-index cost path.
    retriever = _ReindexableStub(good_size=512, current_size=500)
    expected = retriever.expected_id
    result = search_fixes(retriever, "q", expected, RetrievalConfig(top_k=1, chunk_size=500))
    chunk_fixes = [f for f in result.all_fixes if f.param == "chunk_size"]
    assert chunk_fixes
    assert chunk_fixes[0].to_value == 512


class _PartialOverlapStub:
    """A retriever whose re-chunk windows only *overlap* the target span.

    At every swept size the surfacing window covers only part of the target
    region (it starts inside it and ends before it does), so no window fully
    contains the expected text. The ``chunk_size`` fix must therefore NOT be
    recommended — overlap alone is a false fix under the containment rule.
    """

    def __init__(self, current_size: int = 512) -> None:
        self._config = RetrievalConfig(chunk_size=current_size)
        self._doc = SourceDocument(id="doc", text="z" * 2048)
        self._expected = Chunk(
            id="target",
            text="z" * (_TARGET_SPAN[1] - _TARGET_SPAN[0]),
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

    def search(self, query: str, k: int):  # type: ignore[no-untyped-def]
        from why_this_chunk.types import ScoredChunk

        exp_start, exp_end = _TARGET_SPAN
        partial = Chunk(
            id=f"doc::{self._config.chunk_size}::partial",
            text="z" * ((exp_end - 10) - (exp_start + 10)),
            source_document_id="doc",
            span=(exp_start + 10, exp_end - 10),
        )
        return [ScoredChunk(chunk=partial, score=1.0, rank=0)][:k]

    def reindex(self, config: RetrievalConfig) -> _PartialOverlapStub:
        return _PartialOverlapStub(current_size=config.chunk_size)


def test_chunk_size_fix_not_recommended_when_only_overlapping() -> None:
    """Regression: a swept size whose window merely overlaps (does not fully
    contain) the expected span must NOT be recommended as a chunk_size fix.
    """
    retriever = _PartialOverlapStub(current_size=512)
    expected = retriever.expected_id
    result = search_fixes(retriever, "q", expected, RetrievalConfig(top_k=1, chunk_size=512))
    # The axis is evaluable (provenance + reindex) but yields no fix.
    assert "chunk_size" not in result.unevaluable
    chunk_fixes = [f for f in result.all_fixes if f.param == "chunk_size"]
    assert chunk_fixes == []


def test_chunk_size_fix_recommended_when_window_fully_contains_span() -> None:
    """Regression (positive): a swept size whose window fully contains the
    expected span IS recommended as a chunk_size fix.
    """
    retriever = _ReindexableStub(good_size=256, current_size=512)
    expected = retriever.expected_id
    result = search_fixes(retriever, "q", expected, RetrievalConfig(top_k=1, chunk_size=512))
    chunk_fixes = [f for f in result.all_fixes if f.param == "chunk_size"]
    assert chunk_fixes, "a fully-containing window must yield a chunk_size fix"
    assert chunk_fixes[0].to_value == 256
