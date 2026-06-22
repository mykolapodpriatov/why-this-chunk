"""Tests for occlusion attribution, including the degenerate uniform fallback."""

from __future__ import annotations

import pytest

from why_this_chunk import (
    BM25Retriever,
    Chunk,
    Corpus,
    DenseRetriever,
    FakeEmbedder,
    HybridRetriever,
    attribute,
    explain_chunk,
)


def _single_corpus_dense(text: str, embedder: FakeEmbedder) -> DenseRetriever:
    corpus = Corpus.from_chunks([Chunk(id="target", text=text)])
    return DenseRetriever(corpus, embedder)


def test_query_carrying_sentence_gets_highest_share(
    embedder: FakeEmbedder,
) -> None:
    # A realistic multi-chunk corpus so IDF is well-behaved (a single-document
    # corpus triggers BM25's negative-IDF degeneracy, a corpus problem rather
    # than an attribution one).
    chunks = [
        Chunk(
            id="target",
            text=(
                "Paris is the capital city of France. "
                "Quarterly revenue grew on strong market demand."
            ),
        ),
        Chunk(id="c2", text="Berlin is the capital of Germany and a major hub."),
        Chunk(id="c3", text="Revenue and profit figures were reported last quarter."),
        Chunk(id="c4", text="The weather in spring is often mild and pleasant."),
        Chunk(id="c5", text="Tokyo is the largest metropolitan area in the world."),
    ]
    corpus = Corpus.from_chunks(chunks)
    dense = DenseRetriever(corpus, embedder)
    lexical = BM25Retriever(corpus)
    retriever = HybridRetriever(dense, lexical, alpha=0.5)
    result = retriever.search("capital of France Paris", 1)[0]
    assert result.chunk.id == "target"
    explanation = explain_chunk(retriever, "capital of France Paris", result)

    assert not explanation.degenerate
    top = explanation.sentences[0]
    assert "Paris" in top.sentence
    assert top.share == max(s.share for s in explanation.sentences)
    assert abs(sum(s.share for s in explanation.sentences) - 1.0) < 1e-9


def test_empty_chunk_yields_no_attributions(embedder: FakeEmbedder) -> None:
    retriever = _single_corpus_dense("", embedder)
    result = retriever.search("anything", 1)
    # An all-zero embedding still returns a (zero-score) result for k>=1.
    explanation = explain_chunk(retriever, "anything", result[0])
    assert explanation.sentences == []
    assert explanation.degenerate is False


def test_single_sentence_chunk(embedder: FakeEmbedder) -> None:
    retriever = _single_corpus_dense("Just one sentence here.", embedder)
    result = retriever.search("one sentence", 1)[0]
    explanation = explain_chunk(retriever, "one sentence", result)
    assert len(explanation.sentences) == 1
    # A lone positive-delta sentence carries the whole share.
    assert explanation.sentences[0].share == pytest.approx(1.0)


def test_token_granularity(embedder: FakeEmbedder) -> None:
    text = "Paris France capital revenue demand growth."
    retriever = _single_corpus_dense(text, embedder)
    result = retriever.search("Paris capital", 1)[0]
    explanation = explain_chunk(retriever, "Paris capital", result, granularity="token")
    assert explanation.granularity == "token"
    assert len(explanation.sentences) >= 4
    surfaces = {s.sentence for s in explanation.sentences}
    assert "Paris" in surfaces


def test_degenerate_all_non_positive_delta_uniform_share() -> None:
    # A scorer where removing ANY unit raises the score => every delta < 0.
    # Forces the documented uniform 1/n fallback, no ZeroDivisionError.
    text = "Alpha unit. Beta unit. Gamma unit."

    def adversarial_scorer(candidate: str) -> float:
        # Shorter candidate scores higher, so occlusion (which shortens) always
        # increases the score => full - occluded < 0 for every unit.
        return -float(len(candidate))

    attributions, degenerate = attribute(text, adversarial_scorer, "sentence")
    assert degenerate is True
    n = len(attributions)
    assert n == 3
    for attribution in attributions:
        assert attribution.share == pytest.approx(1.0 / n)
        assert attribution.delta < 0.0
    assert sum(a.share for a in attributions) == pytest.approx(1.0)


def test_explanation_flags_degenerate_from_real_retriever() -> None:
    # A stop-word-heavy chunk against an unrelated query: deltas tend to be
    # non-positive, exercising the degenerate path end-to-end.
    embedder = FakeEmbedder(seed=3)
    text = "the the the. and and and. of of of."
    corpus = Corpus.from_chunks([Chunk(id="t", text=text)])
    retriever = DenseRetriever(corpus, embedder)
    result = retriever.search("quantum chromodynamics", 1)[0]
    explanation = explain_chunk(retriever, "quantum chromodynamics", result)
    # Either it found a positive delta or it fell back to uniform; if uniform,
    # shares must be exactly 1/n and the flag set.
    if explanation.degenerate:
        n = len(explanation.sentences)
        for s in explanation.sentences:
            assert s.share == pytest.approx(1.0 / n)


def test_hybrid_attribution_uses_combined_score(dense: DenseRetriever, bm25: BM25Retriever) -> None:
    hybrid = HybridRetriever(dense, bm25, alpha=0.5)
    result = hybrid.search("capital of France Paris", 1)[0]
    explanation = explain_chunk(hybrid, "capital of France Paris", result)
    assert explanation.split is not None
    assert explanation.sentences
    # Shares are normalized.
    assert abs(sum(s.share for s in explanation.sentences) - 1.0) < 1e-9


def test_lexical_attribution(bm25: BM25Retriever) -> None:
    result = bm25.search("Paris France capital", 1)[0]
    explanation = explain_chunk(bm25, "Paris France capital", result)
    assert explanation.split is None  # non-hybrid: no split
    assert explanation.sentences


def test_invalid_granularity_raises(embedder: FakeEmbedder) -> None:
    retriever = _single_corpus_dense("A sentence.", embedder)
    result = retriever.search("sentence", 1)[0]
    with pytest.raises(ValueError, match="granularity"):
        explain_chunk(retriever, "sentence", result, granularity="phrase")


def test_hybrid_scorer_clamps_combined_score_into_unit_interval(
    embedder: FakeEmbedder,
) -> None:
    """Regression: the hybrid combined scorer must clamp ad-hoc text scores to
    [0, 1] so a text scoring outside the pool's [lo, hi] bounds cannot push a
    delta outside the valid range. An ad-hoc text repeating every query term
    many times scores above the pool's lexical max; without clamping its
    normalized combined score would exceed 1.0.
    """
    from why_this_chunk.attribution import _hybrid_scorer

    chunks = [
        Chunk(id="target", text="alpha beta gamma in a sentence about alpha topics."),
        Chunk(id="c2", text="beta and gamma appear here with some other words."),
        Chunk(id="c3", text="unrelated content concerning delta and epsilon only."),
        Chunk(id="c4", text="more padding so the lexical pool has spread across docs."),
    ]
    corpus = Corpus.from_chunks(chunks)
    dense = DenseRetriever(corpus, embedder)
    lexical = BM25Retriever(corpus)
    hybrid = HybridRetriever(dense, lexical, alpha=0.5)
    scorer = _hybrid_scorer(hybrid, "alpha beta gamma")

    # A text that saturates every query term far beyond any pooled chunk.
    saturated = "alpha beta gamma " * 25
    assert 0.0 <= scorer(saturated) <= 1.0
    # And for the indexed texts too.
    for chunk in chunks:
        assert 0.0 <= scorer(chunk.text) <= 1.0


def test_hybrid_attribution_shares_stay_in_unit_interval(
    embedder: FakeEmbedder,
) -> None:
    """Regression: occlusion shares from the hybrid combined score remain within
    [0, 1] (and sum to 1) even when occluded variants score beyond the pool's
    normalization bounds.
    """
    chunks = [
        Chunk(id="target", text="alpha beta gamma. alpha beta gamma. alpha beta gamma."),
        Chunk(id="c2", text="beta gamma in a different context entirely here."),
        Chunk(id="c3", text="delta epsilon zeta unrelated padding sentence content."),
    ]
    corpus = Corpus.from_chunks(chunks)
    dense = DenseRetriever(corpus, embedder)
    lexical = BM25Retriever(corpus)
    hybrid = HybridRetriever(dense, lexical, alpha=0.5)
    result = hybrid.search("alpha beta gamma", 1)[0]
    explanation = explain_chunk(hybrid, "alpha beta gamma", result)

    assert explanation.sentences
    for sentence in explanation.sentences:
        assert 0.0 <= sentence.share <= 1.0
    assert abs(sum(s.share for s in explanation.sentences) - 1.0) < 1e-9


def test_attribute_requires_score_text() -> None:
    class NoScorer:
        pass

    # _make_scorer is internal; explain_chunk surfaces the TypeError.
    from why_this_chunk.attribution import _make_scorer

    with pytest.raises(TypeError, match="score_text"):
        _make_scorer(NoScorer(), "q")
