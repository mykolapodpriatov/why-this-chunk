"""Occlusion-based sentence/token attribution of a chunk's retrieval score.

The method is model-agnostic and deterministic: split a chunk's text into units
(sentences by default, tokens with ``granularity="token"``); for each unit,
recompute the chunk's score with that unit removed; ``delta = score_full -
score_occluded``. Positive deltas are normalized into shares.

**Degenerate fallback (no ``ZeroDivisionError``):** when no unit has a positive
delta (``sum(max(delta, 0)) == 0`` — common for short or stop-word-heavy chunks
where removing any unit *raises* the score), every unit receives the uniform
share ``1 / n`` and the resulting :class:`~why_this_chunk.types.Explanation` is
flagged ``degenerate=True`` so the report can say "no single unit dominates".

Scoring an occluded variant reuses the modality's ``score_text`` (dense and
lexical) without rebuilding any index. For hybrid retrievers, the occluded
chunk's combined score is recomputed against the pool's existing normalization
bounds, so attribution is on the **combined** score exactly as documented.
"""

from __future__ import annotations

from collections.abc import Callable

from why_this_chunk._scoring import DEGENERATE_NORM_VALUE
from why_this_chunk._text import split_sentences, token_spans
from why_this_chunk.contributions import compute_split
from why_this_chunk.retrievers.bm25 import BM25Retriever
from why_this_chunk.retrievers.dense import DenseRetriever
from why_this_chunk.retrievers.hybrid import HybridRetriever
from why_this_chunk.types import (
    ContributionSplit,
    Explanation,
    ScoredChunk,
    SentenceAttribution,
)

__all__ = ["attribute", "explain_chunk"]

# A scorer maps an arbitrary candidate text to a score on the same scale the
# retriever ranks on (so deltas are comparable to the result's score).
_Scorer = Callable[[str], float]


def _make_scorer(retriever: object, query: str) -> _Scorer:
    """Build a text scorer matching the retriever's ranking scale.

    For dense/lexical, this is the raw modality score of the text. For hybrid,
    the text's raw dense and lexical scores are folded into the pool's existing
    min-max bounds (computed once over the full corpus for ``query``) and blended
    by ``alpha`` — i.e. attribution is on the combined score.

    Raises:
        TypeError: If the retriever exposes no usable scoring path.
    """
    if isinstance(retriever, HybridRetriever):
        return _hybrid_scorer(retriever, query)
    if isinstance(retriever, DenseRetriever | BM25Retriever):
        return lambda text: retriever.score_text(query, text)
    score_text = getattr(retriever, "score_text", None)
    if callable(score_text):
        return lambda text: float(score_text(query, text))
    raise TypeError(
        "attribution requires a retriever exposing score_text(query, text); "
        f"{type(retriever).__name__} does not"
    )


def _hybrid_scorer(retriever: HybridRetriever, query: str) -> _Scorer:
    """Combined-score scorer that re-normalizes ad-hoc text into pool bounds."""
    dense = retriever._dense
    lexical = retriever._lexical
    alpha = retriever.alpha
    dense_pool = dense.raw_scores(query)
    lexical_pool = lexical.raw_scores(query)
    dense_lo, dense_hi = (
        (float(dense_pool.min()), float(dense_pool.max())) if dense_pool.size else (0.0, 0.0)
    )
    lexical_lo, lexical_hi = (
        (float(lexical_pool.min()), float(lexical_pool.max())) if lexical_pool.size else (0.0, 0.0)
    )

    def _norm(value: float, lo: float, hi: float) -> float:
        if hi == lo:
            return DEGENERATE_NORM_VALUE
        return (value - lo) / (hi - lo)

    def score(text: str) -> float:
        d_raw = dense.score_text(query, text)
        l_raw = lexical.score_text(query, text)
        d_n = _norm(d_raw, dense_lo, dense_hi)
        l_n = _norm(l_raw, lexical_lo, lexical_hi)
        combined = alpha * d_n + (1.0 - alpha) * l_n
        # An occluded variant can score outside the pool's [lo, hi] bounds (e.g.
        # an ad-hoc text more/less similar than any indexed chunk), pushing the
        # normalized combined score outside [0, 1]. Clamp it so deltas — and thus
        # the attribution shares derived from them — stay within the valid range.
        return min(1.0, max(0.0, combined))

    return score


def _units(text: str, granularity: str) -> list[tuple[str, tuple[int, int]]]:
    """Return the attribution units for ``text`` at the given granularity."""
    if granularity == "sentence":
        return split_sentences(text)
    if granularity == "token":
        return token_spans(text)
    raise ValueError(f"granularity must be 'sentence' or 'token', got {granularity!r}")


def attribute(
    text: str,
    scorer: _Scorer,
    granularity: str = "sentence",
) -> tuple[list[SentenceAttribution], bool]:
    """Attribute ``text``'s score to its units via occlusion.

    Args:
        text: The chunk text to attribute.
        scorer: A function mapping candidate text to a score; called once on the
            full text and once per occluded variant.
        granularity: ``"sentence"`` or ``"token"``.

    Returns:
        A ``(attributions, degenerate)`` pair. ``attributions`` is ordered by
        descending ``share`` then ascending span. ``degenerate`` is ``True`` when
        the uniform ``1 / n`` fallback was used (no positive delta).
    """
    units = _units(text, granularity)
    if not units:
        return [], False

    full_score = scorer(text)
    deltas: list[float] = []
    for _, (start, end) in units:
        occluded = text[:start] + text[end:]
        deltas.append(full_score - scorer(occluded))

    positive_mass = sum(d for d in deltas if d > 0.0)
    degenerate = positive_mass <= 0.0
    n = len(units)

    attributions: list[SentenceAttribution] = []
    for (unit_text, span), delta in zip(units, deltas, strict=True):
        # Degenerate (no positive delta) => documented uniform 1/n fallback.
        share = 1.0 / n if degenerate else max(delta, 0.0) / positive_mass
        attributions.append(
            SentenceAttribution(sentence=unit_text, span=span, delta=delta, share=share)
        )

    attributions.sort(key=lambda a: (-a.share, a.span[0], a.span[1]))
    return attributions, degenerate


def explain_chunk(
    retriever: object,
    query: str,
    result: ScoredChunk,
    granularity: str = "sentence",
) -> Explanation:
    """Produce a full :class:`Explanation` for one ranked result.

    Computes the occlusion attribution and, for hybrid results carrying
    :class:`~why_this_chunk.types.ScoreComponents`, the lexical-vs-dense split.

    Args:
        retriever: The retriever that produced ``result`` (used to re-score
            occluded text on the same scale).
        query: The query string.
        result: The scored chunk to explain.
        granularity: ``"sentence"`` or ``"token"``.

    Returns:
        The explanation, with ``split`` populated only for hybrid results.
    """
    scorer = _make_scorer(retriever, query)
    attributions, degenerate = attribute(result.chunk.text, scorer, granularity)
    split: ContributionSplit | None = None
    if isinstance(retriever, HybridRetriever) and result.components is not None:
        split = compute_split(result.components)
    return Explanation(
        query=query,
        result=result,
        sentences=attributions,
        split=split,
        granularity=granularity,
        degenerate=degenerate,
    )
