"""Bounded counterfactual search for the single smallest config fix.

Searches a fixed, ordered set of *single-axis* changes and returns the lowest
``cost`` change that pulls the expected chunk into the top-K. Every sweep is
finite and config-capped; there is no unbounded search.

Axes, with their documented integer cost scales:

* ``top_k`` — smallest K that includes the chunk; ``cost = new_k - top_k``.
  Always evaluable.
* ``chunk_size`` — try the shared :data:`~why_this_chunk.config.CHUNK_SIZE_SWEEP`
  sizes via ``reindex``; ``cost = |index distance|`` in the sweep. **Requires
  corpus provenance + ``supports_reindex``; otherwise reported unevaluable.**
* ``alpha`` (hybrid only) — sweep :data:`~why_this_chunk.config.ALPHA_SWEEP`;
  ``cost = steps moved``. Requires a hybrid retriever with ``supports_reindex``.
* ``rerank`` — toggle on; ``cost`` is the fixed :data:`RERANK_COST`. Requires
  ``supports_reindex`` and a configured reranker; otherwise reported
  unevaluable.

"Minimal" = lowest cost; ties broken by the fixed axis priority
``top_k < alpha < chunk_size < rerank``.
"""

from __future__ import annotations

from dataclasses import dataclass

from why_this_chunk.config import (
    ALPHA_SWEEP,
    CHUNK_SIZE_SWEEP,
    RERANK_COST,
    RetrievalConfig,
)
from why_this_chunk.corpus import Corpus
from why_this_chunk.retrievers import Retriever
from why_this_chunk.retrievers.hybrid import HybridRetriever
from why_this_chunk.types import FixSuggestion

__all__ = ["AXIS_PRIORITY", "CounterfactualResult", "search_fixes"]

#: Fixed tie-break priority over axes (lower wins on equal cost).
AXIS_PRIORITY: dict[str, int] = {
    "top_k": 0,
    "alpha": 1,
    "chunk_size": 2,
    "rerank": 3,
}


@dataclass(frozen=True, slots=True)
class CounterfactualResult:
    """Outcome of the counterfactual search.

    Attributes:
        best: The single cheapest fix, or ``None`` if no bounded change worked.
        all_fixes: Every fix found, ordered by ``(cost, axis_priority)``.
        unevaluable: Axis names that could not be tested (missing capability or
            provenance), recorded rather than silently skipped.
    """

    best: FixSuggestion | None
    all_fixes: list[FixSuggestion]
    unevaluable: list[str]


def _rank_in(retriever: Retriever, query: str, expected_id: str, k: int) -> int | None:
    """0-based rank of ``expected_id`` in a top-``k`` search, or ``None``."""
    for scored in retriever.search(query, k):
        if scored.chunk.id == expected_id:
            return scored.rank
    return None


def _corpus_of(retriever: Retriever) -> Corpus | None:
    corpus = getattr(retriever, "corpus", None)
    return corpus if isinstance(corpus, Corpus) else None


def _try_top_k(
    retriever: Retriever, query: str, expected_id: str, config: RetrievalConfig
) -> FixSuggestion | None:
    """Find the smallest K (> current top_k) that includes the chunk."""
    corpus_size = retriever.corpus_size
    ceiling = min(10 * config.top_k, corpus_size)
    if ceiling <= config.top_k:
        return None
    rank = _rank_in(retriever, query, expected_id, ceiling)
    if rank is None or rank < config.top_k:
        return None
    new_k = rank + 1
    return FixSuggestion(
        param="top_k",
        from_value=config.top_k,
        to_value=new_k,
        cost=new_k - config.top_k,
        new_rank=rank,
        explanation=f"raise top_k from {config.top_k} to {new_k} to include the chunk",
    )


def _try_chunk_size(
    retriever: Retriever,
    query: str,
    expected_id: str,
    config: RetrievalConfig,
) -> tuple[FixSuggestion | None, bool]:
    """Sweep chunk sizes; return ``(fix, evaluable)``.

    A fix is recorded when, after reindexing at a swept size, a chunk covering
    the expected text (same source document, whose span **fully contains** the
    expected span) lands within ``top_k``. Mere overlap does not count: a window
    that captures only part of the expected text would not surface it intact, so
    recommending that size would be a false fix. ``evaluable`` is ``False`` when
    provenance/reindex is missing.
    """
    corpus = _corpus_of(retriever)
    if corpus is None or not corpus.has_provenance or not retriever.supports_reindex:
        return None, False
    expected = corpus.get(expected_id)
    if expected is None or expected.span is None or expected.source_document_id is None:
        return None, False

    exp_start, exp_end = expected.span
    try:
        current_index = CHUNK_SIZE_SWEEP.index(config.chunk_size)
    except ValueError:
        current_index = _nearest_index(CHUNK_SIZE_SWEEP, config.chunk_size)

    best: FixSuggestion | None = None
    for index, size in enumerate(CHUNK_SIZE_SWEEP):
        if size == config.chunk_size:
            continue
        try:
            rechunked = retriever.reindex(config.with_updates(chunk_size=size))
        except NotImplementedError:
            return None, False
        for scored in rechunked.search(query, config.top_k):
            chunk = scored.chunk
            if (
                chunk.source_document_id == expected.source_document_id
                and chunk.span is not None
                and chunk.span[0] <= exp_start
                and chunk.span[1] >= exp_end
            ):
                cost = abs(index - current_index)
                candidate = FixSuggestion(
                    param="chunk_size",
                    from_value=config.chunk_size,
                    to_value=size,
                    cost=cost,
                    new_rank=scored.rank,
                    explanation=(
                        f"set chunk_size to {size} so the expected text stays in "
                        f"one chunk (ranks {scored.rank})"
                    ),
                )
                if best is None or candidate.cost < best.cost:
                    best = candidate
                break
    return best, True


def _try_alpha(
    retriever: Retriever,
    query: str,
    expected_id: str,
    config: RetrievalConfig,
) -> tuple[FixSuggestion | None, bool]:
    """Sweep hybrid alpha; return ``(fix, evaluable)``."""
    if not isinstance(retriever, HybridRetriever) or not retriever.supports_reindex:
        return None, False
    current_alpha = retriever.alpha
    current_index = _nearest_index(ALPHA_SWEEP, current_alpha)

    best: FixSuggestion | None = None
    for index, alpha in enumerate(ALPHA_SWEEP):
        if index == current_index:
            continue
        rechunked = retriever.reindex(config.with_updates(alpha=alpha))
        rank = _rank_in(rechunked, query, expected_id, config.top_k)
        if rank is not None:
            cost = abs(index - current_index)
            candidate = FixSuggestion(
                param="alpha",
                from_value=current_alpha,
                to_value=alpha,
                cost=cost,
                new_rank=rank,
                explanation=(f"shift hybrid alpha from {current_alpha} to {alpha} (ranks {rank})"),
            )
            if best is None or candidate.cost < best.cost:
                best = candidate
    return best, True


def _try_rerank(
    retriever: Retriever,
    query: str,
    expected_id: str,
    config: RetrievalConfig,
) -> tuple[FixSuggestion | None, bool]:
    """Toggle the reranker on; return ``(fix, evaluable)``.

    Evaluable only when the retriever advertises a configured reranker via a
    ``supports_rerank`` property and supports reindexing; otherwise unevaluable.
    """
    supports_rerank = bool(getattr(retriever, "supports_rerank", False))
    if not supports_rerank or not retriever.supports_reindex:
        # No configured reranker (or no reindex): the axis cannot be tested.
        return None, False
    if config.rerank:
        # Already on: nothing to toggle, but the axis was evaluable.
        return None, True
    reranked = retriever.reindex(config.with_updates(rerank=True))
    rank = _rank_in(reranked, query, expected_id, config.top_k)
    if rank is None:
        return None, True
    return (
        FixSuggestion(
            param="rerank",
            from_value=False,
            to_value=True,
            cost=RERANK_COST,
            new_rank=rank,
            explanation=f"enable the reranker to surface the chunk (ranks {rank})",
        ),
        True,
    )


def search_fixes(
    retriever: Retriever,
    query: str,
    expected_chunk_id: str,
    config: RetrievalConfig | None = None,
) -> CounterfactualResult:
    """Search all axes for the cheapest single change that surfaces the chunk.

    Args:
        retriever: The retriever under test.
        query: The query string.
        expected_chunk_id: The id of the known-correct chunk.
        config: The active configuration; defaults to :class:`RetrievalConfig`.

    Returns:
        A :class:`CounterfactualResult` with the best fix, the full ranked list,
        and any axes reported unevaluable.
    """
    cfg = config or RetrievalConfig()
    fixes: list[FixSuggestion] = []
    unevaluable: list[str] = []

    top_k_fix = _try_top_k(retriever, query, expected_chunk_id, cfg)
    if top_k_fix is not None:
        fixes.append(top_k_fix)

    alpha_fix, alpha_ok = _try_alpha(retriever, query, expected_chunk_id, cfg)
    if not alpha_ok:
        unevaluable.append("alpha")
    elif alpha_fix is not None:
        fixes.append(alpha_fix)

    chunk_fix, chunk_ok = _try_chunk_size(retriever, query, expected_chunk_id, cfg)
    if not chunk_ok:
        unevaluable.append("chunk_size")
    elif chunk_fix is not None:
        fixes.append(chunk_fix)

    rerank_fix, rerank_ok = _try_rerank(retriever, query, expected_chunk_id, cfg)
    if not rerank_ok:
        unevaluable.append("rerank")
    elif rerank_fix is not None:
        fixes.append(rerank_fix)

    fixes.sort(key=lambda f: (f.cost, AXIS_PRIORITY[f.param]))
    best = fixes[0] if fixes else None
    return CounterfactualResult(best=best, all_fixes=fixes, unevaluable=unevaluable)


def _nearest_index(sweep: tuple[float, ...] | tuple[int, ...], value: float) -> int:
    """Index of the sweep entry closest to ``value`` (ties pick the lower)."""
    best_index = 0
    best_distance = abs(sweep[0] - value)
    for index in range(1, len(sweep)):
        distance = abs(sweep[index] - value)
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index
