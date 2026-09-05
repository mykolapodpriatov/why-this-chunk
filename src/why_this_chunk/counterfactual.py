"""Bounded counterfactual search for the smallest config fix.

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

With ``max_axes=2`` a second pass tries **pairs** of axes, but only when the
single-axis pass found nothing. The common real case is a chunk that is both
split badly and out-ranked: a smaller ``chunk_size`` puts the answer in one
piece, and only then does a higher ``alpha`` pull it into the top-K. Neither
alone moves it, and reporting that query as unfixable sends someone off to
re-embed a corpus when two knobs would have done it.

The second pass is opt-in because of cost. Single-axis is the sum of the
per-axis candidate counts; pairs is the product, and one of those axes requires
a reindex per candidate. It is bounded by ``max_combinations``, and hitting that
bound is reported as ``capped`` rather than as "no fix": those two answers are
not the same and must not print the same.

A one-axis fix always outranks a two-axis one, even when the two-axis one is
cheaper by the cost metric. A reviewer applying it has to reason about one thing
instead of two, and this tool exists to be read by a person.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product

from why_this_chunk.config import (
    ALPHA_SWEEP,
    CHUNK_SIZE_SWEEP,
    RERANK_COST,
    RetrievalConfig,
)
from why_this_chunk.corpus import Corpus
from why_this_chunk.retrievers import Retriever
from why_this_chunk.retrievers.hybrid import HybridRetriever
from why_this_chunk.types import Chunk, FixSuggestion

__all__ = [
    "AXIS_PRIORITY",
    "DEFAULT_MAX_COMBINATIONS",
    "CounterfactualResult",
    "FixPlan",
    "search_fixes",
]

#: Ceiling on evaluated axis pairs. Each one can cost a reindex, so the search
#: reports that it stopped rather than pretending it exhausted the space.
DEFAULT_MAX_COMBINATIONS: int = 64

#: Fixed tie-break priority over axes (lower wins on equal cost).
AXIS_PRIORITY: dict[str, int] = {
    "top_k": 0,
    "alpha": 1,
    "chunk_size": 2,
    "rerank": 3,
}


@dataclass(frozen=True, slots=True)
class FixPlan:
    """One or more axis changes that together surface the chunk.

    A single-axis fix is a plan of length one, so callers have one shape to
    handle rather than two.

    Attributes:
        changes: The changes to apply, in axis-priority order.
        cost: Sum of the per-axis costs.
        new_rank: The 0-based rank the expected chunk reaches under the plan.
    """

    changes: tuple[FixSuggestion, ...]
    cost: int
    new_rank: int

    @property
    def axes(self) -> tuple[str, ...]:
        """The axis names this plan touches."""
        return tuple(change.param for change in self.changes)

    @property
    def explanation(self) -> str:
        """Human-readable summary, joining the changes when there are several."""
        return ", then ".join(change.explanation for change in self.changes)

    @classmethod
    def of(cls, fix: FixSuggestion) -> FixPlan:
        """Wrap a single-axis fix as a one-change plan."""
        return cls(changes=(fix,), cost=fix.cost, new_rank=fix.new_rank)


@dataclass(frozen=True, slots=True)
class CounterfactualResult:
    """Outcome of the counterfactual search.

    Attributes:
        best: The single cheapest one-axis fix, or ``None`` if none worked.
        all_fixes: Every one-axis fix found, ordered by ``(cost, axis_priority)``.
        unevaluable: Axis names that could not be tested (missing capability or
            provenance), recorded rather than silently skipped. A pair is
            unevaluable the moment either half is.
        pair_fixes: Two-axis plans found by the second pass, ordered the same
            way. Empty unless ``max_axes=2`` and the one-axis pass found
            nothing.
        capped: Whether the pair pass stopped on ``max_combinations``. An empty
            ``pair_fixes`` with ``capped`` set means "did not finish looking",
            which is not the same as "there is no fix".
    """

    best: FixSuggestion | None
    all_fixes: list[FixSuggestion]
    unevaluable: list[str]
    pair_fixes: list[FixPlan] = field(default_factory=list)
    capped: bool = False

    @property
    def best_plan(self) -> FixPlan | None:
        """The plan to recommend: the cheapest one-axis fix, else the cheapest pair.

        A one-axis fix always wins, even against a cheaper pair. A reviewer
        applying it has to reason about one change instead of two.
        """
        if self.best is not None:
            return FixPlan.of(self.best)
        return self.pair_fixes[0] if self.pair_fixes else None


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


def _covers(chunk: Chunk, expected: Chunk) -> bool:
    """Whether ``chunk`` fully contains ``expected``'s span in the same document.

    Mere overlap does not count: a window capturing only part of the expected
    text would not surface it intact, so recommending that size would be a
    false fix. Shared by the single-axis chunk-size sweep and the pair pass so
    the two cannot drift on what "the chunk is there" means.
    """
    return (
        expected.span is not None
        and chunk.source_document_id == expected.source_document_id
        and chunk.span is not None
        and chunk.span[0] <= expected.span[0]
        and chunk.span[1] >= expected.span[1]
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
            if _covers(scored.chunk, expected):
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


@dataclass(frozen=True, slots=True)
class _AxisCandidate:
    """One value an axis could take, with what it costs and how to apply it."""

    param: str
    from_value: object
    to_value: object
    cost: int
    updates: dict[str, object]
    explanation: str

    def as_fix(self, new_rank: int) -> FixSuggestion:
        """The public suggestion for this candidate at a known rank."""
        return FixSuggestion(
            param=self.param,
            from_value=self.from_value,
            to_value=self.to_value,
            cost=self.cost,
            new_rank=new_rank,
            explanation=self.explanation,
        )


def _chunk_size_candidates(config: RetrievalConfig) -> list[_AxisCandidate]:
    """Every swept chunk size other than the current one."""
    try:
        current_index = CHUNK_SIZE_SWEEP.index(config.chunk_size)
    except ValueError:
        current_index = _nearest_index(CHUNK_SIZE_SWEEP, config.chunk_size)
    return [
        _AxisCandidate(
            param="chunk_size",
            from_value=config.chunk_size,
            to_value=size,
            cost=abs(index - current_index),
            updates={"chunk_size": size},
            explanation=f"set chunk_size to {size} so the expected text stays in one chunk",
        )
        for index, size in enumerate(CHUNK_SIZE_SWEEP)
        if size != config.chunk_size
    ]


def _alpha_candidates(retriever: Retriever, config: RetrievalConfig) -> list[_AxisCandidate]:
    """Every swept alpha other than the current one."""
    current_alpha = getattr(retriever, "alpha", config.alpha)
    current_index = _nearest_index(ALPHA_SWEEP, current_alpha if current_alpha is not None else 0.0)
    return [
        _AxisCandidate(
            param="alpha",
            from_value=current_alpha,
            to_value=alpha,
            cost=abs(index - current_index),
            updates={"alpha": alpha},
            explanation=f"shift hybrid alpha from {current_alpha} to {alpha}",
        )
        for index, alpha in enumerate(ALPHA_SWEEP)
        if index != current_index
    ]


def _rerank_candidates(config: RetrievalConfig) -> list[_AxisCandidate]:
    """Turning the reranker on, when it is currently off."""
    if config.rerank:
        return []
    return [
        _AxisCandidate(
            param="rerank",
            from_value=False,
            to_value=True,
            cost=RERANK_COST,
            updates={"rerank": True},
            explanation="enable the reranker",
        )
    ]


def _reindexable_candidates(
    retriever: Retriever,
    config: RetrievalConfig,
    unevaluable: set[str],
) -> dict[str, list[_AxisCandidate]]:
    """Candidate values per axis, skipping any axis that cannot be evaluated.

    An axis already reported unevaluable by the single-axis pass stays out: a
    pair is unevaluable the moment either half is, and pretending otherwise
    would produce a fix nobody can apply.
    """
    axes: dict[str, list[_AxisCandidate]] = {}
    if "chunk_size" not in unevaluable:
        axes["chunk_size"] = _chunk_size_candidates(config)
    if "alpha" not in unevaluable and isinstance(retriever, HybridRetriever):
        axes["alpha"] = _alpha_candidates(retriever, config)
    if "rerank" not in unevaluable:
        axes["rerank"] = _rerank_candidates(config)
    return {name: values for name, values in axes.items() if values}


def _rank_under(
    retriever: Retriever,
    query: str,
    expected: Chunk,
    cfg: RetrievalConfig,
    k: int,
    *,
    rechunked: bool,
) -> int | None:
    """Rank of the expected chunk (or a chunk covering it) in a top-``k`` search.

    ``rechunked`` selects the matching rule. Reindexing at a different chunk
    size produces new chunk ids, so an id comparison would report "not found"
    for a chunk that is right there; the covering-span rule is used instead.
    """
    for scored in retriever.search(query, k):
        if scored.chunk.id == expected.id or (rechunked and _covers(scored.chunk, expected)):
            return scored.rank
    return None


def _evaluate_pair(
    retriever: Retriever,
    query: str,
    expected: Chunk,
    config: RetrievalConfig,
    first: _AxisCandidate,
    second: _AxisCandidate | None,
) -> FixPlan | None:
    """Apply one or two reindexable changes, optionally widening top_k, and check.

    ``second`` of ``None`` means the pair is "this axis plus top_k": the other
    change is applied, then the smallest sufficient ``top_k`` is derived from
    where the chunk actually lands. That is only a real pair when the chunk is
    still outside the current top-K afterwards; otherwise the axis alone was
    the fix and the single-axis pass already has it.
    """
    changes = [first] if second is None else [first, second]
    updates: dict[str, object] = {}
    for change in changes:
        updates.update(change.updates)
    new_config = config.with_updates(**updates)
    try:
        adjusted = retriever.reindex(new_config)
    except (NotImplementedError, ValueError):
        return None

    rechunked = "chunk_size" in updates
    if second is not None:
        rank = _rank_under(adjusted, query, expected, new_config, config.top_k, rechunked=rechunked)
        if rank is None:
            return None
        fixes = tuple(
            change.as_fix(rank) for change in sorted(changes, key=lambda c: AXIS_PRIORITY[c.param])
        )
        return FixPlan(changes=fixes, cost=sum(c.cost for c in changes), new_rank=rank)

    # Paired with top_k.
    ceiling = min(10 * config.top_k, adjusted.corpus_size)
    if ceiling <= config.top_k:
        return None
    rank = _rank_under(adjusted, query, expected, new_config, ceiling, rechunked=rechunked)
    if rank is None or rank < config.top_k:
        return None
    new_k = rank + 1
    top_k_change = _AxisCandidate(
        param="top_k",
        from_value=config.top_k,
        to_value=new_k,
        cost=new_k - config.top_k,
        updates={"top_k": new_k},
        explanation=f"raise top_k from {config.top_k} to {new_k} to include the chunk",
    )
    ordered = sorted([first, top_k_change], key=lambda c: AXIS_PRIORITY[c.param])
    return FixPlan(
        changes=tuple(change.as_fix(rank) for change in ordered),
        cost=first.cost + top_k_change.cost,
        new_rank=rank,
    )


def _search_pairs(
    retriever: Retriever,
    query: str,
    expected_chunk_id: str,
    config: RetrievalConfig,
    unevaluable: set[str],
    max_combinations: int,
) -> tuple[list[FixPlan], bool]:
    """Second pass over pairs of axes. Returns ``(plans, capped)``."""
    corpus = _corpus_of(retriever)
    expected = corpus.get(expected_chunk_id) if corpus is not None else None
    if expected is None:
        # Without the expected chunk there is nothing to match a rechunked
        # result against, and a pair involving chunk_size is the whole point.
        return [], False

    axes = _reindexable_candidates(retriever, config, unevaluable)
    plans: list[FixPlan] = []
    evaluated = 0

    def record(plan: FixPlan | None) -> None:
        if plan is not None:
            plans.append(plan)

    # Each reindexable axis paired with top_k.
    for candidates in axes.values():
        for candidate in candidates:
            if evaluated >= max_combinations:
                return _ordered(plans), True
            evaluated += 1
            record(_evaluate_pair(retriever, query, expected, config, candidate, None))

    # Pairs of two reindexable axes.
    for left, right in combinations(sorted(axes), 2):
        for first, second in product(axes[left], axes[right]):
            if evaluated >= max_combinations:
                return _ordered(plans), True
            evaluated += 1
            record(_evaluate_pair(retriever, query, expected, config, first, second))

    return _ordered(plans), False


def _ordered(plans: list[FixPlan]) -> list[FixPlan]:
    """Cheapest first, ties broken by axis priority so two runs agree."""
    return sorted(
        plans,
        key=lambda p: (p.cost, tuple(AXIS_PRIORITY[axis] for axis in p.axes)),
    )


def search_fixes(
    retriever: Retriever,
    query: str,
    expected_chunk_id: str,
    config: RetrievalConfig | None = None,
    *,
    max_axes: int = 1,
    max_combinations: int = DEFAULT_MAX_COMBINATIONS,
) -> CounterfactualResult:
    """Search for the cheapest config change that surfaces the chunk.

    Args:
        retriever: The retriever under test.
        query: The query string.
        expected_chunk_id: The id of the known-correct chunk.
        config: The active configuration; defaults to :class:`RetrievalConfig`.
        max_axes: 1 (the default) searches single axes only, which is today's
            behaviour and today's cost. 2 adds a second pass over pairs, run
            only when the first pass found nothing.
        max_combinations: Ceiling on evaluated pairs. Hitting it sets
            ``capped`` on the result rather than reporting no fix.

    Returns:
        A :class:`CounterfactualResult` with the best fix, the full ranked list,
        any axes reported unevaluable, and, when ``max_axes`` is 2 and nothing
        else worked, the pair plans.

    Raises:
        ValueError: If ``max_axes`` is not 1 or 2, or ``max_combinations`` is
            below 1.
    """
    if max_axes not in (1, 2):
        raise ValueError(f"max_axes must be 1 or 2, got {max_axes}")
    if max_combinations < 1:
        raise ValueError(f"max_combinations must be at least 1, got {max_combinations}")
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

    # The pair pass runs only when nothing simpler worked. It costs a reindex
    # per combination, and a one-axis fix would outrank whatever it found
    # anyway.
    pair_fixes: list[FixPlan] = []
    capped = False
    if best is None and max_axes == 2:
        pair_fixes, capped = _search_pairs(
            retriever, query, expected_chunk_id, cfg, set(unevaluable), max_combinations
        )

    return CounterfactualResult(
        best=best,
        all_fixes=fixes,
        unevaluable=unevaluable,
        pair_fixes=pair_fixes,
        capped=capped,
    )


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
