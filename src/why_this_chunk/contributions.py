"""Lexical-vs-dense contribution split for hybrid results.

For a hybrid result the combined score is ``alpha * dense_n + (1 - alpha) *
lexical_n``. This module reports the two normalized scores, their weighted
contributions, and which modality dominated. For non-hybrid retrievers the split
is reported as ``None`` (never faked), which the callers honor.
"""

from __future__ import annotations

from why_this_chunk.types import ContributionSplit, ScoreComponents, ScoredChunk

__all__ = ["compute_split", "split_for_result"]


def compute_split(components: ScoreComponents) -> ContributionSplit:
    """Decompose hybrid :class:`ScoreComponents` into a contribution split.

    Args:
        components: Score components carrying normalized ``dense``/``lexical``
            values and ``alpha``.

    Returns:
        The split with weighted contributions and the dominant modality
        (``"dense"`` on an exact tie, by fixed convention).

    Raises:
        ValueError: If ``components`` lacks the dense/lexical/alpha fields
            required for a split (i.e. it is not a hybrid component set).
    """
    if components.dense is None or components.lexical is None or components.alpha is None:
        raise ValueError(
            "compute_split requires hybrid components with dense, lexical and alpha all set"
        )
    alpha = components.alpha
    dense_n = components.dense
    lexical_n = components.lexical
    dense_contribution = alpha * dense_n
    lexical_contribution = (1.0 - alpha) * lexical_n
    dominant = "dense" if dense_contribution >= lexical_contribution else "lexical"
    return ContributionSplit(
        dense_n=dense_n,
        lexical_n=lexical_n,
        alpha=alpha,
        dense_contribution=dense_contribution,
        lexical_contribution=lexical_contribution,
        dominant=dominant,
    )


def split_for_result(result: ScoredChunk) -> ContributionSplit | None:
    """Return the contribution split for a result, or ``None`` if not hybrid.

    Args:
        result: A scored chunk, possibly carrying hybrid components.

    Returns:
        The split when ``result`` carries complete hybrid components; otherwise
        ``None`` (explicitly, not faked).
    """
    components = result.components
    if (
        components is None
        or components.dense is None
        or components.lexical is None
        or components.alpha is None
    ):
        return None
    return compute_split(components)
