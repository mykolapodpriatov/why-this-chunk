"""Shared score-normalization with the documented degenerate fallback.

Min-max normalization is used by the hybrid retriever to combine modalities.
The fallback when ``max == min`` (identical scores, or a pool of size 1) is to
map every value to ``0.5`` so the hybrid formula is always well-defined and
never divides by zero. This is the single source of truth for that rule.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["DEGENERATE_NORM_VALUE", "min_max_normalize"]

#: The value every score collapses to when a modality's pool is degenerate
#: (``max == min``). Documented and shared so retrievers/contributions agree.
DEGENERATE_NORM_VALUE: float = 0.5


def min_max_normalize(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """Min-max normalize ``scores`` to ``[0, 1]`` with a degenerate fallback.

    Args:
        scores: A 1-D array of raw scores.

    Returns:
        A 1-D array of the same shape. When ``max(scores) == min(scores)`` (all
        identical, or a single element), every output is
        :data:`DEGENERATE_NORM_VALUE` (``0.5``); otherwise outputs are the
        affine map of ``scores`` onto ``[0, 1]``. An empty input returns an
        empty array.
    """
    if scores.size == 0:
        return scores.astype(np.float64, copy=True)
    lo = float(scores.min())
    hi = float(scores.max())
    if hi == lo:
        return np.full(scores.shape, DEGENERATE_NORM_VALUE, dtype=np.float64)
    return (scores - lo) / (hi - lo)
