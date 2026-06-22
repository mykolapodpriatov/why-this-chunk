"""Embedder protocol and built-in implementations.

An :class:`Embedder` maps a list of strings to an ``(n, d)`` array of
**L2-normalized** row vectors, so that a dot product is a cosine similarity.
The default :class:`~why_this_chunk.embedders.fake.FakeEmbedder` is a
deterministic hashing embedder requiring no model download, which keeps the
entire pipeline reproducible and offline. The real
:class:`~why_this_chunk.embedders.sentence_transformers.SentenceTransformerEmbedder`
is an opt-in extra.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from why_this_chunk.embedders.fake import FakeEmbedder

__all__ = ["Embedder", "FakeEmbedder"]


@runtime_checkable
class Embedder(Protocol):
    """Maps texts to L2-normalized dense vectors.

    Implementations MUST return an array of shape ``(len(texts), dim)`` whose
    rows are unit-norm (a zero input vector maps to an all-zero row, which is
    the only permitted exception). This invariant lets dense retrieval treat a
    dot product as a cosine similarity.
    """

    @property
    def dim(self) -> int:
        """The embedding dimensionality."""
        ...

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        """Encode ``texts`` into an ``(n, dim)`` array of unit-norm rows."""
        ...
