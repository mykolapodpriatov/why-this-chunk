"""Deterministic, offline hashing embedder.

:class:`FakeEmbedder` produces stable embeddings from token hashes — no model,
no download, no network. It is the default embedder for tests and demos and
makes the entire retrieval/attribution/counterfactual pipeline reproducible.

Construction:
    Each text is lowercased and tokenized on word characters. Every token is
    hashed deterministically (BLAKE2b over the UTF-8 bytes, seeded by ``seed``)
    into a column index and a sign, accumulating a bag-of-hashed-tokens vector
    (the "hashing trick"). The vector is then L2-normalized. Texts sharing
    tokens therefore have positively correlated vectors, which makes lexical
    overlap behave like semantic similarity — adequate for deterministic tests.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
from numpy.typing import NDArray

__all__ = ["FakeEmbedder"]

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class FakeEmbedder:
    """A deterministic hashing embedder (no model, fully offline).

    Args:
        dim: Embedding dimensionality (number of hash buckets). Must be >= 1.
        seed: Salt mixed into every token hash so different seeds yield
            different but still-deterministic embedding spaces.

    Raises:
        ValueError: If ``dim`` is not a positive integer.
    """

    def __init__(self, dim: int = 64, seed: int = 0) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self._dim = dim
        self._seed = seed
        self._salt = seed.to_bytes(8, "little", signed=False)

    @property
    def dim(self) -> int:
        """The embedding dimensionality."""
        return self._dim

    def _hash(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8, salt=self._salt[:8]).digest()
        value = int.from_bytes(digest, "little", signed=False)
        bucket = value % self._dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return bucket, sign

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        """Encode ``texts`` into an ``(n, dim)`` array of unit-norm rows.

        A text with no tokens (empty or punctuation-only) maps to an all-zero
        row — the single permitted exception to the unit-norm invariant.

        Args:
            texts: The strings to embed.

        Returns:
            An ``(len(texts), dim)`` ``float32`` array of L2-normalized rows.
        """
        matrix = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _tokenize(text):
                bucket, sign = self._hash(token)
                matrix[row, bucket] += sign
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix /= norms
        return matrix
