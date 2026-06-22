"""Retriever protocol and capability model.

The minimal adapter boundary a third-party retriever must implement is a single
method, :meth:`Retriever.search`. Richer behaviour is *advertised* through two
boolean capability properties so the rest of the system degrades gracefully
instead of guessing:

* :attr:`Retriever.supports_components` — can populate
  :class:`~why_this_chunk.types.ScoreComponents` (the dense/lexical split). If
  ``False``, ``explain``'s split is ``None`` and contributions are skipped.
* :attr:`Retriever.supports_reindex` — implements :meth:`Retriever.reindex`,
  returning a *new* retriever under a different
  :class:`~why_this_chunk.config.RetrievalConfig`. The ``alpha``/``rerank`` axes
  need only ``reindex``; the ``chunk_size`` axis additionally needs a corpus
  built with provenance.

When a capability or provenance is missing, the dependent counterfactual axis
and the ``lost_to_chunking`` check are reported **unevaluable**, never silently
skipped.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from why_this_chunk.config import RetrievalConfig
from why_this_chunk.types import ScoredChunk

__all__ = ["Retriever"]


@runtime_checkable
class Retriever(Protocol):
    """The retriever adapter boundary.

    Third-party adapters need only implement :meth:`search` and the two
    capability properties (returning ``False`` is fine). The built-in adapters
    implement everything.
    """

    @property
    def corpus_size(self) -> int:
        """Number of chunks the retriever can return."""
        ...

    @property
    def supports_components(self) -> bool:
        """Whether results carry a dense/lexical :class:`ScoreComponents`."""
        ...

    @property
    def supports_reindex(self) -> bool:
        """Whether :meth:`reindex` is implemented."""
        ...

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Return the top-``k`` scored chunks for ``query``, best first.

        Args:
            query: The query string.
            k: Maximum number of results (the implementation returns at most
                ``min(k, corpus_size)``).

        Returns:
            Scored chunks ordered by descending score, ties broken by ascending
            chunk id for determinism, with 0-based ``rank`` set.
        """
        ...

    def reindex(self, config: RetrievalConfig) -> Retriever:
        """Return a new retriever configured by ``config``.

        Only required when :attr:`supports_reindex` is ``True``.

        Raises:
            NotImplementedError: If the adapter does not support reindexing.
        """
        ...
