"""Source documents, the chunker protocol, and provenance.

A :class:`~why_this_chunk.corpus.Corpus` can be built two ways:

* from pre-chunked :class:`~why_this_chunk.types.Chunk` objects (no provenance),
  in which case the ``chunk_size`` axis and ``lost_to_chunking`` check are
  reported *unevaluable*; or
* from :class:`SourceDocument` objects plus a :class:`Chunker` (provenance
  present), which makes those features evaluable because every produced chunk
  carries ``source_document_id`` and a ``span`` into its source.

:class:`FixedSizeChunker` is a deterministic built-in chunker (character windows
with optional overlap) used by tests and demos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from why_this_chunk.types import Chunk

__all__ = ["Chunker", "FixedSizeChunker", "SourceDocument"]


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """Raw source text prior to chunking.

    Attributes:
        id: Stable, unique document identifier.
        text: The full document body.
    """

    id: str
    text: str


@runtime_checkable
class Chunker(Protocol):
    """Splits source documents into chunks carrying provenance.

    Every returned :class:`~why_this_chunk.types.Chunk` MUST set
    ``source_document_id`` and ``span`` (the ``(start, end)`` char range it was
    cut from), so an expected chunk can be mapped back to its source region and
    re-chunked deterministically.
    """

    def chunk(self, docs: list[SourceDocument], chunk_size: int) -> list[Chunk]:
        """Chunk ``docs`` into provenance-carrying chunks at ``chunk_size``."""
        ...


class FixedSizeChunker:
    """Fixed-size character-window chunker with optional overlap.

    Chunk ids are deterministic and stable for a given ``(doc.id, chunk_size,
    overlap)``: ``"{doc_id}::{chunk_size}::{window_index}"``. This stability
    lets the ``lost_to_chunking`` check and the ``chunk_size`` axis re-chunk and
    compare without ambiguity.

    Args:
        overlap: Number of characters each window shares with the previous one.
            Must be non-negative and strictly less than every ``chunk_size``
            used.

    Raises:
        ValueError: If ``overlap`` is negative.
    """

    def __init__(self, overlap: int = 0) -> None:
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        self._overlap = overlap

    @property
    def overlap(self) -> int:
        """The configured inter-window character overlap."""
        return self._overlap

    def chunk(self, docs: list[SourceDocument], chunk_size: int) -> list[Chunk]:
        """Cut every document into ``chunk_size``-char windows.

        Args:
            docs: The source documents to chunk.
            chunk_size: Window size in characters (must be >= 1 and strictly
                greater than ``overlap``).

        Returns:
            The chunks in document order, each carrying provenance.

        Raises:
            ValueError: If ``chunk_size`` is not greater than ``overlap`` (which
                would make the window advance non-positively and loop forever).
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if chunk_size <= self._overlap:
            raise ValueError(
                f"chunk_size ({chunk_size}) must be greater than overlap "
                f"({self._overlap}) so windows make forward progress"
            )
        step = chunk_size - self._overlap
        chunks: list[Chunk] = []
        for doc in docs:
            text = doc.text
            length = len(text)
            if length == 0:
                continue
            window = 0
            start = 0
            while start < length:
                end = min(start + chunk_size, length)
                chunks.append(
                    Chunk(
                        id=f"{doc.id}::{chunk_size}::{window}",
                        text=text[start:end],
                        metadata={},
                        source_document_id=doc.id,
                        span=(start, end),
                    )
                )
                window += 1
                start += step
        return chunks
