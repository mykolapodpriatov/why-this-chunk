"""The corpus: an ordered, id-addressable collection of chunks.

A :class:`Corpus` records whether it was built with provenance (from
:class:`~why_this_chunk.source.SourceDocument` objects via a
:class:`~why_this_chunk.source.Chunker`) or from pre-chunked
:class:`~why_this_chunk.types.Chunk` objects. The provenance flag and the
retained source documents are what make the ``chunk_size`` counterfactual axis
and the ``lost_to_chunking`` taxonomy check evaluable.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

from why_this_chunk.source import Chunker, SourceDocument
from why_this_chunk.types import Chunk

__all__ = ["Corpus"]


class Corpus:
    """An ordered collection of chunks with stable id lookup.

    Prefer the classmethods :meth:`from_chunks`, :meth:`from_sources`, and
    :meth:`from_jsonl` over calling the constructor directly.

    Args:
        chunks: The chunks, in a stable order.
        source_documents: The source documents the chunks were derived from,
            when known. Their presence (together with ``has_provenance``) gates
            the ``chunk_size`` axis and ``lost_to_chunking`` check.

    Raises:
        ValueError: If two chunks share an id.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        source_documents: Sequence[SourceDocument] | None = None,
    ) -> None:
        self._chunks: list[Chunk] = list(chunks)
        self._by_id: dict[str, Chunk] = {}
        for chunk in self._chunks:
            if chunk.id in self._by_id:
                raise ValueError(f"duplicate chunk id in corpus: {chunk.id!r}")
            self._by_id[chunk.id] = chunk
        self._sources: list[SourceDocument] = list(source_documents or [])

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk]) -> Corpus:
        """Build a corpus from pre-chunked chunks (no provenance)."""
        return cls(chunks, source_documents=None)

    @classmethod
    def from_sources(
        cls,
        docs: Sequence[SourceDocument],
        chunker: Chunker,
        chunk_size: int,
    ) -> Corpus:
        """Build a provenance-carrying corpus by chunking source documents.

        Args:
            docs: The source documents.
            chunker: The chunker producing provenance-carrying chunks.
            chunk_size: The character window size to chunk at.

        Returns:
            A corpus for which the ``chunk_size`` axis and ``lost_to_chunking``
            check are evaluable.
        """
        chunks = chunker.chunk(list(docs), chunk_size)
        return cls(chunks, source_documents=list(docs))

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Corpus:
        """Load a corpus from a JSON-Lines file, preserving any provenance.

        Each line must be a JSON object with at least ``id`` and ``text``;
        optional ``metadata`` (object), ``source_document_id`` (string), and
        ``span`` (two-element array) are honored. Blank lines are skipped.

        When **every** chunk carries both ``source_document_id`` and ``span``
        *and* the spans of each document form a clean reconstruction (sorted by
        start they begin at ``0`` and tile the document with no gaps and no
        conflicting overlaps), the original source documents are reconstructed
        from the chunk text and spans, so the resulting corpus reports
        ``has_provenance=True`` and the ``chunk_size`` axis / ``lost_to_chunking``
        check become evaluable. If any chunk lacks that provenance, or any
        document's spans are gapped, start at a nonzero offset, or overlap with
        conflicting text, the corpus is treated as pre-chunked (no source
        documents) rather than fabricating source text that never existed.

        Args:
            path: Path to the ``.jsonl`` file.

        Returns:
            The loaded corpus, with provenance preserved when present on every
            chunk.

        Raises:
            ValueError: If a line is not valid JSON or lacks ``id``/``text``.
        """
        file_path = Path(path)
        chunks: list[Chunk] = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{file_path}:{line_number}: invalid JSON ({exc.msg})"
                    ) from exc
                if not isinstance(record, dict) or "id" not in record or "text" not in record:
                    raise ValueError(
                        f"{file_path}:{line_number}: each line needs 'id' and 'text' keys"
                    )
                span_value = record.get("span")
                span: tuple[int, int] | None = None
                if isinstance(span_value, list) and len(span_value) == 2:
                    span = (int(span_value[0]), int(span_value[1]))
                source_id = record.get("source_document_id")
                chunks.append(
                    Chunk(
                        id=str(record["id"]),
                        text=str(record["text"]),
                        metadata=dict(record.get("metadata") or {}),
                        source_document_id=str(source_id) if source_id is not None else None,
                        span=span,
                    )
                )
        sources = _reconstruct_sources(chunks)
        if sources is None:
            return cls.from_chunks(chunks)
        return cls(chunks, source_documents=sources)

    @property
    def chunks(self) -> list[Chunk]:
        """The chunks in stable order (a copy)."""
        return list(self._chunks)

    @property
    def source_documents(self) -> list[SourceDocument]:
        """The source documents, if any (a copy)."""
        return list(self._sources)

    @property
    def has_provenance(self) -> bool:
        """Whether source documents are available for re-chunking.

        ``True`` only when the corpus retains source documents, which is the
        precondition for the ``chunk_size`` axis and ``lost_to_chunking`` check.
        """
        return bool(self._sources)

    def get(self, chunk_id: str) -> Chunk | None:
        """Return the chunk with ``chunk_id``, or ``None`` if absent."""
        return self._by_id.get(chunk_id)

    def contains(self, chunk_id: str) -> bool:
        """Whether a chunk with ``chunk_id`` is present."""
        return chunk_id in self._by_id

    def __len__(self) -> int:
        return len(self._chunks)

    def __iter__(self) -> Iterator[Chunk]:
        return iter(self._chunks)


def _reconstruct_sources(chunks: list[Chunk]) -> list[SourceDocument] | None:
    """Rebuild source documents from provenance-carrying chunks.

    Returns the reconstructed documents (in first-seen order) only when the
    chunks support a *faithful* reconstruction; otherwise returns ``None`` so the
    caller falls back to a pre-chunked corpus rather than fabricate source text.

    Reconstruction is attempted only when **every** chunk carries both
    ``source_document_id`` and ``span``. It is then accepted only when, for
    **each** ``source_document_id``, the chunk spans cover the document with no
    invented characters — exactly the kind of output a :class:`Chunker`
    produces. Concretely, per document, with spans sorted by start:

    * the first span starts at ``0`` (no unobserved leading prefix);
    * spans are contiguous or overlapping with **no gaps** (each next
      ``start`` is ``<=`` the running covered ``end``), so no interior region is
      left unobserved;
    * every overlapping region is **consistent** — all chunks covering a
      character agree on it (no conflicting source text);
    * each chunk's span length equals ``len(chunk.text)``.

    If any document violates these (gap, nonzero leading start, conflicting
    overlap, or inconsistent length), provenance is declined for the whole
    corpus. This keeps the ``chunk_size`` axis and ``lost_to_chunking`` check
    re-chunkable on genuine provenance while never inventing source text.
    """
    if not chunks:
        return None
    if any(c.source_document_id is None or c.span is None for c in chunks):
        return None

    order: list[str] = []
    grouped: dict[str, list[tuple[int, int, str]]] = {}
    for chunk in chunks:
        # Narrowing for the type checker; guaranteed non-None by the guard above.
        doc_id = chunk.source_document_id
        span = chunk.span
        assert doc_id is not None and span is not None
        start, end = span
        if start < 0 or end < start or (end - start) != len(chunk.text):
            # Span is inconsistent with the text length: cannot faithfully
            # reconstruct, so decline provenance rather than fabricate it.
            return None
        if doc_id not in grouped:
            grouped[doc_id] = []
            order.append(doc_id)
        grouped[doc_id].append((start, end, chunk.text))

    documents: list[SourceDocument] = []
    for doc_id in order:
        text = _reconstruct_one(grouped[doc_id])
        if text is None:
            # This document's spans do not form a clean reconstruction (gap,
            # nonzero leading start, or conflicting overlap): decline provenance
            # for the whole corpus rather than fabricate any source text.
            return None
        documents.append(SourceDocument(id=doc_id, text=text))
    return documents


def _reconstruct_one(spans: list[tuple[int, int, str]]) -> str | None:
    """Reconstruct one document's text from its ``(start, end, text)`` spans.

    Returns the exact source text when the spans, sorted by start, begin at
    ``0`` and cover a prefix with no gaps and no conflicting overlaps; returns
    ``None`` otherwise so the caller can decline provenance.
    """
    chars: list[str] = []
    covered = 0  # Length of the contiguous, gap-free prefix covered so far.
    for start, end, text in sorted(spans):
        if start > covered:
            # A gap between the covered prefix and this span (and, for the very
            # first span, a nonzero leading start): the in-between characters
            # were never observed, so reconstruction would have to invent them.
            return None
        if len(chars) < end:
            chars.extend([""] * (end - len(chars)))
        for offset, character in enumerate(text):
            position = start + offset
            existing = chars[position]
            if existing and existing != character:
                # An overlapping region where two chunks disagree on the source
                # character: the spans conflict and cannot be reconciled.
                return None
            chars[position] = character
        covered = max(covered, end)
    return "".join(chars)
