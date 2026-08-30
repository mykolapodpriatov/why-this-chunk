"""pgvector adapter (behind the ``[pg]`` extra).

Wraps a PostgreSQL table that already holds chunk texts and their embeddings,
so an existing index can be explained without re-ingesting it. See
:mod:`why_this_chunk.retrievers.adapters` for what the adapter boundary does and
does not support.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from why_this_chunk.embedders import Embedder
from why_this_chunk.retrievers.adapters import VectorStoreRetriever, require
from why_this_chunk.types import Chunk

__all__ = ["PgVectorRetriever"]

#: pip extra that provides this adapter's backend.
EXTRA = "pg"
#: Backend module name, import-guarded.
BACKEND_MODULE = "psycopg"

#: Supported metrics, mapped to their pgvector operator. Each is converted back
#: to a cosine similarity exactly, given the unit-norm vectors the ``Embedder``
#: contract guarantees: ``<=>`` is ``1 - cos``, ``<#>`` is ``-cos``, and ``<->``
#: satisfies ``d**2 = 2 - 2*cos``.
_OPERATORS: Mapping[str, str] = {
    "cosine": "<=>",
    "ip": "<#>",
    "l2": "<->",
}

#: Unquoted SQL identifier: a letter or underscore, then letters/digits/_/$.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


class _Cursor(Protocol):
    """Structural type for the subset of a psycopg cursor this adapter calls."""

    def execute(self, query: str, params: Sequence[object] = ...) -> object:
        """Execute ``query`` with the given bound parameters."""
        ...

    def fetchall(self) -> Sequence[Sequence[object]]:
        """Return every remaining result row."""
        ...

    def __enter__(self) -> _Cursor: ...

    def __exit__(self, *exc: object) -> None: ...


class _Connection(Protocol):
    """Structural type for the subset of a psycopg connection this adapter calls."""

    def cursor(self) -> _Cursor:
        """Return a new cursor."""
        ...

    def close(self) -> None:
        """Close the connection."""
        ...


class _Psycopg(Protocol):
    """Structural type for the ``psycopg`` module surface this adapter calls."""

    def connect(self, conninfo: str) -> _Connection:
        """Open a connection to ``conninfo``."""
        ...


def _quote_identifier(name: str) -> str:
    """Return ``name`` as a quoted SQL identifier, or raise ``ValueError``.

    Table and column names cannot be bound as query parameters, so they are
    validated against :data:`_IDENTIFIER` and then double-quoted. A dotted name
    is treated as ``schema.table`` and each part is validated separately.
    Anything else -- whitespace, quotes, semicolons -- is rejected outright
    rather than escaped, so no caller-supplied string reaches a query
    unvalidated.
    """
    parts = name.split(".")
    if not all(_IDENTIFIER.match(part) for part in parts):
        raise ValueError(f"not a valid SQL identifier: {name!r}")
    return ".".join(f'"{part}"' for part in parts)


def _vector_literal(vector: NDArray[np.float32]) -> str:
    """Render ``vector`` as a pgvector text literal such as ``[1.0,0.0]``.

    Bound as a parameter and cast with ``::vector`` in the query, so the adapter
    works without ``pgvector.psycopg.register_vector``.
    """
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


def _metadata(value: object) -> dict[str, object]:
    """Return a ``jsonb`` column's value as chunk metadata.

    psycopg already decodes ``jsonb`` into Python objects, so this only has to
    reject the shapes a chunk's metadata cannot be. ``NULL`` is the common case
    for rows that carry none.
    """
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    raise TypeError(f"metadata column must hold a JSON object or NULL, got {type(value).__name__}")


class PgVectorRetriever(VectorStoreRetriever):
    """Explain an existing PostgreSQL + pgvector table.

    The table must hold each chunk's text as well as its embedding, since every
    result carries the chunk body — attribution occludes sentences of it.

    Queries are a single ``ORDER BY <distance> LIMIT k``, so a pgvector index
    does the work. All three supported metrics convert back to a cosine
    similarity exactly for the unit-norm vectors the
    :class:`~why_this_chunk.embedders.Embedder` contract guarantees, which keeps
    scores on the same scale as :meth:`score_text` and the built-in dense
    retriever.

    The connection is opened lazily on the first query and reused. Pass
    ``connection`` to reuse one you already own (from a pool, say); an injected
    connection is never closed by :meth:`close`.

    Args:
        dsn: A PostgreSQL connection string.
        table: The table holding the chunks. May be ``schema.table``.
        embedder: The embedder that produced the stored vectors.
        id_column: Column holding the chunk id.
        text_column: Column holding the chunk text.
        embedding_column: Column holding the ``vector`` value.
        metadata_column: Optional ``jsonb`` column carried through to
            :attr:`Chunk.metadata`. Left out of the query when ``None``.
        metric: One of ``cosine``, ``ip`` or ``l2``. Must match the operator
            class of the index on ``embedding_column``, or Postgres falls back
            to a sequential scan.
        connection: An open psycopg connection to reuse instead of ``dsn``.

    Raises:
        MissingDependencyError: If ``psycopg`` is not installed.
        ValueError: If ``metric`` is unsupported, or if any table or column name
            is not a valid SQL identifier.
    """

    def __init__(
        self,
        dsn: str,
        table: str,
        embedder: Embedder,
        *,
        id_column: str = "id",
        text_column: str = "text",
        embedding_column: str = "embedding",
        metadata_column: str | None = None,
        metric: str = "cosine",
        connection: object | None = None,
    ) -> None:
        self._psycopg = require(BACKEND_MODULE, backend="pgvector", extra=EXTRA)
        super().__init__(embedder)
        if metric not in _OPERATORS:
            supported = ", ".join(sorted(_OPERATORS))
            raise ValueError(f"unsupported metric {metric!r}; expected one of: {supported}")
        self._dsn = dsn
        self._metric = metric
        self._connection = connection
        self._owns_connection = connection is None
        self._has_metadata = metadata_column is not None
        quoted_table = _quote_identifier(table)
        columns = [_quote_identifier(id_column), _quote_identifier(text_column)]
        if metadata_column is not None:
            columns.append(_quote_identifier(metadata_column))
        self._count_sql = f"SELECT count(*) FROM {quoted_table}"
        self._search_sql = (
            f"SELECT {', '.join(columns)}, "
            f"{_quote_identifier(embedding_column)} {_OPERATORS[metric]} %s::vector "
            f"AS distance "
            f"FROM {quoted_table} ORDER BY distance ASC LIMIT %s"
        )

    @property
    def metric(self) -> str:
        """The configured distance metric."""
        return self._metric

    def _similarity(self, distance: float) -> float:
        """Convert the operator's result into a cosine similarity."""
        if self._metric == "cosine":
            return 1.0 - distance
        if self._metric == "ip":
            return -distance
        return 1.0 - (distance * distance) / 2.0

    def _connect(self) -> _Connection:
        """Return the connection, opening one from ``dsn`` on first use."""
        if self._connection is None:
            self._connection = cast(_Psycopg, self._psycopg).connect(self._dsn)
        return cast(_Connection, self._connection)

    def _count(self) -> int:
        """Return the number of rows in the table."""
        with self._connect().cursor() as cursor:
            cursor.execute(self._count_sql, ())
            rows = cursor.fetchall()
        return int(cast(int, rows[0][0])) if rows else 0

    def _row(self, row: Sequence[object]) -> tuple[Chunk, float]:
        """Turn one result row into a ``(chunk, cosine_similarity)`` pair.

        The distance is always the last column; the metadata column sits between
        the text and the distance only when one was configured.
        """
        chunk = Chunk(
            id=str(row[0]),
            text=str(row[1]),
            metadata=_metadata(row[2]) if self._has_metadata else {},
        )
        return chunk, self._similarity(float(cast(float, row[-1])))

    def _query(self, vector: NDArray[np.float32], k: int) -> Sequence[tuple[Chunk, float]]:
        """Return up to ``k`` ``(chunk, cosine_similarity)`` pairs."""
        with self._connect().cursor() as cursor:
            cursor.execute(self._search_sql, (_vector_literal(vector), k))
            rows = cursor.fetchall()
        return [self._row(row) for row in rows]

    def close(self) -> None:
        """Close the connection, if this retriever opened it.

        A connection passed in as ``connection`` belongs to the caller and is
        left open. Calling this more than once is harmless.
        """
        if self._owns_connection and self._connection is not None:
            cast(_Connection, self._connection).close()
            self._connection = None
