"""Tests for the pgvector adapter.

Gated behind the ``pg`` extra: skipped cleanly via ``pytest.importorskip`` when
``psycopg`` is absent.

Two layers. The first drives the adapter through a fake connection, so the SQL
it builds, its parameter binding, its similarity conversion and its connection
ownership are checked with no database at all. The second runs the same adapter
against a real PostgreSQL + pgvector, skipped unless ``WHY_THIS_CHUNK_PG_DSN``
points at one -- the ``pgvector`` CI job sets it. Both layers use the
deterministic :class:`FakeEmbedder`, so nothing is downloaded.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("psycopg")

from why_this_chunk import Chunk, FakeEmbedder, RetrievalConfig, explain_chunk
from why_this_chunk.retrievers import Retriever
from why_this_chunk.retrievers.adapters.pgvector import (
    PgVectorRetriever,
    _quote_identifier,
    _vector_literal,
)

QUERY = "river Seine northern France"
PG_DSN = os.environ.get("WHY_THIS_CHUNK_PG_DSN")


class _FakeCursor:
    """Records the statement it was given and replays canned rows."""

    def __init__(self, owner: _FakeConnection) -> None:
        self._owner = owner

    def execute(self, query: str, params: object = ()) -> None:
        self._owner.statements.append((query, tuple(params)))  # type: ignore[arg-type]
        self._owner.last_is_count = query.startswith("SELECT count(*)")

    def fetchall(self) -> object:
        return [(self._owner.count,)] if self._owner.last_is_count else self._owner.rows

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


class _FakeConnection:
    """Minimal stand-in for a psycopg connection."""

    def __init__(self, rows: object = (), count: int = 0) -> None:
        self.rows = rows
        self.count = count
        self.statements: list[tuple[str, tuple[object, ...]]] = []
        self.last_is_count = False
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def close(self) -> None:
        self.closed = True


def _adapter(
    connection: _FakeConnection, embedder: FakeEmbedder, **kwargs: object
) -> PgVectorRetriever:
    return PgVectorRetriever(
        "postgresql:///unused",
        "chunks",
        embedder,
        connection=connection,
        **kwargs,  # type: ignore[arg-type]
    )


# --- identifier and literal helpers ----------------------------------------


@pytest.mark.parametrize("name", ["id", "chunk_id", "public.chunks", "_x$1"])
def test_valid_identifiers_are_quoted(name: str) -> None:
    """Each dotted part is quoted separately."""
    assert _quote_identifier(name) == ".".join(f'"{part}"' for part in name.split("."))


@pytest.mark.parametrize(
    "name",
    ['chunks"; DROP TABLE chunks; --', "chunks chunks", "1chunks", "", "a.", "a-b"],
)
def test_invalid_identifiers_are_rejected(name: str) -> None:
    """Identifiers are rejected outright rather than escaped."""
    with pytest.raises(ValueError, match="not a valid SQL identifier"):
        _quote_identifier(name)


def test_vector_literal_is_pgvectors_bracketed_form() -> None:
    """The literal round-trips floats at full precision."""
    assert _vector_literal(np.array([1, 0.5], dtype=np.float32)) == "[1.0,0.5]"


# --- query construction and mapping, without a database ---------------------


def test_satisfies_the_retriever_protocol(embedder: FakeEmbedder) -> None:
    """The adapter is structurally a Retriever."""
    assert isinstance(_adapter(_FakeConnection(), embedder), Retriever)


def test_search_binds_the_vector_and_limit_as_parameters(embedder: FakeEmbedder) -> None:
    """The vector and k are bound, never interpolated into the statement."""
    connection = _FakeConnection(rows=[("a", "some text", 0.0)])
    _adapter(connection, embedder).search(QUERY, 3)

    statement, params = connection.statements[0]
    assert "%s::vector" in statement
    assert params[1] == 3
    assert isinstance(params[0], str) and params[0].startswith("[")
    assert params[0] not in statement


@pytest.mark.parametrize(
    ("metric", "operator"),
    [("cosine", "<=>"), ("ip", "<#>"), ("l2", "<->")],
)
def test_metric_selects_its_operator(embedder: FakeEmbedder, metric: str, operator: str) -> None:
    """Each supported metric maps onto its pgvector operator."""
    connection = _FakeConnection(rows=[("a", "text", 0.0)])
    _adapter(connection, embedder, metric=metric).search(QUERY, 1)

    statement, _params = connection.statements[0]
    assert f'"embedding" {operator} %s::vector' in statement
    assert "ORDER BY distance ASC LIMIT %s" in statement


@pytest.mark.parametrize(
    ("metric", "distance", "expected"),
    [("cosine", 0.25, 0.75), ("ip", -0.75, 0.75), ("l2", 1.0, 0.5)],
)
def test_distances_convert_to_cosine_similarity(
    embedder: FakeEmbedder, metric: str, distance: float, expected: float
) -> None:
    """Every metric lands on the same cosine scale as score_text."""
    connection = _FakeConnection(rows=[("a", "text", distance)])
    results = _adapter(connection, embedder, metric=metric).search(QUERY, 1)

    assert results[0].score == pytest.approx(expected)


def test_unsupported_metric_is_rejected_at_construction(embedder: FakeEmbedder) -> None:
    """A bad metric fails fast, not on the first query."""
    with pytest.raises(ValueError, match="unsupported metric"):
        _adapter(_FakeConnection(), embedder, metric="jaccard")


def test_custom_table_and_columns_are_quoted_into_the_statement(
    embedder: FakeEmbedder,
) -> None:
    """Table and column names reach the statement quoted."""
    connection = _FakeConnection(rows=[("a", "text", 0.0)])
    PgVectorRetriever(
        "postgresql:///unused",
        "public.docs",
        embedder,
        id_column="chunk_id",
        text_column="body",
        embedding_column="vec",
        connection=connection,
    ).search(QUERY, 1)

    statement, _params = connection.statements[0]
    assert 'SELECT "chunk_id", "body", "vec" <=>' in statement
    assert 'FROM "public"."docs"' in statement


def test_metadata_column_is_selected_and_carried_onto_the_chunk(
    embedder: FakeEmbedder,
) -> None:
    """A configured jsonb column lands in Chunk.metadata."""
    connection = _FakeConnection(rows=[("a", "text", {"page": 3}, 0.0)])
    results = _adapter(connection, embedder, metadata_column="meta").search(QUERY, 1)

    statement, _params = connection.statements[0]
    assert 'SELECT "id", "text", "meta", "embedding" <=>' in statement
    assert results[0].chunk.metadata == {"page": 3}


def test_null_metadata_becomes_an_empty_dict(embedder: FakeEmbedder) -> None:
    """A row with no metadata is not an error."""
    connection = _FakeConnection(rows=[("a", "text", None, 0.0)])
    results = _adapter(connection, embedder, metadata_column="meta").search(QUERY, 1)

    assert results[0].chunk.metadata == {}


def test_non_object_metadata_is_rejected(embedder: FakeEmbedder) -> None:
    """A jsonb array or scalar cannot be chunk metadata."""
    connection = _FakeConnection(rows=[("a", "text", [1, 2], 0.0)])

    with pytest.raises(TypeError, match="JSON object or NULL"):
        _adapter(connection, embedder, metadata_column="meta").search(QUERY, 1)


def test_metadata_column_is_absent_from_the_query_by_default(embedder: FakeEmbedder) -> None:
    """Without a metadata column the query selects only id, text and distance."""
    connection = _FakeConnection(rows=[("a", "text", 0.0)])
    results = _adapter(connection, embedder).search(QUERY, 1)

    assert 'SELECT "id", "text", "embedding" <=>' in connection.statements[0][0]
    assert results[0].chunk.metadata == {}


def test_invalid_table_is_rejected_at_construction(embedder: FakeEmbedder) -> None:
    """An injection attempt in the table name never reaches a query."""
    with pytest.raises(ValueError, match="not a valid SQL identifier"):
        PgVectorRetriever(
            "postgresql:///unused",
            "chunks; DROP TABLE chunks",
            embedder,
            connection=_FakeConnection(),
        )


def test_results_are_ranked_and_carry_dense_components(embedder: FakeEmbedder) -> None:
    """Rows become descending-score results with dense-only components."""
    connection = _FakeConnection(rows=[("a", "one", 0.0), ("b", "two", 0.5), ("c", "three", 1.0)])
    results = _adapter(connection, embedder).search(QUERY, 3)

    assert [r.chunk.id for r in results] == ["a", "b", "c"]
    assert [r.rank for r in results] == [0, 1, 2]
    assert [r.chunk.text for r in results] == ["one", "two", "three"]
    for result in results:
        assert result.components is not None
        assert result.components.lexical_raw is None
        assert result.components.dense_raw == pytest.approx(result.score)


def test_ties_break_by_ascending_chunk_id(embedder: FakeEmbedder) -> None:
    """Equal-distance rows are ordered by chunk id, not by row order."""
    connection = _FakeConnection(rows=[("z", "t", 0.5), ("a", "t", 0.5), ("m", "t", 0.5)])
    results = _adapter(connection, embedder).search(QUERY, 3)

    assert [r.chunk.id for r in results] == ["a", "m", "z"]


def test_corpus_size_counts_the_table(embedder: FakeEmbedder) -> None:
    """corpus_size issues a count query."""
    connection = _FakeConnection(count=42)

    assert _adapter(connection, embedder).corpus_size == 42
    assert connection.statements[0][0].startswith('SELECT count(*) FROM "chunks"')


def test_reindex_is_refused_and_advertised_as_such(embedder: FakeEmbedder) -> None:
    """supports_reindex is False and reindex says why."""
    adapter = _adapter(_FakeConnection(), embedder)
    assert adapter.supports_reindex is False

    with pytest.raises(NotImplementedError, match="lives in the store"):
        adapter.reindex(RetrievalConfig())


def test_non_positive_k_never_reaches_the_database(embedder: FakeEmbedder) -> None:
    """A non-positive k short-circuits before any statement is executed."""
    connection = _FakeConnection(rows=[("a", "t", 0.0)])

    assert _adapter(connection, embedder).search(QUERY, 0) == []
    assert connection.statements == []


def test_empty_table_returns_no_results(embedder: FakeEmbedder) -> None:
    """No rows means no results, not an error."""
    assert _adapter(_FakeConnection(rows=[]), embedder).search(QUERY, 5) == []


def test_close_leaves_an_injected_connection_open(embedder: FakeEmbedder) -> None:
    """A caller-owned connection is the caller's to close."""
    connection = _FakeConnection(rows=[("a", "t", 0.0)])
    adapter = _adapter(connection, embedder)
    adapter.search(QUERY, 1)
    adapter.close()

    assert connection.closed is False


def test_close_closes_a_connection_the_adapter_opened(
    embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A connection opened from the DSN is closed, and closing twice is safe."""
    import psycopg

    opened = _FakeConnection(rows=[("a", "t", 0.0)])
    monkeypatch.setattr(psycopg, "connect", lambda _dsn: opened)
    adapter = PgVectorRetriever("postgresql:///unused", "chunks", embedder)

    adapter.search(QUERY, 1)
    adapter.close()
    adapter.close()

    assert opened.closed is True


# --- against a real PostgreSQL + pgvector -----------------------------------

pg = pytest.mark.skipif(PG_DSN is None, reason="WHY_THIS_CHUNK_PG_DSN is not set")


@pytest.fixture
def pg_table(embedder: FakeEmbedder, tiny_chunks: list[Chunk]) -> object:
    """Create a throwaway pgvector table holding the tiny corpus."""
    import psycopg

    assert PG_DSN is not None
    vectors = embedder.encode([chunk.text for chunk in tiny_chunks])
    connection = psycopg.connect(PG_DSN)
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute("DROP TABLE IF EXISTS wtc_chunks")
        cursor.execute(
            f"CREATE TABLE wtc_chunks (id text PRIMARY KEY, text text, "
            f"embedding vector({embedder.dim}))"
        )
        for chunk, vector in zip(tiny_chunks, vectors, strict=True):
            cursor.execute(
                "INSERT INTO wtc_chunks (id, text, embedding) VALUES (%s, %s, %s::vector)",
                (chunk.id, chunk.text, _vector_literal(vector)),
            )
    connection.commit()
    try:
        yield connection
    finally:
        with connection.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS wtc_chunks")
        connection.commit()
        connection.close()


def _real(embedder: FakeEmbedder, connection: object, metric: str = "cosine") -> PgVectorRetriever:
    assert PG_DSN is not None
    return PgVectorRetriever(PG_DSN, "wtc_chunks", embedder, metric=metric, connection=connection)


@pg
def test_real_ranking_matches_the_builtin_dense_retriever(
    pg_table: object, embedder: FakeEmbedder, dense: object
) -> None:
    """The same corpus and embedder produce the same order as DenseRetriever."""
    adapter = _real(embedder, pg_table)

    adapter_ids = [r.chunk.id for r in adapter.search(QUERY, 5)]
    dense_ids = [r.chunk.id for r in dense.search(QUERY, 5)]  # type: ignore[attr-defined]

    assert adapter_ids == dense_ids


@pg
@pytest.mark.parametrize("metric", ["cosine", "ip", "l2"])
def test_real_scores_match_cosine_similarity(
    pg_table: object, embedder: FakeEmbedder, metric: str
) -> None:
    """Every metric lands on the same scale as score_text."""
    adapter = _real(embedder, pg_table, metric)

    for result in adapter.search(QUERY, 5):
        assert result.score == pytest.approx(adapter.score_text(QUERY, result.chunk.text), abs=1e-5)


@pg
def test_real_corpus_size_and_determinism(pg_table: object, embedder: FakeEmbedder) -> None:
    """corpus_size counts the table and repeated searches agree."""
    adapter = _real(embedder, pg_table)

    assert adapter.corpus_size == 5
    assert adapter.search(QUERY, 5) == adapter.search(QUERY, 5)


@pg
def test_real_k_larger_than_the_table(pg_table: object, embedder: FakeEmbedder) -> None:
    """Asking for more rows than exist returns what's there."""
    assert len(_real(embedder, pg_table).search(QUERY, 99)) == 5


@pg
def test_real_explain_chunk_runs_over_the_adapter(pg_table: object, embedder: FakeEmbedder) -> None:
    """Occlusion attribution works against a real table."""
    adapter = _real(embedder, pg_table)
    top = adapter.search(QUERY, 1)[0]

    explanation = explain_chunk(adapter, QUERY, top)

    assert explanation.sentences
    assert sum(item.share for item in explanation.sentences) == pytest.approx(1.0)
    assert explanation.split is None


@pg
def test_real_connection_is_opened_from_the_dsn_when_not_injected(
    pg_table: object, embedder: FakeEmbedder
) -> None:
    """Without an injected connection the adapter opens (and closes) its own."""
    assert PG_DSN is not None
    adapter = PgVectorRetriever(PG_DSN, "wtc_chunks", embedder)
    try:
        results = adapter.search(QUERY, 2)
    finally:
        adapter.close()

    assert [r.chunk.id for r in results] == ["seine", "paris"]
