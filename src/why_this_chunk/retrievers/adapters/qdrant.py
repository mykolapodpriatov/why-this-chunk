"""Qdrant adapter (behind the ``[qdrant]`` extra).

Wraps a Qdrant collection that already holds chunk texts and their embeddings,
so an existing index can be explained without re-ingesting it. See
:mod:`why_this_chunk.retrievers.adapters` for what the adapter boundary does and
does not support.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from why_this_chunk.embedders import Embedder
from why_this_chunk.retrievers.adapters import VectorStoreRetriever, require
from why_this_chunk.types import Chunk

__all__ = ["QdrantRetriever"]

#: pip extra that provides this adapter's backend.
EXTRA = "qdrant"
#: Backend module name, import-guarded.
BACKEND_MODULE = "qdrant_client"

#: Distance metrics that convert back to cosine similarity for unit vectors.
#: Cosine and Dot already *are* the similarity; Euclid satisfies
#: ``d**2 = 2 - 2*cos``. Manhattan has no such identity, so it is rejected.
_COSINE_METRICS = frozenset({"cosine", "dot"})
_EUCLID_METRIC = "euclid"


class _CountResult(Protocol):
    """Structural type for ``qdrant_client``'s count response."""

    count: int


class _ScoredPoint(Protocol):
    """Structural type for a single Qdrant search hit."""

    id: object
    score: float
    payload: Mapping[str, object] | None


class _QueryResponse(Protocol):
    """Structural type for ``query_points``' response envelope."""

    points: Sequence[_ScoredPoint]


class _QdrantClient(Protocol):
    """Structural type for the subset of ``QdrantClient`` this adapter calls.

    Kept local (rather than importing ``qdrant_client`` for typing) so this
    module never needs the optional dependency to type-check.
    """

    def count(self, collection_name: str, exact: bool = ...) -> _CountResult:
        """Return the number of points stored in the collection."""
        ...

    def get_collection(self, collection_name: str) -> object:
        """Return the collection's info record, including its vector params."""
        ...

    def query_points(
        self,
        collection_name: str,
        query: Sequence[float],
        using: str | None = ...,
        limit: int = ...,
        with_payload: bool = ...,
    ) -> _QueryResponse:
        """Return the nearest points to ``query``."""
        ...


def _distance_name(info: object, vector_name: str | None) -> str:
    """Return the lowercased distance metric of a collection, or ``""``.

    ``params.vectors`` is either a single ``VectorParams`` (unnamed vector) or a
    ``{name: VectorParams}`` mapping (named vectors).
    """
    params = getattr(getattr(info, "config", None), "params", None)
    vectors = getattr(params, "vectors", None)
    if isinstance(vectors, Mapping):
        vectors = vectors.get(vector_name) if vector_name is not None else None
    distance = getattr(vectors, "distance", None)
    name = getattr(distance, "value", distance)
    return name.lower() if isinstance(name, str) else ""


class QdrantRetriever(VectorStoreRetriever):
    """Explain an existing Qdrant collection.

    The collection must store each chunk's text in its payload, since every
    result carries the chunk body — attribution occludes sentences of it. The
    remaining payload keys become :attr:`~why_this_chunk.types.Chunk.metadata`.

    Qdrant point ids are integers or UUIDs, so a corpus keyed by string chunk
    ids normally carries the real id in the payload. ``id_field`` names that
    key; without it, ``str(point.id)`` is used.

    The collection's distance metric is read once at construction. ``Cosine``
    and ``Dot`` scores are already cosine similarities for the unit-norm vectors
    the :class:`~why_this_chunk.embedders.Embedder` contract guarantees;
    ``Euclid`` is converted exactly via ``cos = 1 - d**2 / 2``. ``Manhattan``
    has no such identity and is rejected rather than silently mis-scored.

    Args:
        client: A ``qdrant_client.QdrantClient`` instance.
        collection_name: The collection to query.
        embedder: The embedder that produced the stored vectors.
        text_field: Payload key holding the chunk text.
        id_field: Payload key holding the chunk id. Defaults to the point id.
        vector_name: Named vector to query, for collections defining several.

    Raises:
        MissingDependencyError: If ``qdrant_client`` is not installed.
        ValueError: If the collection's distance metric cannot be converted to
            a cosine similarity.
    """

    def __init__(
        self,
        client: object,
        collection_name: str,
        embedder: Embedder,
        *,
        text_field: str = "text",
        id_field: str | None = None,
        vector_name: str | None = None,
    ) -> None:
        require(BACKEND_MODULE, backend="qdrant", extra=EXTRA)
        super().__init__(embedder)
        self._client = client
        self._collection_name = collection_name
        self._text_field = text_field
        self._id_field = id_field
        self._vector_name = vector_name
        info = cast(_QdrantClient, client).get_collection(collection_name)
        self._metric = _distance_name(info, vector_name)
        if self._metric not in _COSINE_METRICS and self._metric != _EUCLID_METRIC:
            raise ValueError(
                f"collection {collection_name!r} uses the {self._metric or 'unknown'!r} "
                "distance, which does not convert to a cosine similarity; use a "
                "Cosine, Dot or Euclid collection"
            )

    @property
    def metric(self) -> str:
        """The collection's distance metric, lowercased."""
        return self._metric

    def _similarity(self, score: float) -> float:
        """Convert Qdrant's score into a cosine similarity."""
        if self._metric == _EUCLID_METRIC:
            return 1.0 - (score * score) / 2.0
        return score

    def _chunk(self, point: _ScoredPoint) -> Chunk:
        """Build a :class:`Chunk` from a hit's payload."""
        payload = dict(point.payload or {})
        text = payload.pop(self._text_field, None)
        chunk_id = payload.pop(self._id_field, None) if self._id_field else None
        return Chunk(
            id=str(chunk_id) if chunk_id is not None else str(point.id),
            text=str(text) if text is not None else "",
            metadata=payload,
        )

    def _count(self) -> int:
        """Return the number of points in the collection."""
        client = cast(_QdrantClient, self._client)
        return int(client.count(self._collection_name, exact=True).count)

    def _query(self, vector: NDArray[np.float32], k: int) -> Sequence[tuple[Chunk, float]]:
        """Return up to ``k`` ``(chunk, cosine_similarity)`` pairs."""
        client = cast(_QdrantClient, self._client)
        response = client.query_points(
            collection_name=self._collection_name,
            query=[float(value) for value in vector],
            using=self._vector_name,
            limit=k,
            with_payload=True,
        )
        return [
            (self._chunk(point), self._similarity(float(point.score))) for point in response.points
        ]
