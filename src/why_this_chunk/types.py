"""Core data types for ``why-this-chunk``.

This module defines the small, explicit value objects that flow through the
whole pipeline: retrieval, attribution, contribution splits, the failure
taxonomy, and the counterfactual minimal-fix search.

The types are intentionally plain :mod:`dataclasses` rather than pydantic
models.  They are created in tight inner loops (one :class:`SentenceAttribution`
per occluded sentence, one :class:`ScoredChunk` per candidate) where pydantic
validation overhead is not justified, and several of them are produced purely
internally from already-validated inputs.  Configuration objects that *do* form
a public contract (see :mod:`why_this_chunk.config`) use pydantic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "Chunk",
    "ContributionSplit",
    "DiagnosisResult",
    "Explanation",
    "FailureClass",
    "FixSuggestion",
    "ScoreComponents",
    "ScoredChunk",
    "SentenceAttribution",
]


@dataclass(frozen=True, slots=True)
class Chunk:
    """A unit of retrievable text.

    Attributes:
        id: Stable, unique identifier within a corpus. Used for all
            tie-breaking and for matching an expected chunk during diagnosis.
        text: The chunk body.
        metadata: Arbitrary user metadata; never interpreted by the library.
        source_document_id: Optional provenance — the id of the
            :class:`~why_this_chunk.source.SourceDocument` this chunk was cut
            from. ``None`` for pre-chunked corpora.
        span: Optional ``(start, end)`` character offsets into the source
            document. Present only when the chunk was produced by a
            :class:`~why_this_chunk.source.Chunker`. Enables the
            ``lost_to_chunking`` check and the ``chunk_size`` counterfactual
            axis to re-chunk deterministically.
    """

    id: str
    text: str
    metadata: dict[str, object] = field(default_factory=dict)
    source_document_id: str | None = None
    span: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class ScoreComponents:
    """The modality breakdown of a hybrid score.

    Both raw and normalized values are retained so the contribution split can
    report what the retriever saw *and* what entered the weighted sum.

    Attributes:
        dense: Normalized dense similarity in ``[0, 1]`` (``None`` if absent).
        lexical: Normalized lexical (BM25) score in ``[0, 1]`` (``None`` if
            absent).
        alpha: Hybrid mixing weight applied to the dense modality.
        dense_raw: Pre-normalization dense similarity, if known.
        lexical_raw: Pre-normalization lexical score, if known.
    """

    dense: float | None = None
    lexical: float | None = None
    alpha: float | None = None
    dense_raw: float | None = None
    lexical_raw: float | None = None


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk together with its score and rank in a single search."""

    chunk: Chunk
    score: float
    rank: int
    components: ScoreComponents | None = None


@dataclass(frozen=True, slots=True)
class SentenceAttribution:
    """Occlusion attribution for one sentence (or token) of a chunk.

    Attributes:
        sentence: The occluded text unit.
        span: ``(start, end)`` character offsets of the unit within the chunk.
        delta: ``score_full - score_occluded``; how much removing this unit
            lowered the chunk's score. May be negative.
        share: ``max(delta, 0) / sum(max(delta, 0))`` over the chunk, i.e. the
            unit's fraction of the positive attribution mass. In the degenerate
            all-non-positive case this is the uniform ``1 / n``.
    """

    sentence: str
    span: tuple[int, int]
    delta: float
    share: float


@dataclass(frozen=True, slots=True)
class ContributionSplit:
    """Lexical-vs-dense decomposition of a single hybrid result.

    Attributes:
        dense_n: Normalized dense score in ``[0, 1]``.
        lexical_n: Normalized lexical score in ``[0, 1]``.
        alpha: Mixing weight applied to the dense modality.
        dense_contribution: ``alpha * dense_n``.
        lexical_contribution: ``(1 - alpha) * lexical_n``.
        dominant: ``"dense"`` or ``"lexical"`` — whichever contribution is
            larger (``"dense"`` on an exact tie, by fixed convention).
    """

    dense_n: float
    lexical_n: float
    alpha: float
    dense_contribution: float
    lexical_contribution: float
    dominant: str


@dataclass(frozen=True, slots=True)
class Explanation:
    """The full explanation of why one chunk ranked where it did.

    Attributes:
        query: The query string.
        result: The scored chunk being explained.
        sentences: Per-sentence (or per-token) attributions, ordered by
            descending ``share`` then by span for determinism.
        split: The lexical-vs-dense split, or ``None`` for non-hybrid
            retrievers (never faked).
        granularity: ``"sentence"`` or ``"token"``.
        degenerate: ``True`` when no unit had a positive delta and shares fell
            back to uniform ``1 / n`` — i.e. no single unit dominates the score.
    """

    query: str
    result: ScoredChunk
    sentences: list[SentenceAttribution]
    split: ContributionSplit | None = None
    granularity: str = "sentence"
    degenerate: bool = False


class FailureClass(StrEnum):
    """Why a known-correct chunk failed to surface for a query.

    The values are a *retrieval-cause* label (why the chunk ranks where it does
    now), deliberately separated from fixability, which is reported by
    :class:`DiagnosisResult.fix`.
    """

    MISSING_FROM_INDEX = "missing_from_index"
    LOST_TO_CHUNKING = "lost_to_chunking"
    OUT_RANKED = "out_ranked"
    EMBEDDING_BLIND_SPOT = "embedding_blind_spot"


@dataclass(frozen=True, slots=True)
class FixSuggestion:
    """A single-axis config change that surfaces the expected chunk.

    Attributes:
        param: The axis name (``"top_k"``, ``"chunk_size"``, ``"alpha"``,
            ``"rerank"``).
        from_value: The current value of the axis.
        to_value: The value that surfaces the chunk.
        cost: Integer cost on the axis' documented scale; lower is cheaper.
        new_rank: The (0-based) rank the expected chunk reaches under the fix.
        explanation: Human-readable summary of the change.
    """

    param: str
    from_value: object
    to_value: object
    cost: int
    new_rank: int
    explanation: str


@dataclass(frozen=True, slots=True)
class DiagnosisResult:
    """The outcome of diagnosing a failed ``(query, expected_chunk)``.

    Attributes:
        failure_class: The dominant retrieval-cause label, or ``None`` only in
            the impossible-by-construction case where the id is present but no
            taxonomy branch applied.
        unevaluable: Classes that could not be tested with the available
            capabilities/provenance (e.g. ``lost_to_chunking`` without
            provenance). Recorded, never silently skipped.
        evidence: Supporting facts (rank-at-large-K, dense/lexical scores,
            thresholds used, human-readable note).
        fix: The cheapest config change that surfaces the chunk, or ``None`` if
            none was found within the bounded sweeps. Independent of
            ``failure_class``: a cause label never implies "unfixable".
    """

    failure_class: FailureClass | None
    unevaluable: list[FailureClass] = field(default_factory=list)
    evidence: dict[str, object] = field(default_factory=dict)
    fix: FixSuggestion | None = None
