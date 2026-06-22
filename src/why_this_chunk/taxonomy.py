"""Per-query failure taxonomy: why a known-correct chunk failed to surface.

Given ``(query, expected_chunk_id, retriever, config)`` the classifier returns a
:class:`~why_this_chunk.types.DiagnosisResult` whose ``failure_class`` is a
*retrieval-cause* label decided by this **decision order** (first matching branch
wins; unevaluable branches are recorded in ``unevaluable`` and skipped, never
silently mislabeled):

1. ``missing_from_index`` — the expected id is not in the retriever's corpus.
2. ``lost_to_chunking`` — *evaluable only with provenance* (``SourceDocument``\\ s
   + ``Chunker``) and ``supports_reindex``. Re-chunk at the shared
   :data:`~why_this_chunk.config.CHUNK_SIZE_SWEEP` sizes; if a re-chunked chunk
   covering the expected text scores within ``top_k``, classify here. Without
   provenance/reindex, append to ``unevaluable`` and fall through.
3. ``out_ranked`` — the expected chunk appears within a large-K search at rank
   ``>= top_k``, where ``large_K = min(10 * top_k, corpus_size)``. Membership
   within ``large_K`` is the **sole** discriminator versus branch 4.
4. ``embedding_blind_spot`` — terminal: present, not provably lost to chunking,
   yet absent from ``large_K``. Evidence wording is shaped by ``dense_threshold``
   / ``lexical_threshold`` but the *label* is never.

**Cause vs fixability:** ``failure_class`` says what the dominant cause is now;
:attr:`DiagnosisResult.fix` (the counterfactual) says the cheapest change that
surfaces it. They are independent fields — a label never reads as "unfixable".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from why_this_chunk.config import CHUNK_SIZE_SWEEP, RetrievalConfig
from why_this_chunk.corpus import Corpus
from why_this_chunk.retrievers import Retriever
from why_this_chunk.types import Chunk, DiagnosisResult, FailureClass

__all__ = [
    "DEFAULT_DENSE_THRESHOLD",
    "DEFAULT_LEXICAL_THRESHOLD",
    "LARGE_K_MULTIPLIER",
    "diagnose",
    "large_k",
]

#: Multiplier in ``large_K = min(LARGE_K_MULTIPLIER * top_k, corpus_size)``.
LARGE_K_MULTIPLIER: int = 10

#: Default normalized dense score below which the dense modality is considered
#: to be "missing" a chunk (shapes evidence wording only).
DEFAULT_DENSE_THRESHOLD: float = 0.3

#: Default normalized lexical score above which a chunk is "lexically relevant"
#: (shapes evidence wording only).
DEFAULT_LEXICAL_THRESHOLD: float = 0.5


def large_k(top_k: int, corpus_size: int) -> int:
    """Return ``min(LARGE_K_MULTIPLIER * top_k, corpus_size)``.

    This is the documented constant separating ``out_ranked`` (inside ``large_K``)
    from ``embedding_blind_spot`` (outside it).
    """
    return min(LARGE_K_MULTIPLIER * top_k, corpus_size)


def _corpus_of(retriever: Retriever) -> Corpus | None:
    """Return the retriever's corpus if it exposes one, else ``None``."""
    corpus = getattr(retriever, "corpus", None)
    return corpus if isinstance(corpus, Corpus) else None


def _rank_within(results: list[Chunk], expected_id: str) -> int | None:
    """Return the 0-based rank of ``expected_id`` in chunk results, or ``None``."""
    for rank, chunk in enumerate(results):
        if chunk.id == expected_id:
            return rank
    return None


def _raw_scores_of(modality: object, query: str) -> NDArray[np.float64] | None:
    """Call ``raw_scores(query)`` on ``modality`` if it exposes it."""
    method = getattr(modality, "raw_scores", None)
    if callable(method):
        result = method(query)
        return np.asarray(result, dtype=np.float64)
    return None


def _modality_scores(
    retriever: Retriever, query: str
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, list[Chunk] | None]:
    """Best-effort raw dense/lexical score arrays aligned to corpus order.

    Structural (duck-typed), so any retriever exposing ``_dense``/``_lexical``
    sub-modalities or its own ``raw_scores`` participates — built-in hybrid,
    dense, lexical, and third-party look-alikes alike. Returns ``None`` arrays
    for retrievers that expose no usable scoring path.
    """
    corpus = _corpus_of(retriever)
    chunks = corpus.chunks if corpus is not None else None

    dense_mod = getattr(retriever, "_dense", None)
    lexical_mod = getattr(retriever, "_lexical", None)
    if dense_mod is not None or lexical_mod is not None:
        return (
            _raw_scores_of(dense_mod, query),
            _raw_scores_of(lexical_mod, query),
            chunks,
        )

    # A standalone dense or lexical retriever: classify which modality it is by
    # its advertised component shape via a probe result.
    own = _raw_scores_of(retriever, query)
    if own is None:
        return None, None, chunks
    probe = retriever.search(query, 1)
    if probe and probe[0].components is not None:
        components = probe[0].components
        if components.lexical_raw is not None and components.dense_raw is None:
            return None, own, chunks
    return own, None, chunks


def _normalized_at(
    scores: NDArray[np.float64] | None,
    chunks: list[Chunk] | None,
    expected_id: str,
) -> float | None:
    """Min-max normalized score of ``expected_id`` within ``scores``."""
    if scores is None or chunks is None or scores.size == 0:
        return None
    index = next((i for i, c in enumerate(chunks) if c.id == expected_id), None)
    if index is None:
        return None
    from why_this_chunk._scoring import min_max_normalize

    return float(min_max_normalize(scores)[index])


def _lost_to_chunking(
    retriever: Retriever,
    query: str,
    expected: Chunk,
    config: RetrievalConfig,
) -> tuple[bool, dict[str, object]]:
    """Test whether re-chunking surfaces a chunk covering the expected text.

    For each size in :data:`CHUNK_SIZE_SWEEP`, reindex and search at ``top_k``;
    a hit is any returned chunk that shares the expected chunk's source document
    and whose span **fully contains** the expected span. Mere overlap is
    insufficient: if the expected text is split across two re-chunked windows,
    neither window surfaces the whole text, so it is not "lost to chunking" in a
    way this size fixes. Caller guarantees provenance and ``supports_reindex``.

    Returns:
        ``(is_lost, evidence)`` where ``evidence`` records the winning size and
        rank when lost.
    """
    if expected.source_document_id is None or expected.span is None:
        return False, {}
    exp_start, exp_end = expected.span
    for size in CHUNK_SIZE_SWEEP:
        if size == config.chunk_size:
            continue
        try:
            rechunked = retriever.reindex(config.with_updates(chunk_size=size))
        except NotImplementedError:
            continue
        results = rechunked.search(query, config.top_k)
        for rank, scored in enumerate(results):
            chunk = scored.chunk
            if (
                chunk.source_document_id == expected.source_document_id
                and chunk.span is not None
                and chunk.span[0] <= exp_start
                and chunk.span[1] >= exp_end
            ):
                return True, {
                    "rechunk_size": size,
                    "rechunk_rank": rank,
                    "rechunk_chunk_id": chunk.id,
                }
    return False, {}


def diagnose(
    retriever: Retriever,
    query: str,
    expected_chunk_id: str,
    config: RetrievalConfig | None = None,
    *,
    dense_threshold: float = DEFAULT_DENSE_THRESHOLD,
    lexical_threshold: float = DEFAULT_LEXICAL_THRESHOLD,
) -> DiagnosisResult:
    """Classify why ``expected_chunk_id`` failed to surface for ``query``.

    Args:
        retriever: The retriever under test.
        query: The query string.
        expected_chunk_id: The id of the known-correct chunk.
        config: The active configuration; defaults to :class:`RetrievalConfig`.
        dense_threshold: Normalized dense score below which evidence reads
            "dense model misses a lexically-relevant chunk". Shapes evidence
            wording only — never the branch decision.
        lexical_threshold: Normalized lexical score above which the chunk counts
            as lexically relevant for that same evidence string.

    Returns:
        A :class:`DiagnosisResult` with the cause label, any unevaluable
        branches, and supporting evidence. The ``fix`` field is left ``None``
        here; callers compose it from :mod:`why_this_chunk.counterfactual`.
    """
    cfg = config or RetrievalConfig()
    top_k = cfg.top_k
    corpus_size = retriever.corpus_size
    unevaluable: list[FailureClass] = []
    evidence: dict[str, object] = {
        "top_k": top_k,
        "corpus_size": corpus_size,
        "dense_threshold": dense_threshold,
        "lexical_threshold": lexical_threshold,
    }

    # Branch 1: missing from index.
    corpus = _corpus_of(retriever)
    present = corpus.contains(expected_chunk_id) if corpus is not None else None
    if present is False:
        evidence["note"] = "expected chunk id is not present in the corpus"
        return DiagnosisResult(
            failure_class=FailureClass.MISSING_FROM_INDEX,
            unevaluable=unevaluable,
            evidence=evidence,
        )
    if present is None:
        # Retriever does not expose a corpus; we cannot test membership.
        unevaluable.append(FailureClass.MISSING_FROM_INDEX)

    expected_chunk = corpus.get(expected_chunk_id) if corpus is not None else None

    # Branch 2: lost to chunking (provenance + reindex required).
    can_rechunk = (
        corpus is not None
        and corpus.has_provenance
        and retriever.supports_reindex
        and expected_chunk is not None
        and expected_chunk.span is not None
    )
    if can_rechunk:
        assert expected_chunk is not None
        is_lost, lost_evidence = _lost_to_chunking(retriever, query, expected_chunk, cfg)
        if is_lost:
            evidence.update(lost_evidence)
            evidence["note"] = (
                "a re-chunked window covering the expected text ranks within "
                "top_k; the current chunk_size split it apart"
            )
            return DiagnosisResult(
                failure_class=FailureClass.LOST_TO_CHUNKING,
                unevaluable=unevaluable,
                evidence=evidence,
            )
    else:
        unevaluable.append(FailureClass.LOST_TO_CHUNKING)

    # Branches 3 & 4 share a single large-K search; membership is the sole
    # discriminator.
    lk = large_k(top_k, corpus_size)
    evidence["large_k"] = lk
    results = retriever.search(query, lk)
    rank = _rank_within([scored.chunk for scored in results], expected_chunk_id)
    evidence["rank_at_large_k"] = rank

    dense_scores, lexical_scores, chunks = _modality_scores(retriever, query)
    dense_n = _normalized_at(dense_scores, chunks, expected_chunk_id)
    lexical_n = _normalized_at(lexical_scores, chunks, expected_chunk_id)
    evidence["dense_score"] = dense_n
    evidence["lexical_score"] = lexical_n

    if rank is not None and rank >= top_k:
        evidence["note"] = (
            f"expected chunk is retrievable at rank {rank} (>= top_k={top_k}) "
            f"within large_k={lk}; a higher top_k alone would surface it"
        )
        return DiagnosisResult(
            failure_class=FailureClass.OUT_RANKED,
            unevaluable=unevaluable,
            evidence=evidence,
        )

    if rank is not None and rank < top_k:
        # Already inside top_k: the chunk is retrieved and not currently failing
        # under this configuration. ``out_ranked`` is a failure *cause* and must
        # never be attached to a non-failing chunk, so report no failure class.
        evidence["note"] = (
            f"expected chunk already ranks at {rank} within top_k={top_k}; "
            "retrieved within top_k; not currently failing under this configuration"
        )
        return DiagnosisResult(
            failure_class=None,
            unevaluable=unevaluable,
            evidence=evidence,
        )

    # Branch 4 (terminal): present but outside large_k. This requires that
    # presence is established — i.e. the retriever exposed a corpus we could test
    # membership against. When membership is unevaluable (no corpus) and the
    # chunk is also absent from large_k, we cannot prove it is *present*, so the
    # terminal "present-but-blind" verdict is unsound; report it as
    # indeterminate (``failure_class=None``) instead.
    if FailureClass.MISSING_FROM_INDEX in unevaluable:
        evidence["note"] = (
            "expected chunk is absent from large_k and the retriever exposes no "
            "corpus, so its presence cannot be established; the failure cause is "
            "indeterminate"
        )
        return DiagnosisResult(
            failure_class=None,
            unevaluable=unevaluable,
            evidence=evidence,
        )

    if (
        dense_n is not None
        and lexical_n is not None
        and dense_n < dense_threshold
        and lexical_n > lexical_threshold
    ):
        evidence["note"] = "dense model misses a lexically-relevant chunk"
    else:
        evidence["note"] = (
            "expected chunk is present but ranks below large_k "
            f"(dense_n={dense_n}, lexical_n={lexical_n})"
        )
    return DiagnosisResult(
        failure_class=FailureClass.EMBEDDING_BLIND_SPOT,
        unevaluable=unevaluable,
        evidence=evidence,
    )
