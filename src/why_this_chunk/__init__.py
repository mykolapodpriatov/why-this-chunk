"""why-this-chunk: a RAG retrieval explainer.

Given a query and an existing index, this package shows *why* each chunk ranked
where it did (sentence-level occlusion attribution plus a lexical-vs-dense split
for hybrid search), classifies *why* a known-correct chunk failed (a per-query
failure taxonomy), and finds the *single smallest config change* that would have
surfaced it.

The public surface is re-exported here for convenience; submodules remain
importable directly.
"""

from __future__ import annotations

from why_this_chunk.attribution import attribute, explain_chunk
from why_this_chunk.config import (
    ALPHA_SWEEP,
    CHUNK_SIZE_SWEEP,
    RetrievalConfig,
    load_project_config,
)
from why_this_chunk.contributions import compute_split, split_for_result
from why_this_chunk.corpus import Corpus
from why_this_chunk.counterfactual import CounterfactualResult, search_fixes
from why_this_chunk.embedders import Embedder, FakeEmbedder
from why_this_chunk.retrievers import Retriever
from why_this_chunk.retrievers.bm25 import BM25Retriever
from why_this_chunk.retrievers.dense import DenseRetriever
from why_this_chunk.retrievers.hybrid import HybridRetriever
from why_this_chunk.source import Chunker, FixedSizeChunker, SourceDocument
from why_this_chunk.taxonomy import diagnose, large_k
from why_this_chunk.types import (
    Chunk,
    ContributionSplit,
    DiagnosisResult,
    Explanation,
    FailureClass,
    FixSuggestion,
    ScoreComponents,
    ScoredChunk,
    SentenceAttribution,
)

__version__ = "0.1.0"

# Grouped by concern for readability rather than alphabetized.
__all__ = [  # noqa: RUF022
    "__version__",
    # types
    "Chunk",
    "ScoreComponents",
    "ScoredChunk",
    "SentenceAttribution",
    "ContributionSplit",
    "Explanation",
    "FailureClass",
    "FixSuggestion",
    "DiagnosisResult",
    # config
    "RetrievalConfig",
    "CHUNK_SIZE_SWEEP",
    "ALPHA_SWEEP",
    "load_project_config",
    # corpus / source
    "Corpus",
    "SourceDocument",
    "Chunker",
    "FixedSizeChunker",
    # embedders
    "Embedder",
    "FakeEmbedder",
    # retrievers
    "Retriever",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    # analysis
    "attribute",
    "explain_chunk",
    "compute_split",
    "split_for_result",
    "diagnose",
    "large_k",
    "search_fixes",
    "CounterfactualResult",
]
