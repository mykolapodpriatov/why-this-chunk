#!/usr/bin/env python3
"""Runnable, zero-download demo of ``why-this-chunk``.

Uses the deterministic :class:`FakeEmbedder`, so it runs fully offline with no
model downloads. It shows:

1. ``explain`` — attributing a top result's score to its sentences plus the
   hybrid lexical-vs-dense split; and
2. ``diagnose`` + the minimal fix — classifying why an expected chunk failed and
   the cheapest single config change that surfaces it.

Run it with::

    python examples/demo.py
"""

from __future__ import annotations

from rich.console import Console

from why_this_chunk import (
    BM25Retriever,
    Corpus,
    DenseRetriever,
    FakeEmbedder,
    HybridRetriever,
    RetrievalConfig,
    explain_chunk,
    search_fixes,
)
from why_this_chunk.report import render_diagnosis, render_explanation
from why_this_chunk.taxonomy import diagnose
from why_this_chunk.types import DiagnosisResult


def build_retriever() -> HybridRetriever:
    """Build a hybrid retriever over the bundled example corpus (offline)."""
    corpus = Corpus.from_jsonl("examples/corpus.jsonl")
    embedder = FakeEmbedder(seed=0)
    dense = DenseRetriever(corpus, embedder)
    lexical = BM25Retriever(corpus)
    return HybridRetriever(dense, lexical, alpha=0.5)


def main() -> None:
    """Run the explain and diagnose demos."""
    console = Console()
    retriever = build_retriever()

    # 1) Explain why the top result for a query ranked where it did.
    query = "What is the capital of France?"
    top = retriever.search(query, k=1)[0]
    explanation = explain_chunk(retriever, query, top)
    render_explanation(explanation, console)

    console.print()

    # 2) Diagnose a failure. For this query the "numpy" chunk out-ranks the
    #    expected "python" chunk at top_k=1, so "python" is reported as
    #    `out_ranked` and the cheapest fix is to raise top_k to 2.
    failing_query = "NumPy fast numerical arrays for Python data science"
    expected = "python"
    config = RetrievalConfig(top_k=1, alpha=0.5)
    result = diagnose(retriever, failing_query, expected, config)
    fixes = search_fixes(retriever, failing_query, expected, config)
    enriched = DiagnosisResult(
        failure_class=result.failure_class,
        unevaluable=result.unevaluable,
        evidence=result.evidence,
        fix=fixes.best,
    )
    render_diagnosis(enriched, console)


if __name__ == "__main__":
    main()
