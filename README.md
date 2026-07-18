# why-this-chunk

> Retrieval explainer for RAG — shows why a chunk ranked where it did, the lexical-vs-dense split, and the one config change that would have surfaced the right answer.

![status](https://img.shields.io/badge/status-beta-blue) ![python](https://img.shields.io/badge/python-3.11%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)

Point it at a query and an existing retriever/index; it produces per-result explanations: sentence-level (or token-level) attribution of the similarity score and a lexical (BM25) vs dense contribution split for hybrid search. When a known-correct chunk is missing, it classifies *why* and finds the smallest config change that fixes that one query.

## Why

RAG debugging today is mostly guesswork. This tells you which layer is actually at fault — index, chunking, ranking, or the embedding model — and the smallest change that fixes a specific failing query.

## Features

- **Score attribution** — per-result sentence/token attribution of the similarity score via deterministic occlusion (model-agnostic, no model internals needed).
- **Lexical vs dense split** — for hybrid retrievers, decompose the score into BM25 and dense contributions and report which dominated.
- **Failure taxonomy** — classify a failed `(query, expected_chunk)` into `missing_from_index` / `lost_to_chunking` / `out_ranked` / `embedding_blind_spot`, with explicit evidence and an unambiguous, documented decision order.
- **Counterfactual minimal fix** — search the smallest single config change (top-K, chunk size, hybrid alpha, reranker on/off) that surfaces the right chunk, with a documented per-axis cost.
- **Offline by default** — a deterministic hashing embedder (`FakeEmbedder`) makes the whole pipeline reproducible with zero downloads; real local embeddings and rerankers are opt-in extras.
- **CLI + tiny web inspector** — a rich terminal UI (plus Markdown/JSON output) and an optional read-only FastAPI view.

## How it works

Given a query and (optionally) the chunk you expected to win, it attributes the score by re-scoring the chunk with each sentence occluded, then — if the chunk failed — classifies the cause and searches a bounded set of single-axis config changes for the cheapest one that pulls the right chunk into the top-K. *Cause* (why it ranks where it does) and *fixability* (the cheapest change that surfaces it) are reported as separate fields, so a cause label never reads as "unfixable".

## Installation

Requires Python 3.11+.

```bash
pip install why-this-chunk            # core: numpy, rank_bm25, typer, rich, pydantic

# optional extras
pip install "why-this-chunk[st]"      # real local embeddings + cross-encoder rerank (sentence-transformers)
pip install "why-this-chunk[faiss]"   # FAISS dense backend (faiss-cpu)
pip install "why-this-chunk[web]"     # read-only FastAPI inspector
```

From a checkout, for development:

```bash
pip install -e ".[dev]"
```

## Quickstart

### CLI

The corpus is a JSON-Lines file with one `{"id": ..., "text": ...}` object per line (optional `metadata`, `source_document_id`, `span`). A sample lives in [`examples/corpus.jsonl`](examples/corpus.jsonl).

```bash
# Explain why each top-K result ranked where it did (sentence attribution + split).
why-this-chunk explain "What is the capital of France?" --corpus examples/corpus.jsonl --k 3

# Diagnose why an expected chunk failed, and print the minimal fix.
why-this-chunk diagnose "NumPy fast numerical arrays for Python data science" \
    --expect python --corpus examples/corpus.jsonl --k 1

# Just the fix suggestion(s).
why-this-chunk fix "NumPy fast numerical arrays for Python data science" \
    --expect python --corpus examples/corpus.jsonl --k 1 --all

# Diagnose a whole file of (query, expected chunk) pairs and aggregate the results.
why-this-chunk batch --queries examples/queries.jsonl --corpus examples/corpus.jsonl --k 1
```

`batch` reads a queries JSON-Lines file with one `{"query": ..., "expect": ...}` object per line (a sample lives in [`examples/queries.jsonl`](examples/queries.jsonl)), runs the same diagnose + fix path over each row, and prints an aggregate: the count per failure class, the most common suggested fix axis, and a per-query table.

Every command takes `--format {rich,md,json}` (default `rich`) to choose the output shape; `--json` is kept as a deprecated alias for `--format json`. Pick the retriever with `--mode {bm25,dense,hybrid}` (default `hybrid`). All CLI commands run offline using the deterministic `FakeEmbedder`.

### Library

```python
from why_this_chunk import (
    Corpus, FakeEmbedder, BM25Retriever, DenseRetriever, HybridRetriever,
    RetrievalConfig, explain_chunk, diagnose, search_fixes,
)

corpus = Corpus.from_jsonl("examples/corpus.jsonl")
embedder = FakeEmbedder(seed=0)               # deterministic, offline
retriever = HybridRetriever(
    DenseRetriever(corpus, embedder),
    BM25Retriever(corpus),
    alpha=0.5,
)

# Explain the top result.
top = retriever.search("What is the capital of France?", k=1)[0]
explanation = explain_chunk(retriever, "What is the capital of France?", top)
print(explanation.sentences[0].sentence, explanation.sentences[0].share)
print(explanation.split.dominant)            # "dense" or "lexical"

# Diagnose a failure and find the cheapest fix.
config = RetrievalConfig(top_k=1, alpha=0.5)
query = "NumPy fast numerical arrays for Python data science"
result = diagnose(retriever, query, "python", config)
print(result.failure_class)                  # e.g. FailureClass.OUT_RANKED
print(search_fixes(retriever, query, "python", config).best)   # e.g. raise top_k to 2
```

To make the `chunk_size` axis and `lost_to_chunking` check evaluable, build the corpus with provenance from raw documents:

```python
from why_this_chunk import Corpus, SourceDocument, FixedSizeChunker

corpus = Corpus.from_sources(
    [SourceDocument(id="doc1", text="...full document text...")],
    FixedSizeChunker(overlap=0),
    chunk_size=512,
)
```

### Example demo

A runnable, zero-download demo of `explain` and `diagnose` lives in [`examples/demo.py`](examples/demo.py):

```bash
python examples/demo.py
```

### Web inspector (optional)

```bash
pip install "why-this-chunk[web]"
why-this-chunk serve --corpus examples/corpus.jsonl
# open http://127.0.0.1:8000
```

## Bring your own retriever

A third-party retriever only needs to implement `search(query, k) -> list[ScoredChunk]`. Richer behaviour is advertised through two boolean capabilities — `supports_components` (the dense/lexical split) and `supports_reindex` (returning a new retriever under a different `RetrievalConfig`). Any feature or counterfactual axis that depends on a missing capability or on corpus provenance is reported **unevaluable** rather than silently skipped.

## Determinism

The default `FakeEmbedder` produces stable vectors from token hashes, so retrieval, attribution, and the counterfactual search are fully reproducible with no network access. All tie-breaks use stable id ordering. The real `SentenceTransformerEmbedder` and cross-encoder reranker are opt-in via the `[st]` extra.

## Development

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src
pytest -q --cov=why_this_chunk
```

## Status & roadmap

Beta. The core explainer, taxonomy, counterfactual search, CLI, and a minimal web inspector are implemented and tested offline.

- [x] Score attribution + lexical/dense split for a hybrid retriever
- [x] Per-query failure taxonomy classifier
- [x] Counterfactual minimal-config-fix search
- [x] CLI (`explain` / `diagnose` / `fix` / `batch`) with rich / Markdown / JSON output
- [x] Optional local embeddings, cross-encoder rerank, FAISS backend, read-only web inspector
- [ ] pgvector / Qdrant adapters

## License

[MIT](LICENSE) © 2026 Mykola Podpriatov
