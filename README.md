# why-this-chunk

> Retrieval explainer for RAG — shows why a chunk ranked where it did, the lexical-vs-dense split, and the one config change that would have surfaced the right answer.

![status](https://img.shields.io/badge/status-early%20development-orange) ![language](https://img.shields.io/badge/language-Python-blue) ![license](https://img.shields.io/badge/license-MIT-green)

Point it at a query and an existing retriever/index; it produces per-result explanations: token/sentence-level attribution of the similarity score and a lexical (BM25) vs dense contribution split for hybrid search.

## Why

RAG debugging today is mostly guesswork. This tells you which layer is actually at fault and the smallest change that fixes a specific failing query.

## Features

- Per-result token/sentence attribution of the similarity score
- Lexical (BM25) vs dense contribution split for hybrid retrievers
- Counterfactual minimal-config-fix search to surface a known-correct chunk
- Per-query failure taxonomy: missing-from-index / lost-to-chunking / out-ranked / embedding-blind-spot
- CLI + tiny local web inspector over an existing index; local or cloud embeddings/rerankers

## How it works

Given a query and (optionally) the chunk you expected to win, it attributes the score, then searches the smallest single config change (top-K, chunk size, hybrid alpha, reranker on/off) that would have pulled the right chunk into results.

## Tech stack

- Python
- sentence-transformers
- rank_bm25
- cross-encoder rerankers
- FAISS / Chroma
- FastAPI

## Status & roadmap

🚧 **Early development.** This repository is being built in the open; the scaffold and design are in place and the implementation is landing incrementally.

- [ ] Score attribution + lexical/dense split for a hybrid retriever
- [ ] Per-query failure taxonomy classifier
- [ ] Counterfactual minimal-config-fix search
- [ ] pgvector/Qdrant adapters; Markdown root-cause export

## Installation

> Coming soon.

## License

[MIT](LICENSE) © 2026 Mykola Podpriatov
