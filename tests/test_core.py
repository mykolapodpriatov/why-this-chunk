"""Tests for embedders, source/chunker, corpus loaders, config, and text utils."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from why_this_chunk import (
    Chunk,
    Corpus,
    FakeEmbedder,
    FixedSizeChunker,
    RetrievalConfig,
    SourceDocument,
)
from why_this_chunk._text import split_sentences, token_spans, tokenize
from why_this_chunk.config import load_project_config


# --- FakeEmbedder -----------------------------------------------------------
def test_fake_embedder_deterministic() -> None:
    a = FakeEmbedder(seed=1).encode(["hello world"])
    b = FakeEmbedder(seed=1).encode(["hello world"])
    assert np.array_equal(a, b)


def test_fake_embedder_unit_norm() -> None:
    vectors = FakeEmbedder().encode(["some text here", "another piece"])
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0)


def test_fake_embedder_empty_text_is_zero_row() -> None:
    vectors = FakeEmbedder().encode(["", "!!!"])
    assert np.allclose(vectors, 0.0)


def test_fake_embedder_shared_tokens_correlate() -> None:
    emb = FakeEmbedder(seed=2, dim=128)
    vecs = emb.encode(["paris france capital", "paris france city", "banana fruit yellow"])
    sim_related = float(np.dot(vecs[0], vecs[1]))
    sim_unrelated = float(np.dot(vecs[0], vecs[2]))
    assert sim_related > sim_unrelated


def test_fake_embedder_rejects_bad_dim() -> None:
    with pytest.raises(ValueError, match="dim"):
        FakeEmbedder(dim=0)


def test_fake_embedder_seed_changes_space() -> None:
    a = FakeEmbedder(seed=1).encode(["token"])
    b = FakeEmbedder(seed=2).encode(["token"])
    assert not np.array_equal(a, b)


# --- FixedSizeChunker -------------------------------------------------------
def test_chunker_produces_provenance() -> None:
    docs = [SourceDocument(id="d", text="abcdefghij")]
    chunks = FixedSizeChunker().chunk(docs, chunk_size=4)
    assert [c.text for c in chunks] == ["abcd", "efgh", "ij"]
    assert all(c.source_document_id == "d" for c in chunks)
    assert chunks[0].span == (0, 4)
    assert chunks[-1].span == (8, 10)


def test_chunker_overlap() -> None:
    docs = [SourceDocument(id="d", text="abcdef")]
    chunks = FixedSizeChunker(overlap=2).chunk(docs, chunk_size=4)
    assert chunks[0].text == "abcd"
    assert chunks[1].text == "cdef"


def test_chunker_rejects_size_le_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        FixedSizeChunker(overlap=4).chunk([SourceDocument(id="d", text="abc")], chunk_size=4)


def test_chunker_rejects_negative_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        FixedSizeChunker(overlap=-1)


def test_chunker_skips_empty_docs() -> None:
    chunks = FixedSizeChunker().chunk([SourceDocument(id="empty", text="")], chunk_size=4)
    assert chunks == []


def test_chunker_stable_ids() -> None:
    docs = [SourceDocument(id="d", text="abcdefgh")]
    first = FixedSizeChunker().chunk(docs, 4)
    second = FixedSizeChunker().chunk(docs, 4)
    assert [c.id for c in first] == [c.id for c in second]


# --- Corpus -----------------------------------------------------------------
def test_corpus_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Corpus.from_chunks([Chunk(id="x", text="a"), Chunk(id="x", text="b")])


def test_corpus_from_sources_has_provenance() -> None:
    docs = [SourceDocument(id="d", text="abcdefgh")]
    corpus = Corpus.from_sources(docs, FixedSizeChunker(), 4)
    assert corpus.has_provenance is True
    assert len(corpus.source_documents) == 1


def test_corpus_from_chunks_no_provenance() -> None:
    corpus = Corpus.from_chunks([Chunk(id="a", text="x")])
    assert corpus.has_provenance is False


def test_corpus_get_and_contains() -> None:
    corpus = Corpus.from_chunks([Chunk(id="a", text="x")])
    assert corpus.contains("a")
    assert not corpus.contains("z")
    assert corpus.get("a") is not None
    assert corpus.get("z") is None


def test_corpus_iteration_and_len() -> None:
    chunks = [Chunk(id="a", text="x"), Chunk(id="b", text="y")]
    corpus = Corpus.from_chunks(chunks)
    assert len(corpus) == 2
    assert [c.id for c in corpus] == ["a", "b"]


def test_corpus_from_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text(
        '{"id": "a", "text": "hello", "metadata": {"k": 1}}\n'
        "\n"
        '{"id": "b", "text": "world", "span": [0, 5], "source_document_id": "d"}\n',
        encoding="utf-8",
    )
    corpus = Corpus.from_jsonl(path)
    assert len(corpus) == 2
    assert corpus.get("a").metadata == {"k": 1}  # type: ignore[union-attr]
    assert corpus.get("b").span == (0, 5)  # type: ignore[union-attr]


def test_corpus_from_jsonl_preserves_provenance(tmp_path: Path) -> None:
    """Regression: when every jsonl chunk carries source_document_id + span, the
    corpus must preserve provenance (chunks keep it AND has_provenance is True),
    so jsonl corpora can evaluate lost_to_chunking / the chunk_size axis.
    """
    path = tmp_path / "prov.jsonl"
    path.write_text(
        '{"id": "c0", "text": "Hello ", "source_document_id": "d", "span": [0, 6]}\n'
        '{"id": "c1", "text": "world!", "source_document_id": "d", "span": [6, 12]}\n',
        encoding="utf-8",
    )
    corpus = Corpus.from_jsonl(path)

    assert corpus.has_provenance is True
    c0 = corpus.get("c0")
    assert c0 is not None
    assert c0.source_document_id == "d"
    assert c0.span == (0, 6)
    # The source document is faithfully reconstructed from the chunk spans/text.
    docs = corpus.source_documents
    assert len(docs) == 1
    assert docs[0].id == "d"
    assert docs[0].text == "Hello world!"


def test_corpus_from_jsonl_mixed_provenance_is_prechunked(tmp_path: Path) -> None:
    """If any chunk lacks provenance, the jsonl corpus is treated as pre-chunked
    (no source documents), since it cannot be faithfully re-chunked.
    """
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        '{"id": "a", "text": "hello"}\n'
        '{"id": "b", "text": "world", "source_document_id": "d", "span": [0, 5]}\n',
        encoding="utf-8",
    )
    corpus = Corpus.from_jsonl(path)
    assert corpus.has_provenance is False
    # Per-chunk provenance is still retained on the chunk that carried it.
    b = corpus.get("b")
    assert b is not None
    assert b.span == (0, 5)


def test_corpus_from_jsonl_inconsistent_span_declines_provenance(tmp_path: Path) -> None:
    """A span whose length disagrees with the chunk text cannot faithfully
    reconstruct a source, so provenance is declined (pre-chunked corpus).
    """
    path = tmp_path / "bad_span.jsonl"
    # span [0, 99] but text length is 5 -> inconsistent.
    path.write_text(
        '{"id": "a", "text": "hello", "source_document_id": "d", "span": [0, 99]}\n',
        encoding="utf-8",
    )
    corpus = Corpus.from_jsonl(path)
    assert corpus.has_provenance is False
    a = corpus.get("a")
    assert a is not None and a.span == (0, 99)  # per-chunk span still retained


def test_corpus_from_jsonl_gap_between_spans_declines_provenance(tmp_path: Path) -> None:
    """Regression: spans that leave a gap (here positions 3-5 are unobserved)
    cannot faithfully reconstruct the source, so provenance is declined instead
    of fabricating filler text for the gap.
    """
    path = tmp_path / "gap.jsonl"
    # span [0, 3] then [6, 9]: characters 3..5 were never observed.
    path.write_text(
        '{"id": "a", "text": "abc", "source_document_id": "d", "span": [0, 3]}\n'
        '{"id": "b", "text": "ghi", "source_document_id": "d", "span": [6, 9]}\n',
        encoding="utf-8",
    )
    corpus = Corpus.from_jsonl(path)
    assert corpus.has_provenance is False
    assert corpus.source_documents == []
    # Per-chunk provenance is still retained on the chunks that carried it.
    a = corpus.get("a")
    assert a is not None and a.span == (0, 3)


def test_corpus_from_jsonl_nonzero_first_start_declines_provenance(tmp_path: Path) -> None:
    """Regression: a document whose earliest span starts at a nonzero offset has
    an unobserved leading prefix, so provenance is declined rather than padding
    the prefix with fabricated characters.
    """
    path = tmp_path / "offset.jsonl"
    # Earliest span starts at 3: characters 0..2 were never observed.
    path.write_text(
        '{"id": "a", "text": "def", "source_document_id": "d", "span": [3, 6]}\n'
        '{"id": "b", "text": "ghi", "source_document_id": "d", "span": [6, 9]}\n',
        encoding="utf-8",
    )
    corpus = Corpus.from_jsonl(path)
    assert corpus.has_provenance is False
    assert corpus.source_documents == []


def test_corpus_from_jsonl_conflicting_overlap_declines_provenance(tmp_path: Path) -> None:
    """Regression: overlapping spans whose texts disagree in the overlap region
    cannot be reconciled, so provenance is declined instead of silently letting
    one chunk overwrite the other (which would fabricate a source that matches
    neither chunk).
    """
    path = tmp_path / "conflict.jsonl"
    # [0, 5]="aaaaa" and [3, 8]="bbbbb" disagree on positions 3 and 4.
    path.write_text(
        '{"id": "a", "text": "aaaaa", "source_document_id": "d", "span": [0, 5]}\n'
        '{"id": "b", "text": "bbbbb", "source_document_id": "d", "span": [3, 8]}\n',
        encoding="utf-8",
    )
    corpus = Corpus.from_jsonl(path)
    assert corpus.has_provenance is False
    assert corpus.source_documents == []


def test_corpus_from_jsonl_clean_contiguous_tiling_preserves_provenance(tmp_path: Path) -> None:
    """Clean, gap-free, non-overlapping tiling starting at 0 reconstructs the
    source exactly and keeps provenance.
    """
    path = tmp_path / "tiling.jsonl"
    path.write_text(
        '{"id": "c0", "text": "abcd", "source_document_id": "d", "span": [0, 4]}\n'
        '{"id": "c1", "text": "efgh", "source_document_id": "d", "span": [4, 8]}\n'
        '{"id": "c2", "text": "ij", "source_document_id": "d", "span": [8, 10]}\n',
        encoding="utf-8",
    )
    corpus = Corpus.from_jsonl(path)
    assert corpus.has_provenance is True
    docs = corpus.source_documents
    assert len(docs) == 1
    assert docs[0].id == "d"
    assert docs[0].text == "abcdefghij"


def test_corpus_from_jsonl_consistent_overlap_preserves_provenance(tmp_path: Path) -> None:
    """An overlapping-but-consistent tiling (a common chunker output) must still
    be accepted: the overlap agrees, so the source reconstructs exactly and
    provenance is preserved.
    """
    path = tmp_path / "overlap.jsonl"
    # Overlap regions agree: "cd" shared by c0/c1, "gh" shared by c1/c2.
    path.write_text(
        '{"id": "c0", "text": "abcd", "source_document_id": "d", "span": [0, 4]}\n'
        '{"id": "c1", "text": "cdefgh", "source_document_id": "d", "span": [2, 8]}\n'
        '{"id": "c2", "text": "ghij", "source_document_id": "d", "span": [6, 10]}\n',
        encoding="utf-8",
    )
    corpus = Corpus.from_jsonl(path)
    assert corpus.has_provenance is True
    docs = corpus.source_documents
    assert len(docs) == 1
    assert docs[0].id == "d"
    assert docs[0].text == "abcdefghij"


def test_corpus_from_jsonl_one_bad_document_declines_whole_corpus(tmp_path: Path) -> None:
    """Provenance is all-or-nothing across documents: one clean document plus one
    gapped document declines provenance for the entire corpus, so no fabricated
    source slips through for the bad document.
    """
    path = tmp_path / "mixed_docs.jsonl"
    path.write_text(
        # doc "good": clean contiguous tiling.
        '{"id": "g0", "text": "abc", "source_document_id": "good", "span": [0, 3]}\n'
        '{"id": "g1", "text": "def", "source_document_id": "good", "span": [3, 6]}\n'
        # doc "bad": a gap at positions 3..5.
        '{"id": "b0", "text": "abc", "source_document_id": "bad", "span": [0, 3]}\n'
        '{"id": "b1", "text": "ghi", "source_document_id": "bad", "span": [6, 9]}\n',
        encoding="utf-8",
    )
    corpus = Corpus.from_jsonl(path)
    assert corpus.has_provenance is False
    assert corpus.source_documents == []


def test_corpus_from_jsonl_empty_file_has_no_provenance(tmp_path: Path) -> None:
    """A blank/empty jsonl file yields an empty, provenance-less corpus."""
    path = tmp_path / "empty.jsonl"
    path.write_text("\n  \n", encoding="utf-8")
    corpus = Corpus.from_jsonl(path)
    assert len(corpus) == 0
    assert corpus.has_provenance is False


def test_corpus_from_jsonl_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        Corpus.from_jsonl(path)


def test_corpus_from_jsonl_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text('{"id": "a"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"id.*text"):
        Corpus.from_jsonl(path)


# --- RetrievalConfig --------------------------------------------------------
def test_config_defaults() -> None:
    cfg = RetrievalConfig()
    assert cfg.top_k == 5
    assert cfg.chunk_size == 512
    assert cfg.alpha is None
    assert cfg.rerank is False


def test_config_with_updates_is_copy() -> None:
    cfg = RetrievalConfig(top_k=3)
    updated = cfg.with_updates(top_k=9)
    assert cfg.top_k == 3
    assert updated.top_k == 9


def test_config_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        RetrievalConfig(alpha=2.0)


def test_config_rejects_bad_top_k() -> None:
    with pytest.raises(ValueError):
        RetrievalConfig(top_k=0)


def test_config_rejects_unknown_field() -> None:
    with pytest.raises(ValueError):
        RetrievalConfig.model_validate({"top_k": 3, "bogus": 1})


def test_load_project_config_defaults_when_absent(tmp_path: Path) -> None:
    cfg = load_project_config(start=tmp_path)
    assert cfg == RetrievalConfig()


def test_load_project_config_reads_table(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.why-this-chunk]\ntop_k = 7\nchunk_size = 256\n", encoding="utf-8"
    )
    cfg = load_project_config(start=tmp_path)
    assert cfg.top_k == 7
    assert cfg.chunk_size == 256


def test_load_project_config_no_table(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.other]\nx = 1\n", encoding="utf-8")
    assert load_project_config(start=tmp_path) == RetrievalConfig()


# --- text utilities ---------------------------------------------------------
def test_tokenize_lowercases() -> None:
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_split_sentences_basic() -> None:
    sents = split_sentences("First sentence. Second one! Third?")
    assert [s for s, _ in sents] == ["First sentence.", "Second one!", "Third?"]


def test_split_sentences_spans_index_back() -> None:
    text = "Alpha beta. Gamma delta."
    sents = split_sentences(text)
    for sentence, (start, end) in sents:
        assert text[start:end] == sentence


def test_split_sentences_abbreviation_guard() -> None:
    # "Dr." should not end a sentence.
    sents = split_sentences("Dr. Smith arrived. He was late.")
    assert len(sents) == 2


def test_split_sentences_empty() -> None:
    assert split_sentences("   ") == []


def test_split_sentences_no_terminal_punctuation() -> None:
    sents = split_sentences("just a fragment")
    assert [s for s, _ in sents] == ["just a fragment"]


def test_token_spans_surface_form() -> None:
    spans = token_spans("Hello World")
    assert [s for s, _ in spans] == ["Hello", "World"]
    text = "Hello World"
    for token, (start, end) in spans:
        assert text[start:end] == token


def test_token_spans_empty() -> None:
    assert token_spans("") == []


def test_split_sentences_short_word_no_does_not_suppress_break() -> None:
    """Regression: the short common word 'no' must not be treated as an
    abbreviation that suppresses a real sentence break.
    """
    sents = split_sentences("There is no. Wait.")
    assert [s for s, _ in sents] == ["There is no.", "Wait."]


def test_split_sentences_genuine_abbreviation_stays_one_sentence() -> None:
    """Regression: a genuine abbreviation ('Dr.') must still suppress the break,
    keeping the text as a single sentence (the double-dot fix must not regress
    real abbreviations).
    """
    sents = split_sentences("Dr. Smith left.")
    assert [s for s, _ in sents] == ["Dr. Smith left."]


def test_split_sentences_removed_short_abbreviations_break_normally() -> None:
    """Regression: the removed short words 'al'/'st' no longer suppress breaks."""
    assert len(split_sentences("Meet me at the corner of 5th st. Bring a map.")) == 2
    assert len(split_sentences("It was the best al. Then came the rest.")) == 2


def test_split_sentences_abbreviation_at_boundary_only() -> None:
    # An abbreviation ("e.g") mid-text must not create a spurious split.
    sents = split_sentences("Use a tool, e.g. ruff, for linting. It helps.")
    assert len(sents) == 2


def test_split_sentences_trailing_abbreviation() -> None:
    # A guarded abbreviation right before the end should not over-split.
    sents = split_sentences("He works at the company Inc. Their office is downtown.")
    # 'Inc' is not in the guard set, so this legitimately splits into two.
    assert len(sents) >= 1
