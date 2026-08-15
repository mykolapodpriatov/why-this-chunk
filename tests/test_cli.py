"""CLI happy-path tests via Typer's runner — fully offline (FakeEmbedder)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import typer.main
from typer.testing import CliRunner

from why_this_chunk.cli import app

runner = CliRunner()

#: The bundled example corpus/queries, used to pin deterministic batch aggregates.
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

#: Whether the optional 'st' extra (sentence-transformers) is installed; the
#: real --embedder st / --rerank paths only run when it is, mirroring
#: tests/test_optional.py's pattern so the suite stays green on a minimal
#: install (which is what CI installs).
_HAS_ST = importlib.util.find_spec("sentence_transformers") is not None

#: Same skip pattern for the optional FAISS dense backend (--faiss).
_HAS_FAISS = importlib.util.find_spec("faiss") is not None


@pytest.fixture
def corpus_file(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    lines = [
        {"id": "paris", "text": "The capital of France is Paris. It lies on the Seine."},
        {"id": "eiffel", "text": "The Eiffel Tower is an iron landmark in Paris, France."},
        {"id": "python", "text": "Python is a programming language for data science work."},
        {"id": "banana", "text": "Bananas are a yellow fruit rich in dietary potassium."},
        {"id": "seine", "text": "The Seine is a river flowing through northern France."},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    return path


@pytest.fixture
def corpus_with_meta(tmp_path: Path) -> Path:
    """Corpus whose metadata values and source_document_id are unique locators."""
    path = tmp_path / "corpus_meta.jsonl"
    lines = [
        {
            "id": "c-paris",
            "text": "The capital of France is Paris.",
            "metadata": {"doc": "wiki-paris", "topic": "geo"},
            "source_document_id": "src-paris",
        },
        {
            "id": "c-eiffel",
            "text": "The Eiffel Tower is an iron landmark.",
            "metadata": {"doc": "wiki-eiffel", "topic": "landmark"},
            "source_document_id": "src-eiffel",
        },
        {
            "id": "c-python",
            "text": "Python is a programming language.",
            "metadata": {"doc": "wiki-python", "topic": "tech"},
            "source_document_id": "src-python",
        },
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    return path


@pytest.fixture
def sources_file(tmp_path: Path) -> Path:
    """A raw ``{id, text}`` document long enough to split under small chunk sizes.

    At ``--chunk-size 128`` the query terms land in different windows, but a
    single ``256``-char window keeps them together — so the ``chunk_size`` axis
    yields a real, evaluable fix (unlike a pre-chunked corpus, which cannot be
    re-chunked and reports the axis unevaluable).
    """
    path = tmp_path / "sources.jsonl"
    line = {
        "id": "doc1",
        "text": (
            "The capital of France is Paris and it is famous. "
            "Many unrelated filler sentences about weather and food follow here now. "
            "Bananas potassium tropical fruit yellow elongated edible sweet ripe soft. "
            "More filler about programming languages and databases and networking too. "
            "Finally the Seine river flows through the northern part of the country here."
        ),
    }
    path.write_text(json.dumps(line), encoding="utf-8")
    return path


def test_explain_rich_output(corpus_file: Path) -> None:
    result = runner.invoke(
        app, ["explain", "capital of France Paris", "--corpus", str(corpus_file), "--k", "2"]
    )
    assert result.exit_code == 0, result.output
    assert "explain" in result.output
    assert "attribution" in result.output


def test_explain_json_schema(corpus_file: Path) -> None:
    # The hidden --json alias must keep working for back-compat.
    result = runner.invoke(
        app,
        ["explain", "capital of France Paris", "--corpus", str(corpus_file), "--k", "2", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["query"] == "capital of France Paris"
    assert isinstance(payload["explanations"], list)
    first = payload["explanations"][0]
    assert {"query", "granularity", "degenerate", "result", "split", "sentences"} <= first.keys()
    assert {"chunk_id", "score", "rank"} <= first["result"].keys()


def test_explain_format_json(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "explain",
            "capital of France Paris",
            "--corpus",
            str(corpus_file),
            "--k",
            "2",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["query"] == "capital of France Paris"
    assert isinstance(payload["explanations"], list)


def test_explain_json_alias_matches_format_json(corpus_file: Path) -> None:
    args = ["explain", "Paris France", "--corpus", str(corpus_file), "--k", "2"]
    alias = runner.invoke(app, [*args, "--json"])
    explicit = runner.invoke(app, [*args, "--format", "json"])
    assert alias.exit_code == 0, alias.output
    assert explicit.exit_code == 0, explicit.output
    assert json.loads(alias.output) == json.loads(explicit.output)


def test_explain_markdown(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        ["explain", "Paris France", "--corpus", str(corpus_file), "--format", "md", "--k", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "## explain" in result.output
    assert "| share |" in result.output


def test_explain_token_granularity(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "explain",
            "Paris France",
            "--corpus",
            str(corpus_file),
            "--granularity",
            "token",
            "--k",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "token attribution" in result.output


def test_explain_bad_granularity(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        ["explain", "Paris", "--corpus", str(corpus_file), "--granularity", "phrase"],
    )
    # Typer validates the enum itself and exits 2, listing the valid choices.
    assert result.exit_code == 2
    assert "sentence" in result.output
    assert "token" in result.output


def test_granularity_choices_in_help() -> None:
    result = runner.invoke(app, ["explain", "--help"])
    assert result.exit_code == 0
    assert "sentence" in result.output
    assert "token" in result.output


def test_explain_missing_corpus() -> None:
    result = runner.invoke(app, ["explain", "Paris", "--corpus", "/no/such/file.jsonl"])
    assert result.exit_code == 2
    assert "not found" in result.output


def test_diagnose_missing_chunk(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        ["diagnose", "Paris", "--expect", "nonexistent", "--corpus", str(corpus_file)],
    )
    assert result.exit_code == 0, result.output
    assert "missing_from_index" in result.output


def test_diagnose_json(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "diagnose",
            "Paris France",
            "--expect",
            "seine",
            "--corpus",
            str(corpus_file),
            "--k",
            "1",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {"failure_class", "unevaluable", "evidence", "fix"} <= payload.keys()


def test_diagnose_format_json(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "diagnose",
            "Paris France",
            "--expect",
            "seine",
            "--corpus",
            str(corpus_file),
            "--k",
            "1",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {"failure_class", "unevaluable", "evidence", "fix"} <= payload.keys()


def test_diagnose_markdown(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "diagnose",
            "Paris France",
            "--expect",
            "seine",
            "--corpus",
            str(corpus_file),
            "--format",
            "md",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "## diagnose" in result.output


def test_fix_command(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        ["fix", "river northern France Seine", "--expect", "seine", "--corpus", str(corpus_file)],
    )
    assert result.exit_code == 0, result.output


def test_fix_json(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "fix",
            "Paris France",
            "--expect",
            "seine",
            "--corpus",
            str(corpus_file),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {"best", "all_fixes", "unevaluable"} <= payload.keys()


def test_fix_format_json(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "fix",
            "Paris France",
            "--expect",
            "seine",
            "--corpus",
            str(corpus_file),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {"best", "all_fixes", "unevaluable"} <= payload.keys()


def test_fix_markdown(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "fix",
            "Paris France",
            "--expect",
            "seine",
            "--corpus",
            str(corpus_file),
            "--format",
            "md",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "## fix" in result.output


def test_fix_all_flag(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "fix",
            "Paris France",
            "--expect",
            "seine",
            "--corpus",
            str(corpus_file),
            "--all",
        ],
    )
    assert result.exit_code == 0, result.output


def test_explain_bm25_mode(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        ["explain", "Paris France", "--corpus", str(corpus_file), "--mode", "bm25", "--k", "1"],
    )
    assert result.exit_code == 0, result.output


def test_explain_dense_mode(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        ["explain", "Paris France", "--corpus", str(corpus_file), "--mode", "dense", "--k", "1"],
    )
    assert result.exit_code == 0, result.output


def test_batch_aggregate_counts_deterministic() -> None:
    result = runner.invoke(
        app,
        [
            "batch",
            "--queries",
            str(EXAMPLES / "queries.jsonl"),
            "--corpus",
            str(EXAMPLES / "corpus.jsonl"),
            "--k",
            "1",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 8
    assert payload["failure_counts"] == {
        "missing_from_index": 1,
        "out_ranked": 6,
        "none": 1,
    }
    assert payload["fix_axis_counts"] == {"top_k": 5, "alpha": 2}
    assert payload["top_fix_axis"] == "top_k"
    assert len(payload["rows"]) == 8


def test_batch_rich_output() -> None:
    result = runner.invoke(
        app,
        [
            "batch",
            "--queries",
            str(EXAMPLES / "queries.jsonl"),
            "--corpus",
            str(EXAMPLES / "corpus.jsonl"),
            "--k",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "batch" in result.output
    assert "most common fix axis" in result.output


def test_batch_markdown_output() -> None:
    result = runner.invoke(
        app,
        [
            "batch",
            "--queries",
            str(EXAMPLES / "queries.jsonl"),
            "--corpus",
            str(EXAMPLES / "corpus.jsonl"),
            "--k",
            "1",
            "--format",
            "md",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "## batch" in result.output
    assert "| query | expect | failure | fix |" in result.output


def test_batch_malformed_line_exits_2(tmp_path: Path) -> None:
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"query": "ok", "expect": "paris"}\nnot valid json\n', encoding="utf-8")
    result = runner.invoke(
        app,
        ["batch", "--queries", str(queries), "--corpus", str(EXAMPLES / "corpus.jsonl")],
    )
    assert result.exit_code == 2
    # The error is line-numbered; normalize rich's wrapping before asserting.
    assert ":2:" in "".join(result.output.split())


def test_batch_missing_keys_exits_2(tmp_path: Path) -> None:
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"query": "missing expect key"}\n', encoding="utf-8")
    result = runner.invoke(
        app,
        ["batch", "--queries", str(queries), "--corpus", str(EXAMPLES / "corpus.jsonl")],
    )
    assert result.exit_code == 2
    assert ":1:" in "".join(result.output.split())


def test_batch_empty_file(tmp_path: Path) -> None:
    queries = tmp_path / "queries.jsonl"
    queries.write_text("", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "batch",
            "--queries",
            str(queries),
            "--corpus",
            str(EXAMPLES / "corpus.jsonl"),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 0
    assert payload["rows"] == []
    assert payload["failure_counts"] == {}
    assert payload["top_fix_axis"] is None


def test_batch_fail_on_none_is_default_exit_zero() -> None:
    # Even with failures present, the default gate ('none') never fails CI.
    result = runner.invoke(
        app,
        [
            "batch",
            "--queries",
            str(EXAMPLES / "queries.jsonl"),
            "--corpus",
            str(EXAMPLES / "corpus.jsonl"),
            "--k",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output


def test_batch_fail_on_failure_exits_one() -> None:
    result = runner.invoke(
        app,
        [
            "batch",
            "--queries",
            str(EXAMPLES / "queries.jsonl"),
            "--corpus",
            str(EXAMPLES / "corpus.jsonl"),
            "--k",
            "1",
            "--fail-on",
            "failure",
        ],
    )
    # A regression is present at k=1, so the gate trips — but the report is
    # still emitted before the non-zero exit.
    assert result.exit_code == 1, result.output
    assert "batch" in result.output


def test_batch_fail_on_unfixable_exits_one() -> None:
    # The 'does-not-exist' expectation is a failure with no bounded fix.
    result = runner.invoke(
        app,
        [
            "batch",
            "--queries",
            str(EXAMPLES / "queries.jsonl"),
            "--corpus",
            str(EXAMPLES / "corpus.jsonl"),
            "--k",
            "1",
            "--fail-on",
            "unfixable",
        ],
    )
    assert result.exit_code == 1, result.output


def test_batch_fail_on_failure_clean_run_exits_zero(tmp_path: Path) -> None:
    # With k=8 every existing expected chunk is retrieved, so no row fails.
    queries = tmp_path / "clean.jsonl"
    queries.write_text(
        '{"query": "capital of France", "expect": "paris"}\n'
        '{"query": "Python numerical computing library", "expect": "numpy"}\n',
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "batch",
            "--queries",
            str(queries),
            "--corpus",
            str(EXAMPLES / "corpus.jsonl"),
            "--k",
            "8",
            "--fail-on",
            "failure",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["failure_counts"] == {"none": 2}


def test_batch_missing_queries_file() -> None:
    result = runner.invoke(
        app,
        [
            "batch",
            "--queries",
            "/no/such/queries.jsonl",
            "--corpus",
            str(EXAMPLES / "corpus.jsonl"),
        ],
    )
    assert result.exit_code == 2
    assert "not found" in result.output


def test_validate_clean_corpus(corpus_file: Path) -> None:
    result = runner.invoke(app, ["validate", "--corpus", str(corpus_file)])
    assert result.exit_code == 0, result.output
    assert "no problems found" in result.output


def test_validate_clean_corpus_and_queries() -> None:
    result = runner.invoke(
        app,
        [
            "validate",
            "--corpus",
            str(EXAMPLES / "corpus.jsonl"),
            "--queries",
            str(EXAMPLES / "queries.jsonl"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "no problems found" in result.output


def test_validate_duplicate_id_and_bad_queries_line(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"id": "paris", "text": "The capital of France is Paris."}\n'
        '{"id": "seine", "text": "The Seine is a river in France."}\n'
        '{"id": "paris", "text": "Paris again, a duplicate id."}\n',
        encoding="utf-8",
    )
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        '{"query": "ok", "expect": "paris"}\nnot valid json\n',
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["validate", "--corpus", str(corpus), "--queries", str(queries)],
    )
    assert result.exit_code == 2, result.output
    flat = "".join(result.output.split())
    # The duplicate id is reported at its line (3), naming the first sighting (1).
    assert "duplicateid'paris'" in flat
    assert "firstseenonline1" in flat
    # The malformed queries line is reported with its 1-based line number (2).
    assert ":2:" in flat


def test_validate_reports_malformed_and_blank_lines(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"id": "ok", "text": "fine"}\n'
        "not json at all\n"
        '{"id": "", "text": "blank id here"}\n'
        '{"id": "empty", "text": "   "}\n'
        '{"text": "no id key"}\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate", "--corpus", str(corpus)])
    assert result.exit_code == 2, result.output
    flat = "".join(result.output.split())
    assert ":2:" in flat  # invalid JSON
    assert "blankchunkid" in flat  # line 3
    assert "blanktext" in flat  # line 4
    assert "needs'id'and'text'keys" in flat  # line 5
    assert "4problemsfound" in flat


def test_validate_missing_corpus_file() -> None:
    result = runner.invoke(app, ["validate", "--corpus", "/no/such/corpus.jsonl"])
    assert result.exit_code == 2
    assert "not found" in result.output


_SOURCES_QUERY = "capital France Paris Seine river northern"
_EXPECTED_CHUNK = "doc1::128::0"


def test_explain_from_sources_runs(sources_file: Path) -> None:
    result = runner.invoke(
        app,
        ["explain", "capital France", "--from-sources", str(sources_file), "--k", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "explain" in result.output


def test_diagnose_from_sources_makes_chunk_size_evaluable(sources_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "diagnose",
            _SOURCES_QUERY,
            "--expect",
            _EXPECTED_CHUNK,
            "--from-sources",
            str(sources_file),
            "--chunk-size",
            "128",
            "--mode",
            "bm25",
            "--k",
            "1",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # Provenance from --from-sources makes the chunk-splitting branch evaluable:
    # the expected text is recoverable by a larger window.
    assert payload["failure_class"] == "lost_to_chunking"
    assert "lost_to_chunking" not in payload["unevaluable"]
    # The chunk_size axis is now evaluable, not reported unevaluable as it always
    # was from a pre-chunked corpus.
    assert "chunk_size" not in payload["evidence"]["fix_unevaluable_axes"]


def test_fix_from_sources_finds_chunk_size_fix(sources_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "fix",
            _SOURCES_QUERY,
            "--expect",
            _EXPECTED_CHUNK,
            "--from-sources",
            str(sources_file),
            "--chunk-size",
            "128",
            "--mode",
            "bm25",
            "--k",
            "1",
            "--all",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "chunk_size" not in payload["unevaluable"]
    axes = {fix["param"] for fix in payload["all_fixes"]}
    assert "chunk_size" in axes


def test_batch_from_sources_runs(tmp_path: Path, sources_file: Path) -> None:
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"query": _SOURCES_QUERY, "expect": _EXPECTED_CHUNK}) + "\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "batch",
            "--queries",
            str(queries),
            "--from-sources",
            str(sources_file),
            "--chunk-size",
            "128",
            "--mode",
            "bm25",
            "--k",
            "1",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 1


def test_corpus_and_from_sources_are_mutually_exclusive(
    corpus_file: Path, sources_file: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "diagnose",
            "Paris",
            "--expect",
            "paris",
            "--corpus",
            str(corpus_file),
            "--from-sources",
            str(sources_file),
        ],
    )
    assert result.exit_code == 2
    assert "exactly one of --corpus or --from-sources" in result.output


def test_neither_corpus_nor_from_sources_is_an_error() -> None:
    result = runner.invoke(app, ["explain", "Paris"])
    assert result.exit_code == 2
    assert "exactly one of --corpus or --from-sources" in result.output


def test_from_sources_missing_file_exits_2() -> None:
    result = runner.invoke(app, ["explain", "Paris", "--from-sources", "/no/such/sources.jsonl"])
    assert result.exit_code == 2
    assert "not found" in result.output


def test_from_sources_malformed_line_exits_2(tmp_path: Path) -> None:
    sources = tmp_path / "sources.jsonl"
    sources.write_text('{"id": "doc1", "text": "ok"}\nnot valid json\n', encoding="utf-8")
    result = runner.invoke(app, ["explain", "Paris", "--from-sources", str(sources)])
    assert result.exit_code == 2
    assert ":2:" in "".join(result.output.split())


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    # no_args_is_help => exit code 0 with usage text.
    assert "Usage" in result.output


def test_embedder_and_rerank_flags_are_registered() -> None:
    # Inspect the underlying click command's parameters directly rather than
    # asserting on rendered --help text: rich's help panel wraps/truncates
    # option names under a narrow auto-detected terminal width (this varies by
    # environment, e.g. a headless CI runner vs a local tty), which makes
    # substring-matching the rendered output flaky. Introspection is exact and
    # environment-independent.
    explain_cmd = typer.main.get_command(app).commands["explain"]
    option_flags = {opt for param in explain_cmd.params for opt in getattr(param, "opts", [])}
    assert "--embedder" in option_flags
    assert "--rerank" in option_flags
    assert "--rerank-model" in option_flags
    assert "--faiss" in option_flags


def test_faiss_flag_is_registered_on_every_retriever_command() -> None:
    commands = typer.main.get_command(app).commands
    for name in ("explain", "diagnose", "fix", "batch", "serve"):
        option_flags = {
            opt for param in commands[name].params for opt in getattr(param, "opts", [])
        }
        assert "--faiss" in option_flags, name


def test_embedder_bad_choice_exits_2(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        ["explain", "Paris", "--corpus", str(corpus_file), "--embedder", "bogus"],
    )
    assert result.exit_code == 2
    assert "fake" in result.output
    assert "st" in result.output


def test_embedder_st_bm25_mode_does_not_require_extra(corpus_file: Path) -> None:
    # BM25 never touches the embedder, so --embedder st must not force a
    # sentence-transformers import (let alone a download) in bm25 mode.
    result = runner.invoke(
        app,
        [
            "explain",
            "Paris France",
            "--corpus",
            str(corpus_file),
            "--mode",
            "bm25",
            "--embedder",
            "st",
            "--k",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output


def test_diagnose_rerank_axis_unevaluable_without_flag(corpus_file: Path) -> None:
    # Without --rerank the retriever is never wrapped in RerankingRetriever, so
    # the counterfactual rerank axis stays unevaluable, as it always has.
    result = runner.invoke(
        app,
        [
            "diagnose",
            "Paris France",
            "--expect",
            "seine",
            "--corpus",
            str(corpus_file),
            "--k",
            "1",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "rerank" in payload["evidence"]["fix_unevaluable_axes"]


@pytest.mark.skipif(_HAS_ST, reason="runs only when the st extra is absent")
def test_embedder_st_without_extra_gives_clean_error(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "explain",
            "Paris",
            "--corpus",
            str(corpus_file),
            "--mode",
            "dense",
            "--embedder",
            "st",
        ],
    )
    assert result.exit_code == 2, result.output
    assert "st" in result.output
    assert "pip install" in result.output


@pytest.mark.skipif(_HAS_ST, reason="runs only when the st extra is absent")
def test_rerank_without_extra_gives_clean_error(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        ["explain", "Paris", "--corpus", str(corpus_file), "--rerank"],
    )
    assert result.exit_code == 2, result.output
    assert "st" in result.output
    assert "pip install" in result.output


def test_embedder_st_choice_is_wired_through(
    corpus_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--embedder st reaches _build_embedder with the 'st' choice.

    Verified fully offline (no real sentence-transformers download, and no
    dependency on the [st] extra being installed) by stubbing the internal
    embedder builder and asserting it was invoked with the right choice — the
    thing issue #9 says never happened.
    """
    from why_this_chunk import cli as cli_module

    calls: list[cli_module.EmbedderChoice] = []

    def _fake_build_embedder(
        choice: cli_module.EmbedderChoice, seed: int
    ) -> cli_module.FakeEmbedder:
        calls.append(choice)
        return cli_module.FakeEmbedder(seed=seed)

    monkeypatch.setattr(cli_module, "_build_embedder", _fake_build_embedder)

    result = runner.invoke(
        app,
        [
            "explain",
            "Paris France",
            "--corpus",
            str(corpus_file),
            "--mode",
            "dense",
            "--embedder",
            "st",
            "--k",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == [cli_module.EmbedderChoice.ST]


def test_rerank_flag_wires_reranking_retriever(
    corpus_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--rerank wraps the retriever in RerankingRetriever end-to-end.

    Verified with a fake reranker (stubbing the internal reranker builder) so
    the test stays offline and deterministic while still exercising the real
    RetrievalConfig(rerank=True) + RerankingRetriever wiring that makes the
    counterfactual rerank axis evaluable through the CLI.
    """
    from why_this_chunk import cli as cli_module

    class _FakeReranker:
        def score(self, query: str, texts: list[str]) -> list[float]:
            return [1.0] * len(texts)

    monkeypatch.setattr(cli_module, "_build_reranker", lambda model_name: _FakeReranker())

    result = runner.invoke(
        app,
        [
            "diagnose",
            "Paris France",
            "--expect",
            "seine",
            "--corpus",
            str(corpus_file),
            "--k",
            "1",
            "--mode",
            "bm25",
            "--rerank",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "rerank" not in payload["evidence"].get("fix_unevaluable_axes", [])


@pytest.mark.skipif(_HAS_FAISS, reason="runs only when the faiss extra is absent")
def test_faiss_without_extra_gives_clean_error(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "explain",
            "Paris",
            "--corpus",
            str(corpus_file),
            "--mode",
            "dense",
            "--faiss",
        ],
    )
    assert result.exit_code == 2, result.output
    assert "faiss" in result.output
    assert "pip install" in result.output


def test_faiss_bm25_mode_does_not_require_extra(corpus_file: Path) -> None:
    # BM25 never builds a dense index, so --faiss must not force a faiss import
    # (same contract as --embedder st in bm25 mode).
    result = runner.invoke(
        app,
        [
            "explain",
            "Paris France",
            "--corpus",
            str(corpus_file),
            "--mode",
            "bm25",
            "--faiss",
            "--k",
            "1",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["backend"] is None


def test_dense_without_faiss_flag_surfaces_numpy_backend(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "explain",
            "Paris France",
            "--corpus",
            str(corpus_file),
            "--mode",
            "dense",
            "--k",
            "1",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["backend"] == "numpy"


@pytest.mark.skipif(not _HAS_FAISS, reason="faiss extra not installed")
def test_faiss_flag_surfaces_faiss_backend_dense(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "explain",
            "Paris France",
            "--corpus",
            str(corpus_file),
            "--mode",
            "dense",
            "--faiss",
            "--k",
            "1",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["backend"] == "faiss"
    assert payload["explanations"]


@pytest.mark.skipif(not _HAS_FAISS, reason="faiss extra not installed")
def test_faiss_flag_surfaces_faiss_backend_hybrid(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "explain",
            "Paris France",
            "--corpus",
            str(corpus_file),
            "--mode",
            "hybrid",
            "--faiss",
            "--k",
            "1",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["backend"] == "faiss"


@pytest.mark.skipif(not _HAS_FAISS, reason="faiss extra not installed")
def test_faiss_diagnose_fix_batch_surface_backend(corpus_file: Path, tmp_path: Path) -> None:
    diagnose = runner.invoke(
        app,
        [
            "diagnose",
            "Paris France",
            "--expect",
            "seine",
            "--corpus",
            str(corpus_file),
            "--mode",
            "dense",
            "--faiss",
            "--k",
            "1",
            "--format",
            "json",
        ],
    )
    assert diagnose.exit_code == 0, diagnose.output
    assert json.loads(diagnose.output)["backend"] == "faiss"

    fix = runner.invoke(
        app,
        [
            "fix",
            "Paris France",
            "--expect",
            "seine",
            "--corpus",
            str(corpus_file),
            "--mode",
            "dense",
            "--faiss",
            "--format",
            "json",
        ],
    )
    assert fix.exit_code == 0, fix.output
    assert json.loads(fix.output)["backend"] == "faiss"

    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"query": "Paris France", "expect": "seine"}\n', encoding="utf-8")
    batch = runner.invoke(
        app,
        [
            "batch",
            "--queries",
            str(queries),
            "--corpus",
            str(corpus_file),
            "--mode",
            "dense",
            "--faiss",
            "--k",
            "1",
            "--format",
            "json",
        ],
    )
    assert batch.exit_code == 0, batch.output
    assert json.loads(batch.output)["backend"] == "faiss"


@pytest.mark.skipif(not _HAS_FAISS, reason="faiss extra not installed")
def test_faiss_markdown_mentions_backend(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "explain",
            "Paris France",
            "--corpus",
            str(corpus_file),
            "--mode",
            "dense",
            "--faiss",
            "--k",
            "1",
            "--format",
            "md",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "- backend: faiss" in result.output


def test_expect_exact_id_still_works(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "diagnose",
            "Paris France",
            "--expect",
            "seine",
            "--corpus",
            str(corpus_file),
            "--k",
            "1",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["failure_class"] != "missing_from_index"


def test_expect_unique_substring_resolves(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "diagnose",
            "yellow fruit",
            "--expect",
            "dietary potassium",
            "--corpus",
            str(corpus_file),
            "--k",
            "5",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # Resolved to banana; with k=5 the chunk is retrieved (not missing).
    assert payload["failure_class"] != "missing_from_index"


def test_expect_unique_substring_is_case_insensitive(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "diagnose",
            "yellow fruit",
            "--expect",
            "DIETARY POTASSIUM",
            "--corpus",
            str(corpus_file),
            "--k",
            "5",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["failure_class"] != "missing_from_index"


def test_expect_ambiguous_substring_exits_nonzero(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        ["diagnose", "Paris", "--expect", "France", "--corpus", str(corpus_file)],
    )
    assert result.exit_code == 2, result.output
    assert "matches" in result.output
    assert "paris" in result.output
    assert "eiffel" in result.output


def test_expect_missing_id_still_missing_from_index(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "diagnose",
            "Paris",
            "--expect",
            "nonexistent",
            "--corpus",
            str(corpus_file),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["failure_class"] == "missing_from_index"


def test_expect_unique_metadata_value_resolves(corpus_with_meta: Path) -> None:
    result = runner.invoke(
        app,
        [
            "diagnose",
            "Eiffel Tower",
            "--expect",
            "wiki-eiffel",
            "--corpus",
            str(corpus_with_meta),
            "--k",
            "3",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["failure_class"] != "missing_from_index"


def test_expect_unique_source_document_id_resolves(corpus_with_meta: Path) -> None:
    result = runner.invoke(
        app,
        [
            "diagnose",
            "capital France",
            "--expect",
            "src-paris",
            "--corpus",
            str(corpus_with_meta),
            "--k",
            "3",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["failure_class"] != "missing_from_index"


def test_expect_ambiguous_metadata_exits_nonzero(tmp_path: Path) -> None:
    path = tmp_path / "dup.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "text": "alpha", "metadata": {"topic": "shared"}}),
                json.dumps({"id": "b", "text": "beta", "metadata": {"topic": "shared"}}),
            ]
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["diagnose", "alpha", "--expect", "shared", "--corpus", str(path)],
    )
    assert result.exit_code == 2, result.output
    assert "metadata" in result.output
    assert "a" in result.output
    assert "b" in result.output


def test_batch_expect_text_resolves(tmp_path: Path, corpus_file: Path) -> None:
    queries = tmp_path / "q.jsonl"
    queries.write_text(
        '{"query": "yellow fruit", "expect_text": "dietary potassium"}\n',
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "batch",
            "--queries",
            str(queries),
            "--corpus",
            str(corpus_file),
            "--k",
            "5",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["rows"][0]["expect"] == "banana"
    assert payload["rows"][0]["failure_class"] is None


def test_batch_expect_meta_resolves(tmp_path: Path, corpus_with_meta: Path) -> None:
    queries = tmp_path / "q.jsonl"
    queries.write_text(
        '{"query": "Eiffel Tower", "expect_meta": "wiki-eiffel"}\n',
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "batch",
            "--queries",
            str(queries),
            "--corpus",
            str(corpus_with_meta),
            "--k",
            "3",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["rows"][0]["expect"] == "c-eiffel"


def test_batch_expect_text_zero_matches_exits_nonzero(tmp_path: Path, corpus_file: Path) -> None:
    queries = tmp_path / "q.jsonl"
    queries.write_text(
        '{"query": "Paris", "expect_text": "no-such-phrase-in-corpus"}\n',
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["batch", "--queries", str(queries), "--corpus", str(corpus_file)],
    )
    assert result.exit_code == 2, result.output
    assert "matched 0 chunks" in result.output
    assert "missing_from_index" not in result.output


def test_fix_expect_unique_substring_resolves(corpus_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "fix",
            "yellow fruit",
            "--expect",
            "dietary potassium",
            "--corpus",
            str(corpus_file),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "best" in payload
