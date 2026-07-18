"""CLI happy-path tests via Typer's runner — fully offline (FakeEmbedder)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from why_this_chunk.cli import app

runner = CliRunner()

#: The bundled example corpus/queries, used to pin deterministic batch aggregates.
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


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
