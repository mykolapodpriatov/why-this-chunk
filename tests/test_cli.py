"""CLI happy-path tests via Typer's runner — fully offline (FakeEmbedder)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from why_this_chunk.cli import app

runner = CliRunner()


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


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    # no_args_is_help => exit code 0 with usage text.
    assert "Usage" in result.output
