"""Typer command-line interface: ``explain``, ``diagnose``, ``fix``, ``serve``.

All commands build a retriever from a JSON-Lines corpus using the offline
:class:`~why_this_chunk.embedders.fake.FakeEmbedder` by default, so the CLI runs
with zero downloads. A single ``--format {rich,md,json}`` switch selects the
output shape (rich terminal view by default, Markdown, or machine-readable
JSON). ``--json`` is retained as a hidden, deprecated alias for ``--format
json``.
"""

from __future__ import annotations

import json as json_module
import sys
from enum import StrEnum
from pathlib import Path

import typer
from rich.console import Console

from why_this_chunk.attribution import explain_chunk
from why_this_chunk.batch import (
    BatchQuery,
    BatchResult,
    has_failures,
    has_unfixable,
    load_queries,
    run_batch,
)
from why_this_chunk.config import RetrievalConfig
from why_this_chunk.corpus import Corpus
from why_this_chunk.counterfactual import search_fixes
from why_this_chunk.embedders import FakeEmbedder
from why_this_chunk.report import (
    batch_to_dict,
    batch_to_markdown,
    diagnosis_to_dict,
    diagnosis_to_markdown,
    explanation_to_dict,
    explanation_to_markdown,
    fixes_to_dict,
    fixes_to_markdown,
    render_batch,
    render_diagnosis,
    render_explanation,
    render_fixes,
)
from why_this_chunk.retrievers import Retriever
from why_this_chunk.retrievers.bm25 import BM25Retriever
from why_this_chunk.retrievers.dense import DenseRetriever
from why_this_chunk.retrievers.hybrid import HybridRetriever
from why_this_chunk.taxonomy import diagnose as run_diagnose
from why_this_chunk.types import DiagnosisResult

app = typer.Typer(
    help="Explain why a chunk ranked where it did, diagnose failures, and find "
    "the one config change that surfaces the right answer.",
    no_args_is_help=True,
    add_completion=False,
)

_console = Console()
_err_console = Console(stderr=True)


class Mode(StrEnum):
    """Retriever modes selectable on the CLI."""

    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


class Granularity(StrEnum):
    """Attribution unit selectable on the CLI (typer validates the choices)."""

    SENTENCE = "sentence"
    TOKEN = "token"


class OutputFormat(StrEnum):
    """Output shapes selectable via ``--format``."""

    RICH = "rich"
    MD = "md"
    JSON = "json"


class FailOn(StrEnum):
    """CI-gate thresholds for ``batch``'s exit code.

    ``none`` never fails; ``failure`` fails on any classified failure; and
    ``unfixable`` fails only on failures with no bounded single-axis fix.
    """

    NONE = "none"
    FAILURE = "failure"
    UNFIXABLE = "unfixable"


#: Exit code raised when a ``batch`` CI gate trips, distinct from the ``2`` used
#: for usage/IO errors so pipelines can tell a regression from a bad invocation.
_GATE_EXIT_CODE = 1


def _gate_tripped(result: BatchResult, fail_on: FailOn) -> bool:
    """Whether the ``--fail-on`` threshold is met by ``result``."""
    if fail_on is FailOn.FAILURE:
        return has_failures(result)
    if fail_on is FailOn.UNFIXABLE:
        return has_unfixable(result)
    return False


def _resolve_format(output_format: OutputFormat, as_json: bool) -> OutputFormat:
    """Fold the hidden ``--json`` back-compat alias onto ``--format json``.

    ``--json`` wins when set, preserving the historical precedence where the
    boolean flag took priority over ``--format``.
    """
    return OutputFormat.JSON if as_json else output_format


def _load_corpus(corpus_path: Path) -> Corpus:
    if not corpus_path.is_file():
        _err_console.print(f"[red]error:[/red] corpus file not found: {corpus_path}")
        raise typer.Exit(code=2)
    try:
        return Corpus.from_jsonl(corpus_path)
    except ValueError as exc:
        _err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def _load_queries(queries_path: Path) -> list[BatchQuery]:
    if not queries_path.is_file():
        _err_console.print(f"[red]error:[/red] queries file not found: {queries_path}")
        raise typer.Exit(code=2)
    try:
        return load_queries(queries_path)
    except ValueError as exc:
        _err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def _build_retriever(corpus: Corpus, mode: Mode, alpha: float, seed: int) -> Retriever:
    if mode is Mode.BM25:
        return BM25Retriever(corpus)
    embedder = FakeEmbedder(seed=seed)
    if mode is Mode.DENSE:
        return DenseRetriever(corpus, embedder)
    dense = DenseRetriever(corpus, embedder)
    lexical = BM25Retriever(corpus)
    return HybridRetriever(dense, lexical, alpha=alpha)


@app.command()
def explain(
    query: str = typer.Argument(..., help="The query to explain."),
    corpus: Path = typer.Option(..., "--corpus", help="Path to a JSON-Lines corpus."),
    k: int = typer.Option(5, "--k", min=1, help="Number of results to explain."),
    granularity: Granularity = typer.Option(
        Granularity.SENTENCE, "--granularity", help="Attribution unit."
    ),
    mode: Mode = typer.Option(Mode.HYBRID, "--mode", help="Retriever mode."),
    alpha: float = typer.Option(0.5, "--alpha", min=0.0, max=1.0, help="Hybrid alpha."),
    seed: int = typer.Option(0, "--seed", help="FakeEmbedder seed (determinism)."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.RICH, "--format", help="Output shape: rich, md, or json."
    ),
    as_json: bool = typer.Option(
        False, "--json", hidden=True, help="Deprecated alias for --format json."
    ),
) -> None:
    """Attribute each top-K result's score to its sentences (and the hybrid split)."""
    fmt = _resolve_format(output_format, as_json)
    loaded = _load_corpus(corpus)
    retriever = _build_retriever(loaded, mode, alpha, seed)
    results = retriever.search(query, k)
    explanations = [
        explain_chunk(retriever, query, result, granularity.value) for result in results
    ]

    if fmt is OutputFormat.JSON:
        payload = {
            "query": query,
            "explanations": [explanation_to_dict(e) for e in explanations],
        }
        _console.print_json(json_module.dumps(payload))
        return
    if fmt is OutputFormat.MD:
        for explanation in explanations:
            sys.stdout.write(explanation_to_markdown(explanation))
            sys.stdout.write("\n")
        return
    if not explanations:
        _console.print("[yellow]no results for this query[/yellow]")
    for explanation in explanations:
        render_explanation(explanation, _console)


def _diagnose_with_fix(
    retriever: Retriever, query: str, expect: str, config: RetrievalConfig
) -> DiagnosisResult:
    result = run_diagnose(retriever, query, expect, config)
    fixes = search_fixes(retriever, query, expect, config)
    evidence = dict(result.evidence)
    if fixes.unevaluable:
        evidence["fix_unevaluable_axes"] = fixes.unevaluable
    return DiagnosisResult(
        failure_class=result.failure_class,
        unevaluable=result.unevaluable,
        evidence=evidence,
        fix=fixes.best,
    )


@app.command()
def diagnose(
    query: str = typer.Argument(..., help="The failing query."),
    expect: str = typer.Option(..., "--expect", help="Id of the known-correct chunk."),
    corpus: Path = typer.Option(..., "--corpus", help="Path to a JSON-Lines corpus."),
    k: int = typer.Option(5, "--k", min=1, help="Top-K under evaluation."),
    mode: Mode = typer.Option(Mode.HYBRID, "--mode", help="Retriever mode."),
    alpha: float = typer.Option(0.5, "--alpha", min=0.0, max=1.0, help="Hybrid alpha."),
    seed: int = typer.Option(0, "--seed", help="FakeEmbedder seed (determinism)."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.RICH, "--format", help="Output shape: rich, md, or json."
    ),
    as_json: bool = typer.Option(
        False, "--json", hidden=True, help="Deprecated alias for --format json."
    ),
) -> None:
    """Classify why an expected chunk failed and report the minimal fix."""
    fmt = _resolve_format(output_format, as_json)
    loaded = _load_corpus(corpus)
    retriever = _build_retriever(loaded, mode, alpha, seed)
    config = RetrievalConfig(top_k=k, alpha=alpha if mode is Mode.HYBRID else None)
    result = _diagnose_with_fix(retriever, query, expect, config)

    if fmt is OutputFormat.JSON:
        _console.print_json(json_module.dumps(diagnosis_to_dict(result)))
        return
    if fmt is OutputFormat.MD:
        sys.stdout.write(diagnosis_to_markdown(result))
        return
    render_diagnosis(result, _console)


@app.command()
def fix(
    query: str = typer.Argument(..., help="The failing query."),
    expect: str = typer.Option(..., "--expect", help="Id of the known-correct chunk."),
    corpus: Path = typer.Option(..., "--corpus", help="Path to a JSON-Lines corpus."),
    k: int = typer.Option(5, "--k", min=1, help="Top-K under evaluation."),
    mode: Mode = typer.Option(Mode.HYBRID, "--mode", help="Retriever mode."),
    alpha: float = typer.Option(0.5, "--alpha", min=0.0, max=1.0, help="Hybrid alpha."),
    seed: int = typer.Option(0, "--seed", help="FakeEmbedder seed (determinism)."),
    show_all: bool = typer.Option(False, "--all", help="Show every fix, ranked."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.RICH, "--format", help="Output shape: rich, md, or json."
    ),
    as_json: bool = typer.Option(
        False, "--json", hidden=True, help="Deprecated alias for --format json."
    ),
) -> None:
    """Report the single cheapest config change (or all of them with --all)."""
    fmt = _resolve_format(output_format, as_json)
    loaded = _load_corpus(corpus)
    retriever = _build_retriever(loaded, mode, alpha, seed)
    config = RetrievalConfig(top_k=k, alpha=alpha if mode is Mode.HYBRID else None)
    result = search_fixes(retriever, query, expect, config)

    if fmt is OutputFormat.JSON:
        _console.print_json(json_module.dumps(fixes_to_dict(result)))
        return
    if fmt is OutputFormat.MD:
        sys.stdout.write(fixes_to_markdown(result, show_all=show_all))
        return
    render_fixes(result, _console, show_all=show_all)


@app.command()
def batch(
    queries: Path = typer.Option(
        ..., "--queries", help="JSON-Lines file of {'query', 'expect'} rows."
    ),
    corpus: Path = typer.Option(..., "--corpus", help="Path to a JSON-Lines corpus."),
    k: int = typer.Option(5, "--k", min=1, help="Top-K under evaluation."),
    mode: Mode = typer.Option(Mode.HYBRID, "--mode", help="Retriever mode."),
    alpha: float = typer.Option(0.5, "--alpha", min=0.0, max=1.0, help="Hybrid alpha."),
    seed: int = typer.Option(0, "--seed", help="FakeEmbedder seed (determinism)."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.RICH, "--format", help="Output shape: rich, md, or json."
    ),
    fail_on: FailOn = typer.Option(
        FailOn.NONE,
        "--fail-on",
        help="CI gate: exit non-zero on any 'failure' or on an 'unfixable' one.",
    ),
    as_json: bool = typer.Option(
        False, "--json", hidden=True, help="Deprecated alias for --format json."
    ),
) -> None:
    """Diagnose a whole queries file and aggregate the failures and fixes."""
    fmt = _resolve_format(output_format, as_json)
    batch_queries = _load_queries(queries)
    loaded = _load_corpus(corpus)
    retriever = _build_retriever(loaded, mode, alpha, seed)
    config = RetrievalConfig(top_k=k, alpha=alpha if mode is Mode.HYBRID else None)
    result = run_batch(retriever, batch_queries, config)

    if fmt is OutputFormat.JSON:
        _console.print_json(json_module.dumps(batch_to_dict(result)))
    elif fmt is OutputFormat.MD:
        sys.stdout.write(batch_to_markdown(result))
    else:
        render_batch(result, _console)

    # Report the results first, then trip the CI gate so pipelines still see the
    # full diagnosis before the non-zero exit.
    if _gate_tripped(result, fail_on):
        raise typer.Exit(code=_GATE_EXIT_CODE)


@app.command()
def serve(
    corpus: Path = typer.Option(..., "--corpus", help="Path to a JSON-Lines corpus."),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
    mode: Mode = typer.Option(Mode.HYBRID, "--mode", help="Retriever mode."),
    alpha: float = typer.Option(0.5, "--alpha", min=0.0, max=1.0, help="Hybrid alpha."),
    seed: int = typer.Option(0, "--seed", help="FakeEmbedder seed (determinism)."),
) -> None:
    """Launch the optional read-only FastAPI inspector (requires the [web] extra)."""
    try:
        import uvicorn

        from why_this_chunk.web import create_app
    except ImportError as exc:
        _err_console.print(
            "[red]error:[/red] the 'web' extra is required for `serve`. "
            "Install it with: pip install 'why-this-chunk[web]'"
        )
        raise typer.Exit(code=2) from exc

    loaded = _load_corpus(corpus)
    retriever = _build_retriever(loaded, mode, alpha, seed)
    web_app = create_app(retriever)
    uvicorn.run(web_app, host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    app()
