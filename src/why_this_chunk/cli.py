"""Typer CLI: ``explain``, ``diagnose``, ``fix``, ``batch``, ``validate``, ``serve``.

The analysis commands build a retriever over either a pre-chunked JSON-Lines
corpus (``--corpus``) or raw ``{id, text}`` source documents chunked on the fly
(``--from-sources`` with ``--chunk-size``/``--overlap``); the latter carries
provenance, so the ``chunk_size`` counterfactual axis and ``lost_to_chunking``
check become evaluable from the command line. Retrieval uses the offline
:class:`~why_this_chunk.embedders.fake.FakeEmbedder` by default, so the CLI runs
with zero downloads; pass ``--embedder st`` to use the real
:class:`~why_this_chunk.embedders.sentence_transformers.SentenceTransformerEmbedder`
and ``--rerank`` (with an optional ``--rerank-model`` override) to wrap the
retriever in a :class:`~why_this_chunk.rerank.RerankingRetriever`, both gated on
the ``[st]`` extra with a clear CLI error when it is missing. Pass ``--faiss``
to use the FAISS dense index (dense and hybrid modes), gated on the ``[faiss]``
extra the same way; the active backend (``faiss`` or ``numpy``) is reported in
rich/md/json output. A single ``--format {rich,md,json}`` switch selects the
output shape (rich terminal view by default, Markdown, or machine-readable
JSON). ``--json`` is retained as a hidden, deprecated alias for ``--format json``.
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
from why_this_chunk.corpus import Corpus, lint_jsonl
from why_this_chunk.counterfactual import search_fixes
from why_this_chunk.embedders import Embedder, FakeEmbedder
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
from why_this_chunk.rerank import CrossEncoderReranker, RerankingRetriever
from why_this_chunk.retrievers import Retriever
from why_this_chunk.retrievers.bm25 import BM25Retriever
from why_this_chunk.retrievers.dense import DenseRetriever
from why_this_chunk.retrievers.hybrid import HybridRetriever
from why_this_chunk.source import Chunker, FixedSizeChunker, SourceDocument
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


class EmbedderChoice(StrEnum):
    """Embedder backends selectable via ``--embedder``.

    ``fake`` (the default) is the deterministic, offline
    :class:`~why_this_chunk.embedders.fake.FakeEmbedder`. ``st`` is the real
    :class:`~why_this_chunk.embedders.sentence_transformers.SentenceTransformerEmbedder`,
    gated on the ``[st]`` extra.
    """

    FAKE = "fake"
    ST = "st"


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


def _load_sources(sources_path: Path) -> list[SourceDocument]:
    """Load ``{id, text}`` raw documents from a JSON-Lines file.

    Mirrors :meth:`Corpus.from_jsonl`'s strict, line-numbered parsing so a
    malformed source file fails with the same precise diagnostic and exit code.
    """
    if not sources_path.is_file():
        _err_console.print(f"[red]error:[/red] sources file not found: {sources_path}")
        raise typer.Exit(code=2)
    docs: list[SourceDocument] = []
    try:
        with sources_path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    record = json_module.loads(stripped)
                except json_module.JSONDecodeError as exc:
                    raise ValueError(
                        f"{sources_path}:{line_number}: invalid JSON ({exc.msg})"
                    ) from exc
                if not isinstance(record, dict) or "id" not in record or "text" not in record:
                    raise ValueError(
                        f"{sources_path}:{line_number}: each line needs 'id' and 'text' keys"
                    )
                docs.append(SourceDocument(id=str(record["id"]), text=str(record["text"])))
    except ValueError as exc:
        _err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    return docs


def _resolve_corpus(
    corpus: Path | None,
    from_sources: Path | None,
    chunk_size: int,
    overlap: int,
) -> tuple[Corpus, Chunker | None]:
    """Build the corpus from exactly one of ``--corpus`` or ``--from-sources``.

    Returns the corpus and, when built from sources, the chunker that makes the
    ``chunk_size`` axis re-chunkable (``None`` for a pre-chunked corpus). The two
    inputs are mutually exclusive; giving both or neither is a usage error.
    """
    if (corpus is None) == (from_sources is None):
        _err_console.print("[red]error:[/red] provide exactly one of --corpus or --from-sources")
        raise typer.Exit(code=2)
    if from_sources is not None:
        docs = _load_sources(from_sources)
        try:
            chunker = FixedSizeChunker(overlap=overlap)
            built = Corpus.from_sources(docs, chunker, chunk_size)
        except ValueError as exc:
            _err_console.print(f"[red]error:[/red] {exc}")
            raise typer.Exit(code=2) from exc
        return built, chunker
    assert corpus is not None  # guaranteed by the exclusivity check above
    return _load_corpus(corpus), None


def _build_embedder(choice: EmbedderChoice, seed: int) -> Embedder:
    """Construct the embedder selected by ``--embedder``.

    Raises:
        typer.Exit: With a clear message (not a raw traceback) when ``st`` is
            selected but the ``[st]`` extra is not installed.
    """
    if choice is EmbedderChoice.FAKE:
        return FakeEmbedder(seed=seed)
    from why_this_chunk.embedders.sentence_transformers import SentenceTransformerEmbedder

    try:
        return SentenceTransformerEmbedder()
    except ImportError as exc:
        _err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def _build_reranker(model_name: str | None) -> CrossEncoderReranker:
    """Construct the cross-encoder reranker used by ``--rerank``.

    Raises:
        typer.Exit: With a clear message (not a raw traceback) when the
            ``[st]`` extra is not installed.
    """
    try:
        if model_name is None:
            return CrossEncoderReranker()
        return CrossEncoderReranker(model_name=model_name)
    except ImportError as exc:
        _err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def _require_faiss() -> None:
    """Fail with a clear CLI error when ``--faiss`` is set but the extra is missing.

    The library's ``DenseRetriever(use_faiss=True)`` silently falls back to numpy
    if FAISS is not installed; the CLI must not, so users who installed the
    extra (or think they did) get a hard error instead of a silent numpy path.
    """
    try:
        import faiss  # noqa: F401
    except ImportError as exc:
        _err_console.print(
            "[red]error:[/red] --faiss requires the optional 'faiss' extra. "
            "Install it with: pip install 'why-this-chunk[faiss]'"
        )
        raise typer.Exit(code=2) from exc


def _backend_of(retriever: Retriever) -> str | None:
    """Return the dense index backend (``faiss`` / ``numpy``) if the retriever has one."""
    value = getattr(retriever, "backend", None)
    return value if isinstance(value, str) else None


def _print_backend(fmt: OutputFormat, retriever: Retriever) -> None:
    """Emit the active backend in rich or Markdown so users can confirm FAISS."""
    backend = _backend_of(retriever)
    if backend is None:
        return
    if fmt is OutputFormat.MD:
        sys.stdout.write(f"- backend: {backend}\n")
        return
    _console.print(f"backend: [cyan]{backend}[/cyan]")


def _build_retriever(
    corpus: Corpus,
    mode: Mode,
    alpha: float,
    seed: int,
    chunker: Chunker | None = None,
    config: RetrievalConfig | None = None,
    embedder_choice: EmbedderChoice = EmbedderChoice.FAKE,
    rerank: bool = False,
    rerank_model: str | None = None,
    use_faiss: bool = False,
) -> Retriever:
    retriever: Retriever
    if mode is Mode.BM25:
        retriever = BM25Retriever(corpus, chunker=chunker, config=config)
    else:
        if use_faiss:
            _require_faiss()
        embedder = _build_embedder(embedder_choice, seed)
        dense = DenseRetriever(
            corpus, embedder, chunker=chunker, config=config, use_faiss=use_faiss
        )
        if mode is Mode.DENSE:
            retriever = dense
        else:
            lexical = BM25Retriever(corpus, chunker=chunker, config=config)
            retriever = HybridRetriever(dense, lexical, alpha=alpha)
    if rerank:
        retriever = RerankingRetriever(retriever, _build_reranker(rerank_model))
    return retriever


@app.command()
def explain(
    query: str = typer.Argument(..., help="The query to explain."),
    corpus: Path | None = typer.Option(
        None, "--corpus", help="Path to a pre-chunked JSON-Lines corpus."
    ),
    from_sources: Path | None = typer.Option(
        None,
        "--from-sources",
        help="Path to a JSON-Lines file of {id, text} raw documents to chunk on the fly.",
    ),
    chunk_size: int = typer.Option(
        512, "--chunk-size", min=1, help="Chunk size (chars) when --from-sources is used."
    ),
    overlap: int = typer.Option(
        0, "--overlap", min=0, help="Inter-chunk overlap (chars) when --from-sources is used."
    ),
    k: int = typer.Option(5, "--k", min=1, help="Number of results to explain."),
    granularity: Granularity = typer.Option(
        Granularity.SENTENCE, "--granularity", help="Attribution unit."
    ),
    mode: Mode = typer.Option(Mode.HYBRID, "--mode", help="Retriever mode."),
    alpha: float = typer.Option(0.5, "--alpha", min=0.0, max=1.0, help="Hybrid alpha."),
    seed: int = typer.Option(0, "--seed", help="FakeEmbedder seed (determinism)."),
    embedder: EmbedderChoice = typer.Option(
        EmbedderChoice.FAKE,
        "--embedder",
        help="Embedder backend: fake (offline) or st (sentence-transformers, the 'st' extra).",
    ),
    rerank: bool = typer.Option(
        False,
        "--rerank",
        help="Rerank the candidate pool with a cross-encoder (requires the 'st' extra).",
    ),
    rerank_model: str | None = typer.Option(
        None, "--rerank-model", help="Cross-encoder model id override for --rerank."
    ),
    use_faiss: bool = typer.Option(
        False,
        "--faiss",
        help="Use the FAISS dense backend (requires the 'faiss' extra).",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.RICH, "--format", help="Output shape: rich, md, or json."
    ),
    as_json: bool = typer.Option(
        False, "--json", hidden=True, help="Deprecated alias for --format json."
    ),
) -> None:
    """Attribute each top-K result's score to its sentences (and the hybrid split)."""
    fmt = _resolve_format(output_format, as_json)
    loaded, chunker = _resolve_corpus(corpus, from_sources, chunk_size, overlap)
    config = RetrievalConfig(
        top_k=k,
        chunk_size=chunk_size,
        alpha=alpha if mode is Mode.HYBRID else None,
        rerank=rerank,
    )
    retriever = _build_retriever(
        loaded,
        mode,
        alpha,
        seed,
        chunker,
        config,
        embedder,
        rerank,
        rerank_model,
        use_faiss,
    )
    results = retriever.search(query, k)
    explanations = [
        explain_chunk(retriever, query, result, granularity.value) for result in results
    ]

    if fmt is OutputFormat.JSON:
        payload = {
            "query": query,
            "backend": _backend_of(retriever),
            "explanations": [explanation_to_dict(e) for e in explanations],
        }
        _console.print_json(json_module.dumps(payload))
        return
    _print_backend(fmt, retriever)
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
    corpus: Path | None = typer.Option(
        None, "--corpus", help="Path to a pre-chunked JSON-Lines corpus."
    ),
    from_sources: Path | None = typer.Option(
        None,
        "--from-sources",
        help="Path to a JSON-Lines file of {id, text} raw documents to chunk on the fly.",
    ),
    chunk_size: int = typer.Option(
        512, "--chunk-size", min=1, help="Chunk size (chars) when --from-sources is used."
    ),
    overlap: int = typer.Option(
        0, "--overlap", min=0, help="Inter-chunk overlap (chars) when --from-sources is used."
    ),
    k: int = typer.Option(5, "--k", min=1, help="Top-K under evaluation."),
    mode: Mode = typer.Option(Mode.HYBRID, "--mode", help="Retriever mode."),
    alpha: float = typer.Option(0.5, "--alpha", min=0.0, max=1.0, help="Hybrid alpha."),
    seed: int = typer.Option(0, "--seed", help="FakeEmbedder seed (determinism)."),
    embedder: EmbedderChoice = typer.Option(
        EmbedderChoice.FAKE,
        "--embedder",
        help="Embedder backend: fake (offline) or st (sentence-transformers, the 'st' extra).",
    ),
    rerank: bool = typer.Option(
        False,
        "--rerank",
        help="Rerank the candidate pool with a cross-encoder (requires the 'st' extra).",
    ),
    rerank_model: str | None = typer.Option(
        None, "--rerank-model", help="Cross-encoder model id override for --rerank."
    ),
    use_faiss: bool = typer.Option(
        False,
        "--faiss",
        help="Use the FAISS dense backend (requires the 'faiss' extra).",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.RICH, "--format", help="Output shape: rich, md, or json."
    ),
    as_json: bool = typer.Option(
        False, "--json", hidden=True, help="Deprecated alias for --format json."
    ),
) -> None:
    """Classify why an expected chunk failed and report the minimal fix."""
    fmt = _resolve_format(output_format, as_json)
    loaded, chunker = _resolve_corpus(corpus, from_sources, chunk_size, overlap)
    config = RetrievalConfig(
        top_k=k,
        chunk_size=chunk_size,
        alpha=alpha if mode is Mode.HYBRID else None,
        rerank=rerank,
    )
    retriever = _build_retriever(
        loaded,
        mode,
        alpha,
        seed,
        chunker,
        config,
        embedder,
        rerank,
        rerank_model,
        use_faiss,
    )
    result = _diagnose_with_fix(retriever, query, expect, config)

    if fmt is OutputFormat.JSON:
        payload = diagnosis_to_dict(result)
        payload["backend"] = _backend_of(retriever)
        _console.print_json(json_module.dumps(payload))
        return
    _print_backend(fmt, retriever)
    if fmt is OutputFormat.MD:
        sys.stdout.write(diagnosis_to_markdown(result))
        return
    render_diagnosis(result, _console)


@app.command()
def fix(
    query: str = typer.Argument(..., help="The failing query."),
    expect: str = typer.Option(..., "--expect", help="Id of the known-correct chunk."),
    corpus: Path | None = typer.Option(
        None, "--corpus", help="Path to a pre-chunked JSON-Lines corpus."
    ),
    from_sources: Path | None = typer.Option(
        None,
        "--from-sources",
        help="Path to a JSON-Lines file of {id, text} raw documents to chunk on the fly.",
    ),
    chunk_size: int = typer.Option(
        512, "--chunk-size", min=1, help="Chunk size (chars) when --from-sources is used."
    ),
    overlap: int = typer.Option(
        0, "--overlap", min=0, help="Inter-chunk overlap (chars) when --from-sources is used."
    ),
    k: int = typer.Option(5, "--k", min=1, help="Top-K under evaluation."),
    mode: Mode = typer.Option(Mode.HYBRID, "--mode", help="Retriever mode."),
    alpha: float = typer.Option(0.5, "--alpha", min=0.0, max=1.0, help="Hybrid alpha."),
    seed: int = typer.Option(0, "--seed", help="FakeEmbedder seed (determinism)."),
    embedder: EmbedderChoice = typer.Option(
        EmbedderChoice.FAKE,
        "--embedder",
        help="Embedder backend: fake (offline) or st (sentence-transformers, the 'st' extra).",
    ),
    rerank: bool = typer.Option(
        False,
        "--rerank",
        help="Rerank the candidate pool with a cross-encoder (requires the 'st' extra).",
    ),
    rerank_model: str | None = typer.Option(
        None, "--rerank-model", help="Cross-encoder model id override for --rerank."
    ),
    use_faiss: bool = typer.Option(
        False,
        "--faiss",
        help="Use the FAISS dense backend (requires the 'faiss' extra).",
    ),
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
    loaded, chunker = _resolve_corpus(corpus, from_sources, chunk_size, overlap)
    config = RetrievalConfig(
        top_k=k,
        chunk_size=chunk_size,
        alpha=alpha if mode is Mode.HYBRID else None,
        rerank=rerank,
    )
    retriever = _build_retriever(
        loaded,
        mode,
        alpha,
        seed,
        chunker,
        config,
        embedder,
        rerank,
        rerank_model,
        use_faiss,
    )
    result = search_fixes(retriever, query, expect, config)

    if fmt is OutputFormat.JSON:
        payload = fixes_to_dict(result)
        payload["backend"] = _backend_of(retriever)
        _console.print_json(json_module.dumps(payload))
        return
    _print_backend(fmt, retriever)
    if fmt is OutputFormat.MD:
        sys.stdout.write(fixes_to_markdown(result, show_all=show_all))
        return
    render_fixes(result, _console, show_all=show_all)


@app.command()
def batch(
    queries: Path = typer.Option(
        ..., "--queries", help="JSON-Lines file of {'query', 'expect'} rows."
    ),
    corpus: Path | None = typer.Option(
        None, "--corpus", help="Path to a pre-chunked JSON-Lines corpus."
    ),
    from_sources: Path | None = typer.Option(
        None,
        "--from-sources",
        help="Path to a JSON-Lines file of {id, text} raw documents to chunk on the fly.",
    ),
    chunk_size: int = typer.Option(
        512, "--chunk-size", min=1, help="Chunk size (chars) when --from-sources is used."
    ),
    overlap: int = typer.Option(
        0, "--overlap", min=0, help="Inter-chunk overlap (chars) when --from-sources is used."
    ),
    k: int = typer.Option(5, "--k", min=1, help="Top-K under evaluation."),
    mode: Mode = typer.Option(Mode.HYBRID, "--mode", help="Retriever mode."),
    alpha: float = typer.Option(0.5, "--alpha", min=0.0, max=1.0, help="Hybrid alpha."),
    seed: int = typer.Option(0, "--seed", help="FakeEmbedder seed (determinism)."),
    embedder: EmbedderChoice = typer.Option(
        EmbedderChoice.FAKE,
        "--embedder",
        help="Embedder backend: fake (offline) or st (sentence-transformers, the 'st' extra).",
    ),
    rerank: bool = typer.Option(
        False,
        "--rerank",
        help="Rerank the candidate pool with a cross-encoder (requires the 'st' extra).",
    ),
    rerank_model: str | None = typer.Option(
        None, "--rerank-model", help="Cross-encoder model id override for --rerank."
    ),
    use_faiss: bool = typer.Option(
        False,
        "--faiss",
        help="Use the FAISS dense backend (requires the 'faiss' extra).",
    ),
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
    loaded, chunker = _resolve_corpus(corpus, from_sources, chunk_size, overlap)
    config = RetrievalConfig(
        top_k=k,
        chunk_size=chunk_size,
        alpha=alpha if mode is Mode.HYBRID else None,
        rerank=rerank,
    )
    retriever = _build_retriever(
        loaded,
        mode,
        alpha,
        seed,
        chunker,
        config,
        embedder,
        rerank,
        rerank_model,
        use_faiss,
    )
    result = run_batch(retriever, batch_queries, config)

    if fmt is OutputFormat.JSON:
        payload = batch_to_dict(result)
        payload["backend"] = _backend_of(retriever)
        _console.print_json(json_module.dumps(payload))
    else:
        _print_backend(fmt, retriever)
        if fmt is OutputFormat.MD:
            sys.stdout.write(batch_to_markdown(result))
        else:
            render_batch(result, _console)

    # Report the results first, then trip the CI gate so pipelines still see the
    # full diagnosis before the non-zero exit.
    if _gate_tripped(result, fail_on):
        raise typer.Exit(code=_GATE_EXIT_CODE)


@app.command()
def validate(
    corpus: Path = typer.Option(..., "--corpus", help="Corpus JSON-Lines file to lint."),
    queries: Path | None = typer.Option(
        None, "--queries", help="Optional queries JSON-Lines file to lint."
    ),
) -> None:
    """Lint a corpus (and optional queries) file without running retrieval.

    Reports duplicate ids, blank ids/text, and malformed lines with 1-based line
    numbers, so JSONL problems surface in pre-commit/CI instead of as a mid-run
    ``ValueError``. Exits ``0`` when clean and ``2`` when any problem is found.
    """
    if not corpus.is_file():
        _err_console.print(f"[red]error:[/red] corpus file not found: {corpus}")
        raise typer.Exit(code=2)
    problems: list[str] = list(lint_jsonl(corpus))

    if queries is not None:
        if not queries.is_file():
            _err_console.print(f"[red]error:[/red] queries file not found: {queries}")
            raise typer.Exit(code=2)
        try:
            load_queries(queries)
        except ValueError as exc:
            problems.append(str(exc))

    if problems:
        for problem in problems:
            _err_console.print(f"[red]x[/red] {problem}")
        noun = "problem" if len(problems) == 1 else "problems"
        _err_console.print(f"[red]{len(problems)} {noun} found[/red]")
        raise typer.Exit(code=2)
    _console.print("[green]ok[/green] no problems found")


@app.command()
def serve(
    corpus: Path = typer.Option(..., "--corpus", help="Path to a JSON-Lines corpus."),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
    mode: Mode = typer.Option(Mode.HYBRID, "--mode", help="Retriever mode."),
    alpha: float = typer.Option(0.5, "--alpha", min=0.0, max=1.0, help="Hybrid alpha."),
    seed: int = typer.Option(0, "--seed", help="FakeEmbedder seed (determinism)."),
    embedder: EmbedderChoice = typer.Option(
        EmbedderChoice.FAKE,
        "--embedder",
        help="Embedder backend: fake (offline) or st (sentence-transformers, the 'st' extra).",
    ),
    rerank: bool = typer.Option(
        False,
        "--rerank",
        help="Rerank the candidate pool with a cross-encoder (requires the 'st' extra).",
    ),
    rerank_model: str | None = typer.Option(
        None, "--rerank-model", help="Cross-encoder model id override for --rerank."
    ),
    use_faiss: bool = typer.Option(
        False,
        "--faiss",
        help="Use the FAISS dense backend (requires the 'faiss' extra).",
    ),
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
    retriever = _build_retriever(
        loaded,
        mode,
        alpha,
        seed,
        embedder_choice=embedder,
        rerank=rerank,
        rerank_model=rerank_model,
        use_faiss=use_faiss,
    )
    web_app = create_app(retriever)
    uvicorn.run(web_app, host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    app()
