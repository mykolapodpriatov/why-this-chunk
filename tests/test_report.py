"""Tests for the renderer: it must not crash, and exports must have shape."""

from __future__ import annotations

import io

from rich.console import Console

from why_this_chunk import (
    HybridRetriever,
    RetrievalConfig,
    diagnose,
    explain_chunk,
    search_fixes,
)
from why_this_chunk.batch import BatchQuery, run_batch
from why_this_chunk.report import (
    batch_to_dict,
    batch_to_markdown,
    diagnosis_to_dict,
    diagnosis_to_markdown,
    explanation_to_dict,
    explanation_to_markdown,
    render_batch,
    render_diagnosis,
    render_explanation,
)
from why_this_chunk.types import (
    Chunk,
    DiagnosisResult,
    Explanation,
    FailureClass,
    ScoredChunk,
)


def _quiet_console() -> Console:
    # Render to an in-memory buffer so tests stay silent but still exercise the
    # full rendering path (no file handles => no ResourceWarning).
    return Console(file=io.StringIO(), force_terminal=False, width=100)


def test_render_explanation_does_not_crash(
    hybrid: HybridRetriever,
) -> None:
    result = hybrid.search("Paris France", 1)[0]
    explanation = explain_chunk(hybrid, "Paris France", result)
    render_explanation(explanation, _quiet_console())


def test_render_explanation_degenerate_flag() -> None:
    # An explanation explicitly flagged degenerate must render its banner path.
    explanation = Explanation(
        query="q",
        result=ScoredChunk(chunk=Chunk(id="x", text="a. b."), score=0.0, rank=0),
        sentences=[],
        degenerate=True,
    )
    render_explanation(explanation, _quiet_console())


def test_render_diagnosis_does_not_crash(hybrid: HybridRetriever) -> None:
    result = diagnose(hybrid, "Paris", "seine", RetrievalConfig(top_k=1, alpha=0.5))
    render_diagnosis(result, _quiet_console())


def test_render_diagnosis_with_fix(hybrid: HybridRetriever) -> None:
    diag = diagnose(hybrid, "Paris France", "seine", RetrievalConfig(top_k=1, alpha=0.5))
    fixes = search_fixes(hybrid, "Paris France", "seine", RetrievalConfig(top_k=1, alpha=0.5))
    enriched = DiagnosisResult(
        failure_class=diag.failure_class,
        unevaluable=diag.unevaluable,
        evidence=diag.evidence,
        fix=fixes.best,
    )
    render_diagnosis(enriched, _quiet_console())


def test_render_indeterminate_diagnosis() -> None:
    diag = DiagnosisResult(failure_class=None, unevaluable=[FailureClass.LOST_TO_CHUNKING])
    render_diagnosis(diag, _quiet_console())


def test_explanation_markdown_shape(hybrid: HybridRetriever) -> None:
    result = hybrid.search("Paris France", 1)[0]
    explanation = explain_chunk(hybrid, "Paris France", result)
    md = explanation_to_markdown(explanation)
    assert md.startswith("## explain")
    assert "| share | delta |" in md
    assert md.endswith("\n")


def test_diagnosis_markdown_shape(hybrid: HybridRetriever) -> None:
    diag = diagnose(hybrid, "Paris", "seine", RetrievalConfig(top_k=1, alpha=0.5))
    md = diagnosis_to_markdown(diag)
    assert md.startswith("## diagnose")
    assert "| evidence | value |" in md


def test_explanation_dict_is_json_safe(hybrid: HybridRetriever) -> None:
    import json

    result = hybrid.search("Paris France", 1)[0]
    explanation = explain_chunk(hybrid, "Paris France", result)
    json.dumps(explanation_to_dict(explanation))  # must not raise


def test_diagnosis_dict_is_json_safe(hybrid: HybridRetriever) -> None:
    import json

    diag = diagnose(hybrid, "Paris", "seine", RetrievalConfig(top_k=1, alpha=0.5))
    json.dumps(diagnosis_to_dict(diag))  # must not raise


def test_render_batch_and_exports_do_not_crash(hybrid: HybridRetriever) -> None:
    import json

    queries = [
        BatchQuery(query="Paris France", expect="seine"),
        BatchQuery(query="Paris France", expect="nonexistent"),
    ]
    result = run_batch(hybrid, queries, RetrievalConfig(top_k=1, alpha=0.5))
    render_batch(result, _quiet_console())
    md = batch_to_markdown(result)
    assert md.startswith("## batch")
    assert "| query | expect | failure | fix |" in md
    json.dumps(batch_to_dict(result))  # must not raise


def test_render_batch_empty(hybrid: HybridRetriever) -> None:
    result = run_batch(hybrid, [], RetrievalConfig(top_k=1, alpha=0.5))
    render_batch(result, _quiet_console())  # empty path must not crash
    assert "no queries" in batch_to_markdown(result)
    assert batch_to_dict(result)["count"] == 0


def test_markdown_escapes_pipe_in_text() -> None:
    explanation = Explanation(
        query="q",
        result=ScoredChunk(chunk=Chunk(id="x", text="a | b sentence."), score=1.0, rank=0),
        sentences=[],
    )
    # Build one attribution with a pipe to exercise escaping.
    from why_this_chunk.types import SentenceAttribution

    explanation.sentences.append(
        SentenceAttribution(sentence="a | b", span=(0, 5), delta=0.5, share=1.0)
    )
    md = explanation_to_markdown(explanation)
    assert "\\|" in md
