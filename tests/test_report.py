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
from why_this_chunk.counterfactual import CounterfactualResult, FixPlan
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
from why_this_chunk.types import (
    Chunk,
    DiagnosisResult,
    Explanation,
    FailureClass,
    FixSuggestion,
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


# ---------------------------------------------------------------------------
# two-axis fix rendering
# ---------------------------------------------------------------------------


def _pair_result() -> CounterfactualResult:
    """A result whose only fix needs two changes."""
    plan = FixPlan(
        changes=(
            FixSuggestion(
                param="chunk_size",
                from_value=512,
                to_value=256,
                cost=2,
                new_rank=0,
                explanation="set chunk_size to 256 so the expected text stays in one chunk",
            ),
            FixSuggestion(
                param="rerank",
                from_value=False,
                to_value=True,
                cost=4,
                new_rank=0,
                explanation="enable the reranker",
            ),
        ),
        cost=6,
        new_rank=0,
    )
    return CounterfactualResult(best=None, all_fixes=[], unevaluable=[], pair_fixes=[plan])


def test_terminal_marks_a_pair_as_two_changes() -> None:
    # A reader skimming a batch report must not have to count fields to see
    # that applying this means editing two knobs.
    console = Console(file=io.StringIO(), width=200, color_system=None)

    render_fixes(_pair_result(), console)

    out = console.file.getvalue()  # type: ignore[attr-defined]
    assert "2 axes" in out
    assert "chunk_size + rerank" in out
    assert "no single change worked" in out


def test_terminal_distinguishes_capped_from_no_fix() -> None:
    console = Console(file=io.StringIO(), width=200, color_system=None)
    capped = CounterfactualResult(best=None, all_fixes=[], unevaluable=[], capped=True)

    render_fixes(capped, console)

    out = console.file.getvalue()  # type: ignore[attr-defined]
    assert "cap" in out
    assert "--max-combinations" in out


def test_markdown_gives_a_pair_its_own_axes_column() -> None:
    md = fixes_to_markdown(_pair_result())

    assert "two together" in md
    assert "| axes |" in md
    assert "**chunk_size + rerank**" in md


def test_json_carries_the_plan_and_the_cap_flag() -> None:
    payload = fixes_to_dict(_pair_result())

    assert payload["best"] is None
    assert payload["best_plan"]["axes"] == ["chunk_size", "rerank"]
    assert payload["best_plan"]["cost"] == 6
    assert len(payload["best_plan"]["changes"]) == 2
    assert payload["capped"] is False


def test_json_best_plan_wraps_a_single_axis_fix() -> None:
    """One shape for the caller to read, whether the answer is one change or two."""
    fix = FixSuggestion(
        param="top_k", from_value=5, to_value=8, cost=3, new_rank=7, explanation="raise top_k"
    )
    payload = fixes_to_dict(CounterfactualResult(best=fix, all_fixes=[fix], unevaluable=[]))

    assert payload["best_plan"]["axes"] == ["top_k"]
    assert len(payload["best_plan"]["changes"]) == 1
