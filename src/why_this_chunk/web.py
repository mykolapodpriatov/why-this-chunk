"""Optional read-only FastAPI inspector (``[web]`` extra).

A deliberately small, read-only view over a single retriever: a couple of JSON
endpoints plus one HTML page. No persistence, no auth, no mutation — it exists to
eyeball explanations in a browser, nothing more. Build with
:func:`create_app` and serve via ``why-this-chunk serve`` (or any ASGI server).

This module is import-guarded: calling :func:`create_app` without ``fastapi``
raises a clear :class:`ImportError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from why_this_chunk.attribution import explain_chunk
from why_this_chunk.report import explanation_to_dict
from why_this_chunk.retrievers import Retriever

if TYPE_CHECKING:
    from fastapi import FastAPI

__all__ = ["create_app"]

# The client script builds the results table purely with DOM nodes and
# ``textContent`` (never ``innerHTML`` for response data), so chunk text and ids
# cannot inject markup — XSS-safe by construction.
_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>why-this-chunk inspector</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 60rem; }}
 input {{ width: 28rem; padding: .4rem; }}
 button {{ padding: .4rem .8rem; }}
 table {{ border-collapse: collapse; margin-top: 1rem; width: 100%; }}
 th, td {{ border: 1px solid #ccc; padding: .3rem .5rem; text-align: left; }}
 .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
 code {{ background: #f3f3f3; padding: 0 .2rem; }}
</style>
</head>
<body>
<h1>why-this-chunk inspector</h1>
<p>Read-only. Corpus size: <code>{corpus_size}</code>.</p>
<form id="f">
  <input id="q" placeholder="query" autocomplete="off" />
  <button type="submit">explain</button>
</form>
<div id="out"></div>
<script>
function el(tag, text, cls) {{
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}}
function clear(node) {{ while (node.firstChild) node.removeChild(node.firstChild); }}
async function run(ev) {{
  ev.preventDefault();
  const q = document.getElementById('q').value;
  const out = document.getElementById('out');
  clear(out);
  const res = await fetch('/api/explain?query=' + encodeURIComponent(q) + '&k=5');
  const data = await res.json();
  if (!data.explanations || data.explanations.length === 0) {{
    out.appendChild(el('p', 'no results'));
    return;
  }}
  for (const e of data.explanations) {{
    out.appendChild(el('h3',
      'chunk ' + e.result.chunk_id + ' — score ' + e.result.score.toFixed(4) +
      ' (rank ' + e.result.rank + ')'));
    if (e.degenerate) out.appendChild(el('p', 'degenerate: no single unit dominates'));
    const table = el('table');
    const head = el('tr');
    head.appendChild(el('th', 'share'));
    head.appendChild(el('th', 'delta'));
    head.appendChild(el('th', e.granularity));
    table.appendChild(head);
    for (const s of e.sentences) {{
      const row = el('tr');
      row.appendChild(el('td', s.share.toFixed(3), 'num'));
      row.appendChild(el('td', s.delta.toFixed(4), 'num'));
      row.appendChild(el('td', s.sentence));
      table.appendChild(row);
    }}
    out.appendChild(table);
  }}
}}
document.getElementById('f').addEventListener('submit', run);
</script>
</body>
</html>
"""


def create_app(retriever: Retriever) -> FastAPI:
    """Build the read-only inspector app over ``retriever``.

    Endpoints:
        * ``GET /`` — the HTML inspector page.
        * ``GET /api/health`` — liveness and corpus size.
        * ``GET /api/explain?query=...&k=...`` — JSON explanations for the
          top-``k`` results (sentence granularity).

    Args:
        retriever: The retriever to inspect (read-only).

    Returns:
        A configured :class:`fastapi.FastAPI` application.

    Raises:
        ImportError: If the ``[web]`` extra (``fastapi``) is not installed.
    """
    try:
        from fastapi import FastAPI, Query
        from fastapi.responses import HTMLResponse
    except ImportError as exc:  # pragma: no cover - only without extra
        raise ImportError(
            "create_app requires the optional 'web' extra. "
            "Install it with: pip install 'why-this-chunk[web]'"
        ) from exc

    api = FastAPI(title="why-this-chunk inspector", docs_url="/api/docs")

    @api.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _PAGE.format(corpus_size=retriever.corpus_size)

    @api.get("/api/health")
    def health() -> dict[str, Any]:
        payload: dict[str, Any] = {"status": "ok", "corpus_size": retriever.corpus_size}
        backend = getattr(retriever, "backend", None)
        if isinstance(backend, str):
            payload["backend"] = backend
        return payload

    @api.get("/api/explain")
    def explain(
        query: str = Query(..., min_length=1),
        k: int = Query(5, ge=1, le=50),
    ) -> dict[str, Any]:
        results = retriever.search(query, k)
        explanations = [
            explanation_to_dict(explain_chunk(retriever, query, result)) for result in results
        ]
        payload: dict[str, Any] = {"query": query, "explanations": explanations}
        backend = getattr(retriever, "backend", None)
        if isinstance(backend, str):
            payload["backend"] = backend
        return payload

    return api
