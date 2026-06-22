"""Lightweight, dependency-free text utilities.

A deliberately small sentence/token splitter is used instead of a heavy NLP
dependency. The sentence splitter is regex-based with a short abbreviation guard
and is documented as *approximate* — it is pluggable at the attribution layer if
a user needs better segmentation.
"""

from __future__ import annotations

import re

__all__ = ["split_sentences", "token_spans", "tokenize"]

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Sentence boundary: terminal punctuation (. ! ?) possibly followed by quotes/
# brackets, then whitespace. We then drop boundaries that follow a known
# abbreviation to reduce false splits.
_BOUNDARY_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+")

_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "fig",
        "vol",
        "pp",
    }
)

# Trailing token allowing internal dots (so "e.g" / "i.e" are matched whole).
_TRAILING_ABBR_RE = re.compile(r"([A-Za-z](?:\.?[A-Za-z])*)\.?\s*$")


def tokenize(text: str) -> list[str]:
    """Lowercase word-character tokenization (matches the embedder/BM25)."""
    return _TOKEN_RE.findall(text.lower())


def _ends_with_abbreviation(text: str) -> bool:
    """Whether ``text`` ends with a guarded abbreviation + period.

    Handles both plain abbreviations (``Dr.``) and dotted ones (``e.g.``,
    ``i.e.``) by matching the trailing run of letters with optional internal
    dots.
    """
    tail = text.rstrip()
    if not tail.endswith("."):
        return False
    match = _TRAILING_ABBR_RE.search(tail[:-1])
    if match is None:
        return False
    return match.group(1).lower() in _ABBREVIATIONS


def split_sentences(text: str) -> list[tuple[str, tuple[int, int]]]:
    """Split ``text`` into sentences with their char spans.

    The splitter is approximate: it breaks on terminal punctuation followed by
    whitespace, guarding a small set of common abbreviations. Each returned span
    is ``(start, end)`` into the original ``text`` and the substring excludes
    surrounding whitespace. If no boundary is found, the whole (trimmed) text is
    a single sentence.

    Args:
        text: The text to segment.

    Returns:
        A list of ``(sentence, (start, end))`` pairs in reading order. Empty if
        ``text`` is blank.
    """
    if not text.strip():
        return []

    # Candidate cut points (offset just after the boundary whitespace).
    pieces: list[tuple[str, tuple[int, int]]] = []
    cursor = 0
    for match in _BOUNDARY_RE.finditer(text):
        cut = match.start()
        # ``text[cursor:cut]`` already ends with the terminal punctuation (the
        # boundary regex uses a ``(?<=[.!?])`` lookbehind), so pass it directly
        # to the abbreviation guard — appending another "." would double the dot.
        # Suppress the split if the segment ends with a guarded abbreviation.
        if _ends_with_abbreviation(text[cursor:cut]):
            continue
        stripped, span = _strip_span(text, cursor, match.end())
        if stripped:
            pieces.append((stripped, span))
        cursor = match.end()

    stripped, span = _strip_span(text, cursor, len(text))
    if stripped:
        pieces.append((stripped, span))
    return pieces


def _strip_span(text: str, start: int, end: int) -> tuple[str, tuple[int, int]]:
    """Return the whitespace-trimmed substring and its adjusted span."""
    left = start
    right = end
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    return text[left:right], (left, right)


def token_spans(text: str) -> list[tuple[str, tuple[int, int]]]:
    """Return tokens with their char spans (for token-granularity attribution).

    Tokens are matched on word characters but reported in their original
    (non-lowercased) surface form so spans index back into ``text`` directly.
    """
    return [(m.group(0), (m.start(), m.end())) for m in _TOKEN_RE.finditer(text)]
