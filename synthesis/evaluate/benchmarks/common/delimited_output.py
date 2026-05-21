"""Shared << >> delimited-span extraction for benchmark scoring."""

from __future__ import annotations

import re

DELIMITED_SPAN_PATTERN = re.compile(r"<<\s*([^<>]+?)\s*>>")
_MARKDOWN_SQL_FENCE = re.compile(
    r"```(?:sql)?\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_SELECT_STMT = re.compile(
    r"(SELECT\b[\s\S]*?)(?:;|\s*```|$)",
    re.IGNORECASE,
)


def find_delimited_spans(text: str) -> list[str]:
    """Return inner spans from all ``<< ... >>`` regions in document order."""
    if not text:
        return []
    return DELIMITED_SPAN_PATTERN.findall(text)


def normalize_inline_text(text: str, *, strip_semicolon: bool = False) -> str:
    """Collapse whitespace and optional trailing semicolons for SQL-like spans."""
    cleaned = text.replace("\n", " ").replace("\r", " ").strip()
    cleaned = " ".join(cleaned.split())
    if strip_semicolon:
        cleaned = cleaned.rstrip(";").strip()
    return cleaned


def extract_last_delimited_span(
    text: str,
    *,
    normalize_whitespace: bool = False,
    strip_semicolon: bool = False,
) -> tuple[str | None, bool]:
    """
    Return the last ``<< >>`` inner span and whether any delimiter was found.

    When ``normalize_whitespace`` is true, apply :func:`normalize_inline_text`.
    """
    matches = find_delimited_spans(text)
    if not matches:
        return None, False
    span = matches[-1]
    if normalize_whitespace:
        span = normalize_inline_text(span, strip_semicolon=strip_semicolon)
    else:
        span = span.strip()
    return (span or None), True


def _extract_sql_from_markdown_or_select(text: str) -> str | None:
    """Pull a SELECT statement out of markdown fences or noisy CoT prose."""
    if not text:
        return None
    fences = _MARKDOWN_SQL_FENCE.findall(text)
    if fences:
        candidate = normalize_inline_text(fences[-1], strip_semicolon=True)
        if candidate and re.search(r"\bselect\b", candidate, flags=re.IGNORECASE):
            return candidate
    matches = list(_SELECT_STMT.finditer(text))
    if matches:
        candidate = normalize_inline_text(matches[-1].group(1), strip_semicolon=True)
        if candidate:
            return candidate
    return None


def extract_sql_scored_output(scored_output: str) -> tuple[str | None, str]:
    """
    Extract a SQL answer from model output.

    Prefer the last ``<< >>`` span; otherwise try markdown ``sql`` fences or the
    last SELECT-shaped span; finally use the first-paragraph fallback.
    """
    if not scored_output:
        return None, "none"

    span, found = extract_last_delimited_span(
        scored_output,
        normalize_whitespace=True,
        strip_semicolon=True,
    )
    if found and span and re.search(r"\bselect\b", span, flags=re.IGNORECASE):
        return span, "last_visible_span"

    from_markdown = _extract_sql_from_markdown_or_select(scored_output)
    if from_markdown:
        return from_markdown, "markdown_or_select_span"

    raw = scored_output.split("\n\n")[0]
    cleaned = normalize_inline_text(
        raw.replace("<<", " ").replace(">>", " "),
        strip_semicolon=True,
    )
    cleaned = cleaned.rstrip("`").strip()
    if cleaned and re.search(r"\bselect\b", cleaned, flags=re.IGNORECASE):
        return cleaned, "raw_text_fallback"
    return (cleaned or None), ("raw_text_fallback" if cleaned else "none")
