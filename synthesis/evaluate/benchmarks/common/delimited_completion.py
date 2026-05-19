"""Normalize constrained-decoder output into ``<<answer>>`` for shared scoring."""

from __future__ import annotations


def wrap_constrained_completion(completion: str) -> str:
    """Wrap a grammar-closed body (optionally ending in ``>>``) as one visible span."""
    text = (completion or "").strip()
    if not text:
        return ""

    if text.startswith("<<") and ">>" in text:
        return text if text.rstrip().endswith(">>") else f"{text.rstrip()}>>"

    if text.endswith(">>"):
        inner = text[:-2].strip()
        return f"<<{inner}>>" if inner else ""

    return f"<<{text}>>"
