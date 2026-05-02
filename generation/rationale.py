"""
Utilities for extracting tool-choice rationale and proof sketches from generated
strategy bodies.

We embed rationale as a comment block at the top of the method body:

  # CSD_RATIONALE_BEGIN
  # ... explanation ...
  # CSD_RATIONALE_END

Generated strategies may also include a proof-sketch block:

  # CSD_PROOF_SKETCH_BEGIN
  # ... why the invariants/contracts should verify ...
  # CSD_PROOF_SKETCH_END

Older Dafny-era outputs used `//` comments, which we still accept.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RationaleExtraction:
    rationale: str | None
    body_without_rationale: str
    has_markers: bool


@dataclass(frozen=True)
class CommentBlockExtraction:
    """Result of extracting a required comment block from a strategy body."""

    text: str | None
    body_without_block: str
    has_markers: bool


BEGIN_MARKERS = {"// CSD_RATIONALE_BEGIN", "# CSD_RATIONALE_BEGIN"}
END_MARKERS = {"// CSD_RATIONALE_END", "# CSD_RATIONALE_END"}
PROOF_BEGIN_MARKERS = {"// CSD_PROOF_SKETCH_BEGIN", "# CSD_PROOF_SKETCH_BEGIN"}
PROOF_END_MARKERS = {"// CSD_PROOF_SKETCH_END", "# CSD_PROOF_SKETCH_END"}


def _extract_comment_block(
    strategy_body: str,
    *,
    begin_markers: set[str],
    end_markers: set[str],
) -> CommentBlockExtraction:
    """
    Extract an embedded comment block from a generated strategy body.

    - Returns `text=None` if markers are missing or empty.
    - Leaves the remaining code in `body_without_block` (best-effort).
    """
    lines = strategy_body.splitlines()

    begin_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip() in begin_markers:
            begin_idx = i
            break
    if begin_idx is None:
        return CommentBlockExtraction(text=None, body_without_block=strategy_body, has_markers=False)

    for j in range(begin_idx + 1, len(lines)):
        if lines[j].strip() in end_markers:
            end_idx = j
            break
    if end_idx is None:
        # Marker start without end: treat as missing to avoid mis-parsing code.
        return CommentBlockExtraction(text=None, body_without_block=strategy_body, has_markers=False)

    block_lines: list[str] = []
    for raw in lines[begin_idx + 1 : end_idx]:
        s = raw.strip()
        if not s:
            continue
        if s.startswith("//"):
            s = s[2:].lstrip()
        elif s.startswith("#"):
            s = s[1:].lstrip()
        block_lines.append(s)

    text = "\n".join([ln for ln in block_lines if ln]).strip() or None

    # Remove the block from the body (including markers), preserving remaining lines.
    body_without = "\n".join(lines[:begin_idx] + lines[end_idx + 1 :]).lstrip("\n")

    return CommentBlockExtraction(text=text, body_without_block=body_without, has_markers=True)


def extract_rationale(strategy_body: str) -> RationaleExtraction:
    """
    Extract an embedded rationale block from a generated strategy body.

    - Returns `rationale=None` if markers are missing or empty.
    - Leaves the remaining code in `body_without_rationale` (best-effort).
    """
    extracted = _extract_comment_block(
        strategy_body,
        begin_markers=BEGIN_MARKERS,
        end_markers=END_MARKERS,
    )
    return RationaleExtraction(
        rationale=extracted.text,
        body_without_rationale=extracted.body_without_block,
        has_markers=extracted.has_markers,
    )


def extract_proof_sketch(strategy_body: str) -> CommentBlockExtraction:
    """Extract the generated proof-sketch comment block from a strategy body."""
    return _extract_comment_block(
        strategy_body,
        begin_markers=PROOF_BEGIN_MARKERS,
        end_markers=PROOF_END_MARKERS,
    )
