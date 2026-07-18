"""Typed data for the 6 near-pure feedback_loop.py string builders:

  - _delimiter_miss_hint
  - _token_cap_exhaustion_hint
  - _span_not_closed_hint
  - _constraint_bypassed_hint
  - _final_span_failure_hint
  - SynthesisExhaustionError.get_failure_summary

Every field here is a value the surrounding function already computes
(counts, ratios, pre-classified/truncated strings); this module only
carries that data to the `feedback_loop/*.j2` templates. All guard
branches (the various "" early returns, and get_failure_summary's
"No attempts were made") stay plain Python returns in the calling
function — only the non-empty message branches are routed through a
template, matching the exact idiom used by `models/verify.py`.

`HintLinesModel` is shared by `_final_span_failure_hint` and
`get_failure_summary`: both build a Python list of already-formatted
lines (including "" entries for blank lines) and hand it to the same
`feedback_loop/hint_lines.j2` template, which lays the lines out one
per row — mirroring the "\\n".join(lines) idiom those two functions used
before conversion.
"""
from typing import List

from synthesis.prompt_rendering.base import PromptModel


class DelimiterMissOpenNotClosedModel(PromptModel):
    """Data for _delimiter_miss_hint's "spans opened but never closed" branch."""

    n_open_not_closed: int
    n: int


class DelimiterMissDefaultModel(PromptModel):
    """Data for _delimiter_miss_hint's default (force-delimiter) branch.

    No dynamic fields: the message is entirely static text.
    """


class TokenCapExhaustionModel(PromptModel):
    """Data for _token_cap_exhaustion_hint's non-empty branch."""

    n_capped: int
    n: int
    max_steps: int


class SpanNotClosedModel(PromptModel):
    """Data for _span_not_closed_hint's non-empty branch."""

    n_affected: int
    n: int


class ConstraintBypassedModel(PromptModel):
    """Data for _constraint_bypassed_hint's non-empty branch."""

    n_engaged: int
    n_rel: int
    n_bypassed: int


class HintLinesModel(PromptModel):
    """A pre-built list of lines, one per template row.

    Shared by `_final_span_failure_hint` and
    `SynthesisExhaustionError.get_failure_summary`: all classification,
    counting, and truncation happens in Python; this model only carries
    the finished lines (including "" entries for blank lines) to the
    template.
    """

    lines: List[str]
