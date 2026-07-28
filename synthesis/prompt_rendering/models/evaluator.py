"""Typed data for `EvaluationResult._render_mode_examples`.

Every field here is a value the surrounding method already computes (the
representative-sample picks from `_pick_representative_samples_by_mode`, the
prompt/question fallback, and the None-to-"" coalescing for `actual`/
`expected`). This module only carries that data to the
`evaluator/mode_examples.j2` template; it does not compute anything new.
"""
from typing import List

from synthesis.prompt_rendering.base import PromptModel


class ModeExampleEntry(PromptModel):
    """One verbatim failing-rollout block for a representative failure mode."""

    mode: str
    prompt: str
    qwen_output: str
    actual_str: str
    expected_str: str


class ModeExamplesModel(PromptModel):
    """All data needed to render the full `_render_mode_examples` output."""

    blocks: List[ModeExampleEntry]
