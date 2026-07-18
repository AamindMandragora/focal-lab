"""Typed data for the 6 Family-1 AUTHOR prompt builders in `synthesis/generate/prompts.py`:

  - build_initial_prompt
  - build_verification_error_prompt
  - build_runtime_error_prompt
  - build_compilation_error_prompt
  - build_format_repair_prompt
  - build_evaluation_failure_prompt

This is a byte-identical conversion from `.format()` string templates to Jinja2.
Every field here is a value the build function already computes (the
`_build_*_block` composer functions are unchanged) or, for the two `:.1%`/`:.1f`
format-spec numbers in `EvaluationFailurePromptModel`, a pre-formatted string
computed with the same Python format-spec mini-language the original
`.format()` call used, so the number renders identically.
"""
from synthesis.prompt_rendering.base import PromptModel


class InitialPromptModel(PromptModel):
    """Data for build_initial_prompt."""

    task_description: str
    allowed_helpers_block: str
    tool_reference_block: str
    verified_examples: str


class VerificationErrorPromptModel(PromptModel):
    """Data for build_verification_error_prompt."""

    task_description: str
    allowed_helpers_block: str
    tool_reference_block: str
    verified_examples: str
    search_memory_block: str
    previous_strategy: str
    strategy_context_block: str
    error_message: str
    structured_feedback_block: str
    error_history_block: str
    behavioral_context_block: str


class RuntimeErrorPromptModel(PromptModel):
    """Data for build_runtime_error_prompt."""

    task_description: str
    allowed_helpers_block: str
    tool_reference_block: str
    search_memory_block: str
    previous_strategy: str
    error_traceback: str


class CompilationErrorPromptModel(PromptModel):
    """Data for build_compilation_error_prompt."""

    allowed_helpers_block: str
    tool_reference_block: str
    search_memory_block: str
    previous_strategy: str
    error_message: str


class FormatRepairPromptModel(PromptModel):
    """Data for build_format_repair_prompt."""

    allowed_helpers_block: str
    tool_reference_block: str
    search_memory_block: str
    previous_strategy: str


class EvaluationFailurePromptModel(PromptModel):
    """Data for build_evaluation_failure_prompt.

    previous_accuracy_str / previous_syntax_rate_str / goal_accuracy_str /
    goal_syntax_rate_str / accuracy_gap_pp_str / syntax_gap_pp_str are
    pre-formatted with Python's own `:.1%` / `:.1f` format specs (not a Jinja
    filter) so the rendered digits match the original `.format()` call exactly.
    """

    task_description: str
    allowed_helpers_block: str
    tool_reference_block: str
    verified_examples: str
    search_memory_block: str
    previous_strategy: str
    previous_accuracy_str: str
    previous_syntax_rate_str: str
    num_examples: int
    goal_accuracy_str: str
    goal_syntax_rate_str: str
    eval_budget_block: str
    accuracy_gap_pp_str: str
    syntax_gap_pp_str: str
    evaluation_feedback: str
    attempt_outcome_ledger_block: str
    mode_examples_block: str
    best_so_far_block: str
