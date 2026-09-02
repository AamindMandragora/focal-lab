"""Golden (characterization) tests for the Family-1 AUTHOR prompt builders (prompts.py).

Pins, byte-for-byte on current code:
  - the static SYSTEM_PROMPT, and the derived TOOL_REFERENCE / _ALL_HELPER_NAMES /
    VERIFIED_EXAMPLES (index-sliced / regex-extracted from the constants — these feed
    fairness-sensitive machinery, so they MUST not drift),
  - the rendered user prompt from each of the 6 build_*_prompt composers, in a minimal
    variant (all optional sub-blocks empty) and, where they have conditional sub-blocks,
    a fully-populated variant.

This is the byte-identity gate for the Phase-1 (byte-identical) Jinja conversion of the
author prompts, and the "before" baseline for the Phase-2 descriptive change.

Regenerate: REGEN_GOLDENS=1 pytest <thisfile>  (only against known-good current code).
"""
import os
import pathlib

import pytest

from synthesis.generate import prompts

GOLDEN_DIR = pathlib.Path(__file__).parent / "fixtures" / "author_prompts"


def _check(name: str, produced: str):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_DIR / f"{name}.golden.txt"
    if os.environ.get("REGEN_GOLDENS"):
        path.write_text(produced)
        pytest.skip(f"regenerated {path.name}")
    assert path.read_text() == produced


# A fixed, small allowed-helpers subset to exercise the non-None filtering path
# (must be real helper names so _build_tool_reference_block keeps their doc blocks).
_SUBSET = sorted(prompts._ALL_HELPER_NAMES)[:3]


# ---------------------------------------------------------------------------
# Static constant + derived-value pins (fairness-sensitive)
# ---------------------------------------------------------------------------

def test_system_prompt_constant():
    _check("system_prompt", prompts.SYSTEM_PROMPT)


def test_tool_reference_constant():
    _check("tool_reference", prompts.TOOL_REFERENCE)


def test_all_helper_names():
    _check("all_helper_names", "\n".join(sorted(prompts._ALL_HELPER_NAMES)))


def test_verified_examples_constant():
    _check("verified_examples", prompts.VERIFIED_EXAMPLES)


# ---------------------------------------------------------------------------
# build_initial_prompt
# ---------------------------------------------------------------------------

def test_initial_prompt_full_helpers():
    _sys, user = prompts.build_initial_prompt("Solve the task.", allowed_helpers=None)
    _check("initial__full", user)


def test_initial_prompt_subset_helpers():
    _sys, user = prompts.build_initial_prompt("Solve the task.", allowed_helpers=_SUBSET)
    _check("initial__subset", user)


# ---------------------------------------------------------------------------
# build_verification_error_prompt  (many conditional sub-blocks)
# ---------------------------------------------------------------------------

def test_verification_error_minimal():
    _sys, user = prompts.build_verification_error_prompt(
        task_description="Solve the task.",
        previous_strategy="// prev strategy",
        error_message="postcondition might not hold",
    )
    _check("verify_err__minimal", user)


def test_verification_error_full():
    _sys, user = prompts.build_verification_error_prompt(
        task_description="Solve the task.",
        previous_strategy="// prev strategy",
        error_message="postcondition might not hold",
        behavioral_context="Example 1: token_count 42",
        structured_feedback="Diagnostic: postcondition at line 12",
        error_history="Attempt 2: postcondition at line 12",
        strategy_context="- Attempt 1: accuracy 25.0%",
        search_memory="Prior memory note.",
    )
    _check("verify_err__full", user)


# ---------------------------------------------------------------------------
# build_runtime_error_prompt
# ---------------------------------------------------------------------------

def test_runtime_error_minimal():
    _sys, user = prompts.build_runtime_error_prompt(
        previous_strategy="// prev strategy",
        error_traceback="Traceback: IndexError",
    )
    _check("runtime_err__minimal", user)


def test_runtime_error_with_memory():
    _sys, user = prompts.build_runtime_error_prompt(
        previous_strategy="// prev strategy",
        error_traceback="Traceback: IndexError",
        task_description="Solve the task.",
        search_memory="Prior memory note.",
    )
    _check("runtime_err__with_memory", user)


# ---------------------------------------------------------------------------
# build_compilation_error_prompt
# ---------------------------------------------------------------------------

def test_compilation_error_minimal():
    _sys, user = prompts.build_compilation_error_prompt(
        previous_strategy="// prev strategy",
        error_message="Compilation failed with 1 error(s):",
    )
    _check("compile_err__minimal", user)


# ---------------------------------------------------------------------------
# build_format_repair_prompt
# ---------------------------------------------------------------------------

def test_format_repair_minimal():
    _sys, user = prompts.build_format_repair_prompt(previous_strategy="// prev strategy")
    _check("format_repair__minimal", user)


# ---------------------------------------------------------------------------
# build_evaluation_failure_prompt  (best-so-far / mode-examples / ledger / budget blocks)
# ---------------------------------------------------------------------------

def test_evaluation_failure_minimal():
    _sys, user = prompts.build_evaluation_failure_prompt(
        task_description="Solve the task.",
        previous_strategy="// prev strategy",
        previous_accuracy=0.25,
        previous_syntax_rate=0.60,
        num_examples=20,
        goal_accuracy=0.50,
        goal_syntax_rate=0.90,
        evaluation_feedback="Accuracy 25.0%; syntax 60.0%.",
    )
    _check("eval_fail__minimal", user)


def test_evaluation_failure_full():
    _sys, user = prompts.build_evaluation_failure_prompt(
        task_description="Solve the task.",
        previous_strategy="// prev strategy",
        previous_accuracy=0.25,
        previous_syntax_rate=0.60,
        num_examples=20,
        goal_accuracy=0.50,
        goal_syntax_rate=0.90,
        evaluation_feedback="Accuracy 25.0%; syntax 60.0%.",
        best_strategy="// best strategy",
        best_accuracy=0.40,
        best_syntax_rate=0.85,
        search_memory="Prior memory note.",
        eval_max_seconds_per_example=30.0,
        mode_examples="--- Example of runtime_or_generation_error ---",
        attempt_outcome_ledger="Best result:",
    )
    _check("eval_fail__full", user)
