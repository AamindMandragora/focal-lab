"""Fairness gate on the LIVE rendered author prompts (the .j2 templates).

Why this file exists
--------------------
The old `synthesis/generate/test_prompt_fairness.py` and the fairness_cloud
suite used to assert on module-level constants like `INITIAL_GENERATION_PROMPT`
/ `EVALUATION_FAILURE_REFINEMENT_PROMPT`. After the pydantic+Jinja conversion
the `build_*_prompt` functions no longer render from those constants -- they
render from `templates/author_prompts/*.j2`, and `INITIAL_GENERATION_PROMPT`
survives only as the source text `VERIFIED_EXAMPLES` is sliced out of. So the
old constant-based fairness test was auditing dead strings, not the prompt the
author model actually receives, and has been deleted; the fairness_cloud suite
was migrated to check the live rendered text instead.

This file closes that gap: it renders each of the 6 build_*_prompt composers and
asserts the fairness contract on the REAL rendered output --
  (1) no GSM-arithmetic task-specific phrase leaks into any author prompt, and
  (2) the general CSD-mechanism guidance that must be preserved is still present
      in the rendered evaluation_failure prompt (guard against over-stripping).
A positive anchor on each descriptive template proves the test is exercising the
live .j2 prose (not a stale/empty render).
"""
from synthesis.generate import prompts


# GSM-arithmetic task-specific phrases banned from every author prompt.
BANNED_TASK_SPECIFIC = [
    "exact arithmetic",
    "integer division",
    "exponentiation",
    "never use `^`",
]


def _all_rendered_author_prompts():
    """(name, rendered_user_prompt) for every build_*_prompt, full variants."""
    out = []

    _sys, user = prompts.build_initial_prompt("Solve the task.", allowed_helpers=None)
    out.append(("initial", user))

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
    out.append(("verification_error", user))

    _sys, user = prompts.build_runtime_error_prompt(
        previous_strategy="// prev strategy",
        error_traceback="Traceback: IndexError",
        task_description="Solve the task.",
        search_memory="Prior memory note.",
    )
    out.append(("runtime_error", user))

    _sys, user = prompts.build_compilation_error_prompt(
        previous_strategy="// prev strategy",
        error_message="Compilation failed with 1 error(s):",
    )
    out.append(("compilation_error", user))

    _sys, user = prompts.build_format_repair_prompt(previous_strategy="// prev strategy")
    out.append(("format_repair", user))

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
    out.append(("evaluation_failure", user))

    return out


def test_no_task_specific_arithmetic_guidance_in_any_rendered_author_prompt():
    for name, rendered in _all_rendered_author_prompts():
        for phrase in BANNED_TASK_SPECIFIC:
            assert phrase not in rendered, (
                f"task-specific phrase {phrase!r} leaked into the rendered {name} prompt"
            )


def test_rendered_evaluation_failure_preserves_general_csd_mechanism_guidance():
    rendered = dict(_all_rendered_author_prompts())["evaluation_failure"]
    # General span guidance must stay (guard against over-stripping).
    assert "visible delimiter spans" in rendered
    assert "exact content" in rendered
    # The exact-token-equality prohibition must survive verbatim.
    assert "Never detect a visible delimiter with exact token equality" in rendered


def test_rendered_prompts_actually_carry_the_descriptive_prose():
    """Positive anchors -- proves each render hits the live descriptive .j2,
    so the banned-phrase check above is scanning real prompt content."""
    rendered = dict(_all_rendered_author_prompts())
    assert "What you are building" in rendered["initial"]
    assert "checks your method body against its contract statically" in rendered["verification_error"]
    assert "passed Dafny verification but failed at runtime" in rendered["runtime_error"]
    assert "Verification and compilation are separate stages" in rendered["compilation_error"]
    assert "missing the required rationale block markers" in rendered["format_repair"]
    assert "did not meet evaluation thresholds" in rendered["evaluation_failure"]


if __name__ == "__main__":
    test_no_task_specific_arithmetic_guidance_in_any_rendered_author_prompt()
    test_rendered_evaluation_failure_preserves_general_csd_mechanism_guidance()
    test_rendered_prompts_actually_carry_the_descriptive_prose()
    print("All rendered-fairness tests passed.")
