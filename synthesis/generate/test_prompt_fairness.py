"""Tests that synthesis author prompts carry no TASK-SPECIFIC guidance (Bucket B).

Project ruling: naming CSD tools and explaining CSD mechanism is FAIR; only
guidance tied to a specific task (e.g. GSM arithmetic: "use `**`, never `^`",
"exact arithmetic") biases the controlled study and must be removed. General
span-shaping / tool-naming prose (ConstrainedStep, visible delimiter spans) is
kept on purpose -- these tests also guard against over-stripping it.
"""
from synthesis.generate.prompts import (
    INITIAL_GENERATION_PROMPT,
    EVALUATION_FAILURE_REFINEMENT_PROMPT,
)

# GSM-arithmetic-task-specific phrases that must NOT appear in any author prompt.
BANNED_TASK_SPECIFIC = [
    "exact arithmetic",
    "integer division",
    "exponentiation",
    "never use `^`",
]


def test_initial_prompt_has_no_task_specific_arithmetic_guidance():
    for phrase in BANNED_TASK_SPECIFIC:
        assert phrase not in INITIAL_GENERATION_PROMPT, (
            f"task-specific phrase {phrase!r} must be removed from INITIAL_GENERATION_PROMPT"
        )


def test_refinement_prompt_has_no_task_specific_arithmetic_guidance():
    for phrase in BANNED_TASK_SPECIFIC:
        assert phrase not in EVALUATION_FAILURE_REFINEMENT_PROMPT, (
            f"task-specific phrase {phrase!r} must be removed from EVALUATION_FAILURE_REFINEMENT_PROMPT"
        )


def test_general_csd_mechanism_guidance_is_preserved():
    # Guard against over-stripping the refinement prompt (delivered verbatim):
    # general span guidance stays, and "exact arithmetic" was generalized to
    # "exact content" rather than deleted wholesale.
    assert "visible delimiter spans" in EVALUATION_FAILURE_REFINEMENT_PROMPT
    assert "exact content" in EVALUATION_FAILURE_REFINEMENT_PROMPT


if __name__ == "__main__":
    test_initial_prompt_has_no_task_specific_arithmetic_guidance()
    test_refinement_prompt_has_no_task_specific_arithmetic_guidance()
    test_general_csd_mechanism_guidance_is_preserved()
    print("All tests passed.")
