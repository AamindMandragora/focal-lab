from pathlib import Path

from synthesis.generate import prompts


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_strategy_guidance_phrases_are_absent_from_tool_reference():
    rendered = prompts._build_tool_reference_block(None)
    for phrase in ("When to use:", "How to use:", "Example call shape:", "Suggested starting values:"):
        assert phrase not in rendered


def test_grounding_mechanism_remains_without_worked_promotion():
    rendered = prompts._build_tool_reference_block(None)
    examples = prompts._build_verified_examples_block(None)
    assert "RegenerateUnitOnGroundingFailure(" in rendered
    assert "RegenerateUnitOnGroundingFailure" not in examples
    assert "RegenerateUnitOnCheckFailure" not in examples
    assert "// Grounded-unit constrained CSD." not in prompts._VERIFIED_EXAMPLE_PREFIXES


def test_unit_rewind_hint_is_removed():
    source = (REPO_ROOT / "synthesis/evaluate/feedback_loop.py").read_text()
    assert "_unit_rewind_hint" not in source
