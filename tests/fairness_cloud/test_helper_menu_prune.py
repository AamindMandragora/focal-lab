from synthesis.generate import prompts


REMOVED = {
    "RollbackToValidPrefix",
    "RollbackToCompletePrefix",
    "RollbackConstrainedSpan",
    "RollbackAndRegenerate",
    "RollbackAndContinue",
    "SaveLogitsSnapshot",
    "RestoreLogitsSnapshot",
    "RolloutConstrainedWithPenalties",
    "SpeculativeConstrainedRollout",
    "RegenerateUnitOnCheckFailure",
}
KEPT = {
    "RollbackConstrainedSuffix",
    "RollbackConstrainedToComplete",
    "DeadEndDetection",
    "RegenerateUnitOnGroundingFailure",
    "PrefixAppearsInPrompt",
    "PrefixResemblesPromptExamples",
}


def test_removed_helpers_left_author_menu():
    rendered = prompts._build_tool_reference_block(None)
    assert not REMOVED & prompts._ALL_HELPER_NAMES
    for name in REMOVED:
        assert f"helpers.{name}(" not in rendered
        assert f"CSDHelpers.{name}(" not in rendered


def test_kept_helpers_remain_author_visible():
    rendered = prompts._build_tool_reference_block(None)
    assert KEPT <= prompts._ALL_HELPER_NAMES
    for name in KEPT:
        assert name in rendered


def test_unfair_class_membership_helper_is_absent():
    rendered = prompts._build_tool_reference_block(None)
    for name in ("PrefixMatchesPromptMoleculeClass", "SpanMatchesPromptMoleculeClass"):
        assert name not in prompts._ALL_HELPER_NAMES
        assert name not in rendered
