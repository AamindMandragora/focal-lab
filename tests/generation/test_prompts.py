from generation import prompts


def test_build_initial_prompt_excludes_helper_reference_by_default(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    system_prompt, user_prompt = prompts.build_initial_prompt("demo task")

    assert "CSD Helper Library — Function Reference" not in system_prompt
    assert "[BEGIN CURATED_HELPER_REFERENCE]" not in system_prompt
    assert "demo task" in user_prompt


def test_build_initial_prompt_includes_curated_helper_reference_when_enabled(monkeypatch):
    monkeypatch.setenv("CSD_HELPER_REFERENCE_MODE", "curated")
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    system_prompt, user_prompt = prompts.build_initial_prompt("demo task")

    assert "Curated Helper Mini-Reference" in system_prompt
    assert "[BEGIN CURATED_HELPER_REFERENCE]" in system_prompt
    assert "[BEGIN VERIFIED_AGENT_SYNTHESIS_MD]" not in system_prompt
    assert "demo task" in user_prompt


def test_curated_helper_reference_clarifies_guarded_constrained_calls(monkeypatch):
    monkeypatch.setenv("CSD_HELPER_REFERENCE_MODE", "curated")
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    system_prompt, _ = prompts.build_initial_prompt("demo task")

    assert "helpers.CanConstrain(generated)" in system_prompt
    assert "helpers.IsComplete(generated)" in system_prompt
    assert "AppendConstrainedOrRightDelimiterStep" in system_prompt


def test_curated_helper_reference_requires_visible_delimiter_calls(monkeypatch):
    monkeypatch.setenv("CSD_HELPER_REFERENCE_MODE", "curated")
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    system_prompt, _ = prompts.build_initial_prompt("demo task")

    assert "Explicit Delimiters For Non-Natural Runs" in system_prompt
    assert "helpers.AppendLeftDelimiter(generated, stepsLeft)" in system_prompt
    assert "helpers.AppendRightDelimiter(generated, stepsLeft)" in system_prompt
    assert "AppendConstrainedOrRightDelimiterStep" in system_prompt


def test_initial_prompt_clarifies_min_steps_ownership(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    system_prompt, user_prompt = prompts.build_initial_prompt("demo task")

    combined_prompt = system_prompt + user_prompt

    assert "helpers.MinStepsToComplete(generated)" in combined_prompt
    assert "helpers.ParserDistanceToComplete(generated)" in combined_prompt


def test_initial_prompt_rejects_old_split_channel_helpers(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    system_prompt, user_prompt = prompts.build_initial_prompt("demo task")

    combined_prompt = system_prompt + user_prompt

    assert "no soft constrained steps" in combined_prompt
    assert "no top-k constrained" in combined_prompt
    assert "no budget-aware switching" in combined_prompt


def test_initial_prompt_standardizes_delimiter_order_and_topk(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    system_prompt, user_prompt = prompts.build_initial_prompt("demo task")

    combined_prompt = system_prompt + user_prompt

    assert "AppendUnconstrainedNudgeLeftDelimiterStep" in combined_prompt
    assert "AppendConstrainedOrRightDelimiterStep" in combined_prompt
    assert "helpers.EndsWithLeftDelimiter(generated)" in combined_prompt
    assert "helpers.EndsWithRightDelimiter(generated)" in combined_prompt
    assert "no `stepsLeft -= 1`" in combined_prompt
    assert "Helper calls already consume budget" in combined_prompt
    assert "immediately above" in combined_prompt
    assert "Use only the curated helper surface" in combined_prompt


def test_initial_prompt_rejects_canconstrain_before_left_delimiter(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    system_prompt, user_prompt = prompts.build_initial_prompt("demo task")

    combined_prompt = system_prompt + user_prompt

    assert "ordinary reasoning should use" in combined_prompt
    assert "answer-opening pressure should use" in combined_prompt
    assert "all parser-handled content" in combined_prompt


def test_structure_repair_prompt_targets_unguarded_constrained_helpers(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    _, user_prompt = prompts.build_structure_repair_prompt(
        "# CSD_RATIONALE_BEGIN\n# demo\n# CSD_RATIONALE_END\nphase = 0",
        "Unguarded calls: AppendSoftConstrainedStep.",
    )

    assert "# CSD_RATIONALE_BEGIN" in user_prompt
    assert "curated helper surface" in user_prompt
    assert "AppendConstrainedOrRightDelimiterStep" in user_prompt
    assert "AppendSoftConstrainedStep" in user_prompt
    assert "Do not use removed helpers" in user_prompt


def test_structure_repair_prompt_targets_missing_delimiters(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    _, user_prompt = prompts.build_structure_repair_prompt(
        "# CSD_RATIONALE_BEGIN\n# demo\n# CSD_RATIONALE_END\nphase = 0",
        "The body must emit both LeftDelimiter and RightDelimiter. Missing: LeftDelimiter.",
    )

    assert "structural issue" in user_prompt
    assert "AppendUnconstrainedNudgeLeftDelimiterStep" in user_prompt
    assert "AppendConstrainedOrRightDelimiterStep" in user_prompt
    assert "Missing: LeftDelimiter" in user_prompt
    assert "EndsWithLeftDelimiter" in user_prompt


def test_structure_repair_prompt_standardizes_delimiter_protocol(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    _, user_prompt = prompts.build_structure_repair_prompt(
        "# CSD_RATIONALE_BEGIN\n# demo\n# CSD_RATIONALE_END\nphase = 0",
        "RightDelimiter emission must be guarded.",
    )

    assert "AppendUnconstrainedNudgeLeftDelimiterStep" in user_prompt
    assert "EndsWithLeftDelimiter" in user_prompt
    assert "EndsWithRightDelimiter" in user_prompt
    assert "Do not use removed helpers" in user_prompt


def test_verification_repair_prompt_mentions_topk_and_decreases_fixes(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    _, user_prompt = prompts.build_verification_error_prompt(
        "generated, stepsLeft = helpers.AppendTopKConstrainedStep(prompt, generated, k=5, stepsLeft=stepsLeft)",
        "decreases expression might not decrease",
    )

    assert "curated helper surface" in user_prompt
    assert "Do not use removed" in user_prompt
    assert "top-k" in user_prompt
    assert "Return only the corrected Python body" in user_prompt


def test_structure_repair_prompt_rejects_manual_stepsleft_mutation(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    _, user_prompt = prompts.build_structure_repair_prompt(
        "# CSD_RATIONALE_BEGIN\n# demo\n# CSD_RATIONALE_END\nstepsLeft -= 1",
        "Do not manually increment, decrement, or recompute `stepsLeft`.",
    )

    assert "Do not manually increment, decrement, or recompute `stepsLeft`" in user_prompt
    assert "stepsLeft -= 1" in user_prompt
    assert "Append* helper result back into" in user_prompt or "curated helper surface" in user_prompt


def test_build_initial_prompt_includes_full_helper_reference_when_enabled(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.setenv("CSD_INCLUDE_HELPER_REFERENCE_MD", "1")

    system_prompt, user_prompt = prompts.build_initial_prompt("demo task")

    assert "Verified Agent Synthesis Helper Surface" in system_prompt
    assert "[BEGIN VERIFIED_AGENT_SYNTHESIS_MD]" in system_prompt
    assert "demo task" in user_prompt
