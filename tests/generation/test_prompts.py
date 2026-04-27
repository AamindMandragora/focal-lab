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

    assert "elif phase == 2 and helpers.CanConstrain(generated):" in system_prompt
    assert "Invalid unguarded form" in system_prompt
    assert "AppendSoftConstrainedStep(prompt, generated, 0.5, stepsLeft)" in system_prompt


def test_initial_prompt_clarifies_min_steps_ownership(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    system_prompt, user_prompt = prompts.build_initial_prompt("demo task")

    combined_prompt = system_prompt + user_prompt

    assert "parser.MinStepsToComplete" in combined_prompt
    assert "helpers.MinStepsToComplete(generated)" in combined_prompt
    assert "parser.ParserDistanceToComplete(helpers.LongestValidSuffix(generated))" in combined_prompt


def test_structure_repair_prompt_targets_unguarded_constrained_helpers(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    _, user_prompt = prompts.build_structure_repair_prompt(
        "# CSD_RATIONALE_BEGIN\n# demo\n# CSD_RATIONALE_END\nphase = 0",
        "Unguarded calls: AppendSoftConstrainedStep.",
    )

    assert "elif phase == 2 and helpers.CanConstrain(generated):" in user_prompt
    assert "AppendSoftConstrainedStep" in user_prompt
    assert "Never put" in user_prompt


def test_build_initial_prompt_includes_full_helper_reference_when_enabled(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.setenv("CSD_INCLUDE_HELPER_REFERENCE_MD", "1")

    system_prompt, user_prompt = prompts.build_initial_prompt("demo task")

    assert "CSD Helper Library — Function Reference" in system_prompt
    assert "[BEGIN VERIFIED_AGENT_SYNTHESIS_MD]" in system_prompt
    assert "demo task" in user_prompt
