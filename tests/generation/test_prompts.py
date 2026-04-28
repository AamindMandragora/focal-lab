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


def test_curated_helper_reference_requires_visible_delimiter_calls(monkeypatch):
    monkeypatch.setenv("CSD_HELPER_REFERENCE_MODE", "curated")
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    system_prompt, _ = prompts.build_initial_prompt("demo task")

    assert "structural validator rejects bodies" in system_prompt
    assert "helpers.AppendLeftDelimiter(generated, stepsLeft)" in system_prompt
    assert "helpers.AppendRightDelimiter(generated, stepsLeft)" in system_prompt
    assert "helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)" in system_prompt


def test_initial_prompt_clarifies_min_steps_ownership(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    system_prompt, user_prompt = prompts.build_initial_prompt("demo task")

    combined_prompt = system_prompt + user_prompt

    assert "parser.MinStepsToComplete" in combined_prompt
    assert "helpers.MinStepsToComplete(generated)" in combined_prompt
    assert "parser.ParserDistanceToComplete(helpers.LongestValidSuffix(generated))" in combined_prompt


def test_initial_prompt_rejects_old_split_channel_helpers(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    system_prompt, user_prompt = prompts.build_initial_prompt("demo task")

    combined_prompt = system_prompt + user_prompt

    assert "no `helpers.ExpressiveStep`" in combined_prompt
    assert "no `helpers.ConstrainedAnswerStep`" in combined_prompt
    assert "no local\n  `answer` channel" in combined_prompt


def test_initial_prompt_standardizes_delimiter_order_and_topk(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    system_prompt, user_prompt = prompts.build_initial_prompt("demo task")

    combined_prompt = system_prompt + user_prompt

    assert "Emit the left delimiter before any constrained answer-token helper call" in combined_prompt
    assert "Put `RightDelimiter` emission inside a branch whose condition explicitly mentions" in combined_prompt
    assert "helpers.AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft)" in combined_prompt
    assert "Avoid float state in control flow" in combined_prompt
    assert "Never assign `pressure = 0.5`, `penalty = 0.5`" in combined_prompt
    assert "no `stepsLeft -= 1`" in combined_prompt
    assert "Helper calls already consume budget" in combined_prompt
    assert "directly above the `while` line" in combined_prompt
    assert "not indented inside" in combined_prompt
    assert "Declare all local state variables before the invariant/decreases block" in combined_prompt
    assert "Do not put `phase = ...`" in combined_prompt
    assert "Completion is a permission to close, not an instruction to close immediately" in combined_prompt
    assert "helpers.CanExtendConstrained(generated)" in combined_prompt


def test_initial_prompt_rejects_canconstrain_before_left_delimiter(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    system_prompt, user_prompt = prompts.build_initial_prompt("demo task")

    combined_prompt = system_prompt + user_prompt

    assert "Do not use `helpers.CanConstrain(generated)` to decide when to emit `LeftDelimiter`" in combined_prompt
    assert "Free-form text can accidentally have a grammar-shaped suffix" in combined_prompt
    assert "branch containing `AppendConstrainedStep`, `AppendSoftConstrainedStep`, or" in combined_prompt
    assert "must have `helpers.CanConstrain(generated)` in that branch's own" in combined_prompt
    assert "Saying \"emit the left delimiter\" in the rationale or comments is not enough" in combined_prompt
    assert "Use explicit delimiter phases" in combined_prompt
    assert "a left-delimiter phase branch emits" in combined_prompt


def test_structure_repair_prompt_targets_unguarded_constrained_helpers(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    _, user_prompt = prompts.build_structure_repair_prompt(
        "# CSD_RATIONALE_BEGIN\n# demo\n# CSD_RATIONALE_END\nphase = 0",
        "Unguarded calls: AppendSoftConstrainedStep.",
    )

    assert "# CSD_RATIONALE_BEGIN" not in user_prompt
    assert "invalid body is intentionally not shown" in user_prompt
    assert "elif phase == 2 and helpers.CanConstrain(generated):" in user_prompt
    assert "AppendSoftConstrainedStep" in user_prompt
    assert "Never put" in user_prompt
    assert "Do not repair an unguarded constrained call by setting a local boolean" in user_prompt


def test_structure_repair_prompt_targets_missing_delimiters(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    _, user_prompt = prompts.build_structure_repair_prompt(
        "# CSD_RATIONALE_BEGIN\n# demo\n# CSD_RATIONALE_END\nphase = 0",
        "The body must emit both LeftDelimiter and RightDelimiter. Missing: LeftDelimiter.",
    )

    assert "If the issue says the body must emit both delimiters" in user_prompt
    assert "helpers.AppendLeftDelimiter(generated, stepsLeft)" in user_prompt
    assert "helpers.AppendRightDelimiter(generated, stepsLeft)" in user_prompt
    assert "instead of inventing an `answer` channel" in user_prompt
    assert "Missing: LeftDelimiter" in user_prompt
    assert "use it for executable left-delimiter emission" in user_prompt


def test_structure_repair_prompt_standardizes_delimiter_protocol(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    _, user_prompt = prompts.build_structure_repair_prompt(
        "# CSD_RATIONALE_BEGIN\n# demo\n# CSD_RATIONALE_END\nphase = 0",
        "RightDelimiter emission must be guarded.",
    )

    assert "Emit the left delimiter before any constrained answer-token helper call" in user_prompt
    assert "Do not put delimiter calls after the loop" in user_prompt
    assert "Use positional helper arguments only" in user_prompt
    assert "AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft)" in user_prompt
    assert "Do not place `phase = ...`" in user_prompt


def test_verification_repair_prompt_mentions_topk_and_decreases_fixes(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    _, user_prompt = prompts.build_verification_error_prompt(
        "generated, stepsLeft = helpers.AppendTopKConstrainedStep(prompt, generated, k=5, stepsLeft=stepsLeft)",
        "decreases expression might not decrease",
    )

    assert "Remove keyword arguments from helper calls" in user_prompt
    assert "AppendTopKConstrainedStep` fails because of `1 <= k <= |lm.Tokens|`" in user_prompt
    assert "phase-only\n  terminal branches with `break`" in user_prompt
    assert "Do not manually change `stepsLeft`" in user_prompt
    assert "stepsLeft = stepsLeft - ..." in user_prompt


def test_structure_repair_prompt_rejects_manual_stepsleft_mutation(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    _, user_prompt = prompts.build_structure_repair_prompt(
        "# CSD_RATIONALE_BEGIN\n# demo\n# CSD_RATIONALE_END\nstepsLeft -= 1",
        "Do not manually increment, decrement, or recompute `stepsLeft`.",
    )

    assert "Do not manually change `stepsLeft`" in user_prompt
    assert "stepsLeft -= 1" in user_prompt
    assert "Helper calls already consume budget" in user_prompt


def test_build_initial_prompt_includes_full_helper_reference_when_enabled(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.setenv("CSD_INCLUDE_HELPER_REFERENCE_MD", "1")

    system_prompt, user_prompt = prompts.build_initial_prompt("demo task")

    assert "CSD Helper Library — Function Reference" in system_prompt
    assert "[BEGIN VERIFIED_AGENT_SYNTHESIS_MD]" in system_prompt
    assert "demo task" in user_prompt
