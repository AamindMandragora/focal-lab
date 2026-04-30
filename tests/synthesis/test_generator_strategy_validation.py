from generation.generator import StrategyGenerator
from generation.generator import StrategyGenerationError
from generation.generator import _auto_select_device


def test_auto_select_device_uses_cpu_when_cuda_gpus_are_too_full(monkeypatch):
    monkeypatch.setenv("CSD_GENERATOR_LOAD_IN_4BIT", "0")
    monkeypatch.setattr("generation.generator.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("generation.generator.torch.cuda.device_count", lambda: 2)
    monkeypatch.setattr(
        "generation.generator.torch.cuda.mem_get_info",
        lambda gpu_id: ((4 if gpu_id == 0 else 8) * 1024**3, 40 * 1024**3),
    )
    monkeypatch.setattr("generation.generator.torch.backends.mps.is_available", lambda: False)

    assert _auto_select_device() == "cpu"


def test_auto_select_device_uses_lower_threshold_for_4bit_generation(monkeypatch):
    monkeypatch.setenv("CSD_GENERATOR_LOAD_IN_4BIT", "1")
    monkeypatch.delenv("CSD_MIN_CUDA_FREE_GB", raising=False)
    monkeypatch.setattr("generation.generator.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("generation.generator.torch.cuda.device_count", lambda: 2)
    monkeypatch.setattr(
        "generation.generator.torch.cuda.mem_get_info",
        lambda gpu_id: ((5 if gpu_id == 0 else 12) * 1024**3, 40 * 1024**3),
    )

    assert _auto_select_device() == "cuda:1"


def test_auto_select_device_picks_specific_cuda_device_with_enough_memory(monkeypatch):
    monkeypatch.setenv("CSD_GENERATOR_LOAD_IN_4BIT", "0")
    monkeypatch.setattr("generation.generator.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("generation.generator.torch.cuda.device_count", lambda: 2)
    monkeypatch.setattr(
        "generation.generator.torch.cuda.mem_get_info",
        lambda gpu_id: ((10 if gpu_id == 0 else 34) * 1024**3, 40 * 1024**3),
    )

    assert _auto_select_device() == "cuda:1"


def test_strategy_generator_treats_auto_device_string_as_auto(monkeypatch):
    monkeypatch.setattr("generation.generator._auto_select_device", lambda: "cpu")
    monkeypatch.setattr(StrategyGenerator, "_load_template", lambda self: "")

    generator = StrategyGenerator(device="auto")

    assert generator.device == "cpu"


def test_normalize_rationale_block_comments_plain_lines():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
This rationale line is not commented.
# CSD_RATIONALE_END
flag = False
"""

    normalized = generator._normalize_rationale_block(strategy)

    assert "# This rationale line is not commented." in normalized


def test_generate_valid_strategy_records_rejected_candidate_diagnostics():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    generator.max_new_tokens = 192
    generator.temperature = 0.7
    generator._generate_text = lambda *args, **kwargs: "not a strategy body"
    generator._extract_strategy = lambda raw: raw
    generator._ensure_rationale_block = lambda strategy: (_ for _ in ()).throw(
        ValueError("missing rationale")
    )

    try:
        generator._generate_valid_strategy(
            "system",
            "user",
            failure_context="Qwen did not produce a usable initial strategy",
        )
    except StrategyGenerationError:
        pass
    else:
        raise AssertionError("expected generation to fail")

    assert len(generator.last_generation_diagnostics) == generator.SEARCH_ATTEMPTS
    first = generator.last_generation_diagnostics[0]
    assert first["raw_output"] == "not a strategy body"
    assert first["raw_output_empty"] is False
    assert first["extracted_strategy"] == "not a strategy body"
    assert first["accepted"] is False
    assert first["issue"] == "missing rationale"


def test_structural_issue_rejects_unknown_helper_methods():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
constrained_count = 0
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    next_token = eosToken
    new_steps = stepsLeft
    if phase == 0:
        next_token, new_steps = helpers.UnconstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        if helpers.VariableFound(generated):
            phase = 1
    elif phase == 1:
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 2
    elif phase == 2 and not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        next_token, new_steps = helpers.ConstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        constrained_count = constrained_count + 1
    elif phase == 2:
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, RightDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "VariableFound" in issue


def test_structural_issue_rejects_parser_methods_called_on_helpers():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constraint_mode = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0 and reasoning_tokens < 1:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        reasoning_tokens = reasoning_tokens + 1
        if helpers.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            phase = 1
    elif phase == 1:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 2
    elif phase == 2 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constraint_mode = constraint_mode + 1
    elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "parser methods, not helper methods" in issue
    assert "IsCompletePrefix" in issue


def test_structural_issue_rejects_unknown_parser_methods():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
constrained_count = 0
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    next_token = eosToken
    new_steps = stepsLeft
    if phase == 0:
        next_token, new_steps = helpers.UnconstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        if parser.PotentialConstrainedSegment(generated):
            phase = 1
    elif phase == 1:
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 2
    elif phase == 2 and not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        next_token, new_steps = helpers.ConstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        constrained_count = constrained_count + 1
    elif phase == 2:
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, RightDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "PotentialConstrainedSegment" in issue


def test_structural_issue_rejects_parser_methods_on_generated():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
constrained_count = 0
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    next_token = eosToken
    new_steps = stepsLeft
    if phase == 0:
        next_token, new_steps = helpers.UnconstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        if parser.IsValidPrefix(generated):
            phase = 1
    elif phase == 1:
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 2
    elif phase == 2 and not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        next_token, new_steps = helpers.ConstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        constrained_count = constrained_count + 1
    elif phase == 2:
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, RightDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "generated" in issue
    assert "IsValidPrefix" in issue


def test_structural_issue_rejects_old_api_calls():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
answer_tokens = 0
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 2:
    next_token = eosToken
    new_steps = stepsLeft
    if phase == 0:
        next_token, new_steps = helpers.ExpressiveStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 1
    else:
        next_token, new_steps = helpers.ConstrainedAnswerStep(prompt, generated, answer, stepsLeft)
        answer = answer + [next_token]
        stepsLeft = new_steps
        answer_tokens = answer_tokens + 1
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "old" in issue.lower() or "replaced" in issue.lower()


def test_structural_issue_rejects_repair_salvage_helpers():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
answer_tokens = 0
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        answer_tokens = answer_tokens + 1
    elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated = helpers.RollbackToValidPrefix(generated)
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "repair/salvage" in issue
    assert "RollbackToValidPrefix" in issue


def test_structural_issue_accepts_valid_new_api_strategy():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# Hybrid strategy using the new suffix-based API.
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constrained_count = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    next_token = eosToken
    new_steps = stepsLeft
    if phase == 0 and reasoning_tokens < 3 and stepsLeft > 2:
        next_token, new_steps = helpers.UnconstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        reasoning_tokens = reasoning_tokens + 1
        if reasoning_tokens >= 3:
            phase = 1
    elif phase == 1:
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 2
    elif phase == 2 and not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        next_token, new_steps = helpers.ConstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        constrained_count = constrained_count + 1
    elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, RightDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is None


def test_structural_issue_rejects_missing_standard_loop_invariants():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constrained_count = 0
while stepsLeft > 0 and phase < 3:
    if phase == 0 and reasoning_tokens < 2 and stepsLeft > 2:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        reasoning_tokens = reasoning_tokens + 1
        if reasoning_tokens >= 2:
            phase = 1
    elif phase == 1:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 2
    elif phase == 2 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constrained_count = constrained_count + 1
    elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "standard loop invariant" in issue or "standard decreases clause" in issue


def test_structural_issue_accepts_append_helper_strategy():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# Hybrid strategy using the append-style helper wrappers.
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constraint_mode = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0 and reasoning_tokens < 2 and stepsLeft > 2:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        reasoning_tokens = reasoning_tokens + 1
        if reasoning_tokens >= 2:
            phase = 1
    elif phase == 1:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 2
    elif phase == 2 and helpers.CanConstrain(generated) and not helpers.IsComplete(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constraint_mode = constraint_mode + 1
    elif phase == 2 and helpers.IsComplete(generated):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is None


def test_structural_issue_rejects_phase_only_top_level_loop_branch():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# Bad branch only mutates phase inside a decreases loop.
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constraint_mode = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0 and reasoning_tokens < 2 and stepsLeft > 2:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        reasoning_tokens = reasoning_tokens + 1
    elif phase == 0:
        phase = 1
    elif phase == 1:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 2
    elif phase == 2 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constraint_mode = constraint_mode + 1
    elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "must either consume a helper step or `break`" in issue


def test_structural_issue_rejects_constrained_body_without_delimiters():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constraint_mode = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 2:
    if phase == 0 and reasoning_tokens < 2 and stepsLeft > 1:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        reasoning_tokens = reasoning_tokens + 1
        if reasoning_tokens >= 2:
            phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        if constraint_mode == 0:
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        else:
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constraint_mode = constraint_mode + 1
    else:
        phase = 2
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "emit both LeftDelimiter and RightDelimiter" in issue


def test_structural_issue_allows_reasoning_after_right_delimiter_for_interleaving():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
answer_steps = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 4:
    if phase == 0:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        phase = 1
    elif phase == 1:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 2
    elif phase == 2 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        answer_steps = answer_steps + 1
    elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
    elif phase == 3:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is None


def test_structural_issue_rejects_fixed_phase_quota_strategy():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_steps = 0
answer_steps = 0
min_reasoning_steps = 1
min_answer_steps = 2
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0 and reasoning_steps < min_reasoning_steps:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        reasoning_steps = reasoning_steps + 1
        phase = 1
    elif phase == 1:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 2
    elif phase == 2 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        answer_steps = answer_steps + 1
    elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "fixed phase-quota constants" in issue


def test_structural_issue_rejects_nontrivial_fixed_phase_quotas():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_steps = 0
scratch_steps = 0
answer_steps = 0
min_reason_steps = 2
min_scratch_steps = 3
min_final_steps = 2
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 5:
    if phase == 0 and reasoning_steps < min_reason_steps:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        reasoning_steps = reasoning_steps + 1
    elif phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        scratch_steps = scratch_steps + 1
    elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 2
    elif phase == 2:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        phase = 3
    elif phase == 3:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 4
    elif phase == 4 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        answer_steps = answer_steps + 1
    elif phase == 4 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 5
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "fixed phase-quota constants" in issue


def test_structural_issue_rejects_reason_limit_phase_quota():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reason_steps = 0
answer_tokens = 0
reason_limit = 2
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0 and reason_steps < reason_limit:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        reason_steps = reason_steps + 1
    elif phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        answer_tokens = answer_tokens + 1
    elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 2
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "fixed phase-quota constants" in issue


def test_structural_issue_rejects_negative_indexing():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
answer_steps = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        if len(generated) > 0 and generated[-1] == ".":
            phase = 1
    elif phase == 1:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 2
    elif phase == 2 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        answer_steps = answer_steps + 1
    elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "negative list indexing" in issue


def test_structural_issue_rejects_nonliteral_bias_left_delimiter_arg():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
next_token = eosToken
new_steps = stepsLeft
biasStrength = 3
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        next_token, new_steps = helpers.UnconstrainedStep(prompt, generated, biasStrength, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        if next_token == LeftDelimiter or next_token == SpacedLeftDelimiter:
            phase = 1
    elif phase == 1 and (helpers.CanConstrain(generated) or parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))):
        next_token, new_steps = helpers.ConstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        if next_token == RightDelimiter:
            phase = 2
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "literal positive float bias" in issue


def test_structural_issue_rejects_incomplete_tuple_helper_assignment():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
answer_steps = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        answer_steps = answer_steps + 1
    elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendConstrainedStep
        phase = 2
    elif phase == 2:
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "Tuple assignment is only supported" in issue


def test_structural_issue_rejects_bare_forced_token_step_calls():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constrained_count = 0
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    next_token = eosToken
    new_steps = stepsLeft
    if phase == 0 and reasoning_tokens < 2:
        next_token, new_steps = helpers.UnconstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        reasoning_tokens = reasoning_tokens + 1
        if reasoning_tokens >= 2:
            phase = 1
    elif phase == 1:
        helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)
        generated = generated + [LeftDelimiter]
        phase = 2
    elif phase == 2 and not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        next_token, new_steps = helpers.ConstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        constrained_count = constrained_count + 1
    elif phase == 2:
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, RightDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "ForcedTokenStep" in issue
    assert "bare statement" in issue


def test_structural_issue_rejects_bare_append_helper_calls():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constraint_mode = 0
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0 and reasoning_tokens < 1:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        reasoning_tokens = reasoning_tokens + 1
        phase = 1
    elif phase == 1:
        helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 2
    elif phase == 2 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constraint_mode = constraint_mode + 1
    else:
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "Append*" in issue
    assert "must not be used as bare statements" in issue


def test_structural_issue_rejects_unguarded_append_constrained_step():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
budget_for_reasoning = stepsLeft // 3
budget_for_constrained = stepsLeft - budget_for_reasoning
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0 and len(generated) < budget_for_reasoning:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        if len(generated) >= budget_for_reasoning:
            phase = 1
    elif phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 2
    elif phase == 1:
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
    elif phase == 2 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
    else:
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "Every constrained helper call" in issue
    assert "AppendConstrainedStep" in issue


def test_structural_issue_rejects_unbounded_while_loops():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constrained_count = 0
delim_phase = 0
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
    next_token = eosToken
    new_steps = stepsLeft
    if delim_phase == 0:
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        delim_phase = 1
    elif delim_phase == 1:
        next_token, new_steps = helpers.ConstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        constrained_count = constrained_count + 1
        if constrained_count > 2:
            delim_phase = 2
    else:
        next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, RightDelimiter, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        phase = 1
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "budget-bounded" in issue


def test_structural_issue_rejects_string_methods_on_generated():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
repair_mode = False
delimiter_round = 0
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0 and not generated.endswith(LeftDelimiter):
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        delimiter_round = delimiter_round + 1
        if delimiter_round > 1:
            phase = 2
    elif phase == 2 and not generated.startswith(LeftDelimiter):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        repair_mode = True
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "generated" in issue
    assert "startswith" in issue
    assert "endswith" in issue


def test_structural_issue_rejects_joined_generated_string():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
closed_spans = 0
seen = 0
joined = ""
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    joined = "".join(generated[-12:])
    if phase == 0 and "answer" in joined.lower():
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 0:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        seen = seen + 1
    elif phase == 1 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
    elif phase == 1 and helpers.IsComplete(generated):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        closed_spans = closed_spans + 1
        phase = 2
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "join" in issue
    assert "generated" in issue


def test_structural_issue_rejects_string_methods_on_longest_valid_suffix():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
repair_mode = False
delimiter_round = 0
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        delimiter_round = delimiter_round + 1
        if delimiter_round > 1:
            phase = 2
    elif phase == 2 and not helpers.LongestValidSuffix(generated).endswith(RightDelimiter):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        repair_mode = True
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "LongestValidSuffix" in issue
    assert "endswith" in issue


def test_structural_issue_rejects_append_helper_assigned_to_next_token():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constraint_mode = 0
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0 and reasoning_tokens < 1:
        next_token, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        reasoning_tokens = reasoning_tokens + 1
        phase = 1
    elif phase == 1:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 2
    elif phase == 2 and helpers.CanConstrain(generated):
        next_token, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constraint_mode = constraint_mode + 1
    else:
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "Append*" in issue
    assert "next_token" in issue


def test_structural_issue_rejects_constrained_calls_before_left_delimiter():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constraint_mode = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constraint_mode = constraint_mode + 1
        phase = 1
    elif phase == 1:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 2
    elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "after executable LeftDelimiter" in issue


def test_structural_issue_rejects_delimiter_appended_after_loop():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constraint_mode = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 2:
    if phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constraint_mode = constraint_mode + 1
    else:
        break
if parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
    generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "inside a budget-bounded decoding while loop" in issue


def test_structural_issue_rejects_right_delimiter_without_completion_guard():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constraint_mode = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constraint_mode = constraint_mode + 1
        phase = 2
    else:
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "parser.IsCompletePrefix" in issue
    assert "RightDelimiter" in issue


def test_structural_issue_rejects_helper_keyword_arguments():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constraint_mode = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt=prompt, prefix=generated, stepsLeft=stepsLeft)
        constraint_mode = constraint_mode + 1
        phase = 2
    elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "positional arguments" in issue
    assert "AppendConstrainedStep" in issue


def test_structural_issue_rejects_topk_k_larger_than_one():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constraint_mode = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendTopKConstrainedStep(prompt, generated, 5, stepsLeft)
        constraint_mode = constraint_mode + 1
        phase = 2
    elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "helper methods that do not exist" in issue
    assert "AppendTopKConstrainedStep" in issue


def test_structural_issue_rejects_manual_stepsleft_mutation():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constraint_mode = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 4:
    if phase == 0:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        reasoning_tokens = reasoning_tokens + 1
        phase = 1
    elif phase == 1:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 2
    elif phase == 2 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constraint_mode = constraint_mode + 1
    elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
    stepsLeft -= 1
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "Do not manually increment, decrement, or recompute `stepsLeft`" in issue


def test_structural_issue_rejects_open_constrain_branch_before_completion_check(monkeypatch):
    monkeypatch.setenv("CSD_STRICT_COMPLETE_ORDER", "1")
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
answer_steps = 0
constraint_mode = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        answer_steps = answer_steps + 1
    elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 2
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "check `helpers.IsComplete" in issue
    assert "before an open-ended" in issue


def test_structural_issue_allows_constrain_branch_guarded_by_not_complete():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
answer_steps = 0
constraint_mode = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated) and not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        answer_steps = answer_steps + 1
    elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 2
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is None


def test_structural_issue_rejects_mutable_float_control_state():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
constraint_strength = 0.5
constraint_mode = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        constraint_strength = constraint_strength - 0.1
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constraint_mode = constraint_mode + 1
        phase = 2
    elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "mutable float" in issue


def test_structural_issue_rejects_return_print_and_remaining_steps():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
constraint_mode = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constraint_mode = constraint_mode + 1
        phase = 2
    elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
remainingSteps = stepsLeft
print("done")
return generated, remainingSteps
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "remainingSteps" in issue


def test_structural_issue_rejects_invariants_inside_loop_body():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reasoning_tokens = 0
while stepsLeft > 0 and phase < 3:
    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    if phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        reasoning_tokens = reasoning_tokens + 1
    elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 2
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "immediately above each decoding `while` line" in issue


def test_structural_issue_rejects_stray_expression_statement():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
reason_signal = 0
inside_span = 0
constraint_mode = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 4:
    if phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constraint_mode = constraint_mode + 1
        phase = 2
    elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
    elif phase == 3:
        inside_span
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "stray expression statement" in issue


def test_structural_issue_rejects_nested_nonprogress_branch_inside_decreases_loop():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
inside_span = 0
reasoning_tokens = 0
constraint_mode = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        if reasoning_tokens < 1:
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            reasoning_tokens = reasoning_tokens + 1
        else:
            phase = 1
    elif phase == 1:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 2
    elif phase == 2 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constraint_mode = constraint_mode + 1
    elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "must either consume a helper step or `break`" in issue


def test_structural_issue_rejects_dafny_reserved_local_names():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
opened = 0
constraint_mode = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        constraint_mode = constraint_mode + 1
        phase = 2
    elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 3
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "reserved identifiers" in issue
    assert "opened" in issue


def test_structural_issue_rejects_natural_delimiter_helpers_in_spider_single_span_mode(monkeypatch):
    monkeypatch.setenv("CSD_SPIDER_FORCE_SINGLE_SQL_SPAN", "1")
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
closed_spans = 0
freeform_steps = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        if helpers.EndsWithLeftDelimiter(generated):
            phase = 1
    elif phase == 1 and (helpers.CanConstrain(generated) or helpers.IsComplete(generated)):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        if helpers.EndsWithRightDelimiter(generated):
            closed_spans = closed_spans + 1
            phase = 2
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "CSD_SPIDER_FORCE_SINGLE_SQL_SPAN=1" in issue
    assert "natural LEFT-delimiter helpers" in issue


def test_structural_issue_rejects_long_freeform_threshold_in_spider_single_span_mode(monkeypatch):
    monkeypatch.setenv("CSD_SPIDER_FORCE_SINGLE_SQL_SPAN", "1")
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
freeformCount = 0
freeformLimit = 8
sql_tokens = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0 and freeformCount < freeformLimit:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        freeformCount = freeformCount + 1
    elif phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and (helpers.CanConstrain(generated) or helpers.IsComplete(generated)):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        if helpers.EndsWithRightDelimiter(generated):
            phase = 2
        else:
            sql_tokens = sql_tokens + 1
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "keep unconstrained prelude short" in issue


def test_structural_issue_accepts_explicit_single_sql_span_mode_for_spider(monkeypatch):
    monkeypatch.setenv("CSD_SPIDER_FORCE_SINGLE_SQL_SPAN", "1")
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
freeformCount = 0
sql_tokens = 0
opened_span = False
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0 and freeformCount < 2:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        freeformCount = freeformCount + 1
    elif phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        opened_span = True
        phase = 1
    elif phase == 1 and (helpers.CanConstrain(generated) or helpers.IsComplete(generated)):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        if helpers.EndsWithRightDelimiter(generated):
            phase = 2
        else:
            sql_tokens = sql_tokens + 1
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is None


def test_structural_issue_rejects_missing_right_closure_helper_in_spider_single_span_mode(monkeypatch):
    monkeypatch.setenv("CSD_SPIDER_FORCE_SINGLE_SQL_SPAN", "1")
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
freeformCount = 0
sql_tokens = 0
opened_span = False
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0 and freeformCount < 2:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        freeformCount = freeformCount + 1
    elif phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        opened_span = True
        phase = 1
    elif phase == 1 and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        sql_tokens = sql_tokens + 1
    elif phase == 1 and helpers.IsComplete(generated):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = 2
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "right-closure-capable helper" in issue


def test_structural_issue_rejects_unconstrained_steps_in_spider_start_at_span_mode(monkeypatch):
    monkeypatch.setenv("CSD_SPIDER_FORCE_SINGLE_SQL_SPAN", "1")
    monkeypatch.setenv("CSD_SPIDER_FORCE_SPAN_AT_START", "1")
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
sql_tokens = 0
opened_span = False
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        phase = 1
    elif phase == 1:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        opened_span = True
        phase = 2
    elif phase == 2 and (helpers.CanConstrain(generated) or helpers.IsComplete(generated)):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        if helpers.EndsWithRightDelimiter(generated):
            break
        else:
            sql_tokens = sql_tokens + 1
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "CSD_SPIDER_FORCE_SPAN_AT_START=1" in issue
    assert "Start directly with `helpers.AppendLeftDelimiter(...)`" in issue


def test_structural_issue_accepts_spider_start_at_span_mode(monkeypatch):
    monkeypatch.setenv("CSD_SPIDER_FORCE_SINGLE_SQL_SPAN", "1")
    monkeypatch.setenv("CSD_SPIDER_FORCE_SPAN_AT_START", "1")
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
sql_tokens = 0
opened_span = False
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        opened_span = True
        phase = 1
    elif phase == 1 and (helpers.CanConstrain(generated) or helpers.IsComplete(generated)):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        if helpers.EndsWithRightDelimiter(generated):
            phase = 2
            break
        else:
            sql_tokens = sql_tokens + 1
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is None


def test_structural_issue_rejects_nonprompt_first_arg_for_prompted_helpers():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
opened_span = False
sql_tokens = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        opened_span = True
        phase = 1
    elif phase == 1 and (helpers.CanConstrain(generated) or helpers.IsComplete(generated)):
        generated, stepsLeft = helpers.AppendConstrainedStep("", generated, stepsLeft)
        if helpers.EndsWithRightDelimiter(generated):
            phase = 2
        else:
            sql_tokens = sql_tokens + 1
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "must pass the function input `prompt`" in issue


def test_structural_issue_rejects_spider_premature_not_can_constrain_break(monkeypatch):
    monkeypatch.setenv("CSD_SPIDER_FORCE_SINGLE_SQL_SPAN", "1")
    monkeypatch.setenv("CSD_SPIDER_FORCE_SPAN_AT_START", "1")
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
sql_tokens = 0
opened_span = False
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        opened_span = True
        phase = 1
    elif phase == 1:
        if helpers.EndsWithRightDelimiter(generated):
            phase = 2
            break
        elif not helpers.CanConstrain(generated):
            break
        else:
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            if helpers.EndsWithRightDelimiter(generated):
                phase = 2
                break
            else:
                sql_tokens = sql_tokens + 1
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "CanConstrain" in issue
    assert "unclosed" in issue


def test_structural_issue_rejects_spider_nonterminal_right_delimiter_branch(monkeypatch):
    monkeypatch.setenv("CSD_SPIDER_FORCE_SINGLE_SQL_SPAN", "1")
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
sql_tokens = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3:
    if phase == 0:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = 1
    elif phase == 1 and (helpers.CanConstrain(generated) or helpers.IsComplete(generated)):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        if helpers.IsComplete(generated):
            generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        elif helpers.EndsWithRightDelimiter(generated):
            phase = 2
        else:
            sql_tokens = sql_tokens + 1
    elif phase == 2:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
    else:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "once `helpers.EndsWithRightDelimiter(generated)` is true" in issue
    assert "terminate immediately" in issue
