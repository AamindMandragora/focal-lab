import ast

import pytest

from generation.generator import StrategyGenerator
from generation.generator import StrategyGenerationError
from generation.generator import _auto_select_device


def test_ensure_nontrivial_strategy_accepts_non_house_style_body_for_gpt54():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    generator.model_name = "gpt-5.4"
    generator.strategy_language = "python"
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
# CSD_PROOF_SKETCH_BEGIN
# test
# CSD_PROOF_SKETCH_END
phase = 0
generated = generated + ["free"]
"""

    strict_issue = generator._structural_issue(strategy)
    accepted = generator._ensure_nontrivial_strategy(strategy, max_repairs=0)

    assert "while loop" in strict_issue
    assert accepted == strategy
    assert generator.last_structure_validation_summary["style_validation_enforced"] is False
    assert generator.last_structure_validation_summary["exploration_first_generation"] is True


def test_ensure_nontrivial_strategy_keeps_strict_style_gate_for_non_gpt54():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    generator.model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    generator.strategy_language = "python"
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
# CSD_PROOF_SKETCH_BEGIN
# test
# CSD_PROOF_SKETCH_END
phase = 0
generated = generated + ["free"]
"""

    with pytest.raises(ValueError, match="structurally invalid"):
        generator._ensure_nontrivial_strategy(strategy, max_repairs=0)


def test_structural_issue_skips_forced_delimiter_requirement_in_natural_mode(monkeypatch):
    generator = StrategyGenerator.__new__(StrategyGenerator)
    monkeypatch.setenv("CSD_REQUIRE_NATURAL_DELIMITERS", "1")
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = "reason"
inside_span = False
closed_spans = 0
constrained_steps = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0:
    if inside_span:
        if helpers.EndsWithRightDelimiter(generated):
            inside_span = False
            closed_spans = closed_spans + 1
            break
        if helpers.IsComplete(generated) or helpers.CanConstrain(generated):
            next_token, stepsLeft = helpers.ConstrainedStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            constrained_steps = constrained_steps + 1
            if helpers.IsRightDelimiterToken(next_token):
                inside_span = False
                closed_spans = closed_spans + 1
                break
        else:
            break
    else:
        next_token, stepsLeft = helpers.UnconstrainedStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        if helpers.IsLeftDelimiterToken(next_token):
            inside_span = True
            phase = "answer"
        else:
            phase = "reason"
"""

    issue = generator._structural_issue(strategy)

    assert issue is None or "must emit both LeftDelimiter and RightDelimiter" not in issue


def test_structural_issue_accepts_split_prefix_gsm_policy_in_natural_mode(monkeypatch):
    generator = StrategyGenerator.__new__(StrategyGenerator)
    monkeypatch.setenv("CSD_REQUIRE_NATURAL_DELIMITERS", "1")
    strategy = """# CSD_RATIONALE_BEGIN
# split-prefix demo
# CSD_RATIONALE_END
# CSD_PROOF_SKETCH_BEGIN
# split-prefix demo
# CSD_PROOF_SKETCH_END
inside_constrained = False
current_constrained = []
saw_close_context = False
valid_token_groups = []
narrow_threshold = 12
stable_prefix = []
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0:
    stepsLeftBeforeIteration = stepsLeft
    if not inside_constrained:
        prev_token, found_prev = helpers.LastTokenBefore(generated, ">>")
        saw_close_context = found_prev
        if saw_close_context:
            generated, inside_constrained, current_constrained, stepsLeft = helpers.OpenConstrainedSpan(generated, stepsLeft)
            saw_close_context = False
        else:
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
    else:
        if parser.IsCompletePrefix(current_constrained):
            generated, inside_constrained, current_constrained, stepsLeft = helpers.CloseConstrainedSpan(generated, current_constrained, stepsLeft)
            break
        else:
            stable_prefix = generated
            next_token, stepsLeft = helpers.AdaptiveConstrainedStep(prompt, stable_prefix, current_constrained, valid_token_groups, 4.0, narrow_threshold, eosToken, stepsLeft)
            if next_token == eosToken:
                break
            generated, inside_constrained, current_constrained = helpers.AppendConstrainedToken(generated, current_constrained, next_token)
    if stepsLeft >= stepsLeftBeforeIteration:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is None


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


def test_structural_issue_rejects_budget_only_opening():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
opened = False
closed_spans = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0:
    if not opened and stepsLeft <= 3:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        opened = True
    elif opened and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
    elif opened and helpers.IsComplete(generated):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        closed_spans = closed_spans + 1
        break
    else:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "based only on remaining token budget" in issue


def test_structural_issue_allows_budget_open_when_gated_by_intent_state():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
inside_span = False
final_ready = 1
closed_spans = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0:
    if not inside_span and final_ready == 1 and closed_spans == 0 and stepsLeft <= 3:
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        inside_span = True
    elif inside_span and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
    elif inside_span and helpers.IsComplete(generated):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        closed_spans = closed_spans + 1
        break
    else:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
"""

    issue = generator._structural_issue(strategy)

    assert issue is None


def test_autofix_rewrites_complete_branch_append_constrained_step_to_close():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = 0
inside_span = True
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0:
    if inside_span and helpers.IsComplete(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
    else:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
"""

    fixed = generator._autofix_python_strategy(strategy)

    assert "helpers.AppendRightDelimiter(generated, stepsLeft)" in fixed
    assert "helpers.AppendConstrainedStep(prompt, generated, stepsLeft)" not in fixed


def test_ensure_nontrivial_strategy_revalidates_after_autofix(monkeypatch):
    generator = StrategyGenerator.__new__(StrategyGenerator)
    generator.strategy_language = "python"
    generator.last_structure_repair_trace = []
    generator.last_structure_validation_summary = {}
    generator.last_rationale_repair_count = 0

    repaired_strategy = """# CSD_RATIONALE_BEGIN
# repaired
# CSD_RATIONALE_END
phase = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0:
    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
    break
"""

    def fake_structural_issue(body: str) -> str | None:
        if body == "original":
            return None
        if body == "broken_after_autofix":
            return "autofix introduced a structural issue"
        if body == repaired_strategy:
            return None
        return None

    monkeypatch.setattr(generator, "_structural_issue", fake_structural_issue)
    monkeypatch.setattr(
        generator,
        "_autofix_python_strategy",
        lambda body: "broken_after_autofix" if body == "original" else body,
    )
    monkeypatch.setattr(generator, "_ensure_rationale_block", lambda body, max_repairs=2: body)
    monkeypatch.setattr(generator, "_extract_strategy", lambda body: body)
    monkeypatch.setattr(generator, "_generate_text", lambda system_prompt, user_prompt, **_: repaired_strategy)
    monkeypatch.setattr(
        "generation.generator.build_structure_repair_prompt",
        lambda previous_strategy, issue, strategy_language="python": ("system", "user"),
    )

    fixed = generator._ensure_nontrivial_strategy("original", max_repairs=1)

    assert fixed == repaired_strategy
    assert generator.last_structure_validation_summary["structural_repairs"] == 1
    assert generator.last_structure_validation_summary["autofix_passes"] >= 1


def test_autofix_trims_truncated_tail_until_parseable():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = "reason"
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0:
    if phase == "reason":
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
    elif helpers.ValidContinuationCount(generated) <=
"""

    fixed = generator._autofix_python_strategy(strategy)
    wrapped = "def _strategy():\n" + "\n".join(f"    {line}" for line in fixed.splitlines())
    ast.parse(wrapped)
    assert "helpers.ValidContinuationCount(generated) <=" not in fixed


def test_autofix_rewrites_mixed_complete_or_canconstrain_guard_to_canconstrain():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
inside_span = True
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0:
    if inside_span and (helpers.IsComplete(generated) or helpers.CanConstrain(generated)):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
    else:
        break
"""

    fixed = generator._autofix_python_strategy(strategy)

    assert "helpers.AppendConstrainedStep(prompt, generated, stepsLeft)" in fixed
    assert "helpers.IsComplete(generated) or helpers.CanConstrain(generated)" not in fixed
    assert "helpers.CanConstrain(generated)" in fixed


def test_autofix_uses_snapshot_guard_instead_of_forcing_else_break():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = "reason"
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0:
    if phase == "reason":
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
    else:
        phase = "done"
"""

    fixed = generator._autofix_python_strategy(strategy)

    assert "else:" in fixed
    assert "phase = \"done\"" in fixed
    assert "stepsLeftBeforeIteration = stepsLeft" in fixed
    assert "if stepsLeft >= stepsLeftBeforeIteration:" in fixed
    assert "phase = \"done\"\n        break" not in fixed


def test_autofix_rewrites_continue_to_break():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = "reason"
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0:
    if phase == "reason":
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        continue
    else:
        break
"""

    fixed = generator._autofix_python_strategy(strategy)

    assert "continue" not in fixed
    assert "break" in fixed


def test_autofix_rewrites_stepsleft_only_append_assignment():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = "reason"
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0:
    if phase == "reason":
        stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
    else:
        break
"""

    fixed = generator._autofix_python_strategy(strategy)

    assert "\n        stepsLeft = helpers.AppendUnconstrainedStep" not in fixed
    assert "generated, stepsLeft = helpers.AppendUnconstrainedStep" in fixed


def test_structural_issue_rejects_dangling_if_chain_without_fallback():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = "open"
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0:
    if phase == "open":
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = "answer"
    elif phase == "answer" and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        phase = "close"
    elif phase == "close" and helpers.IsComplete(generated):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "top-level `if/elif` chains must have an explicit final `else`" in issue


def test_structural_issue_rejects_dangling_if_chain_with_trailing_top_level_break():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = "open"
closed_spans = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0:
    if phase == "open":
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = "answer"
    elif phase == "answer" and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        phase = "close"
    elif phase == "close" and helpers.IsComplete(generated):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        closed_spans = closed_spans + 1
    break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "bare top-level `break`" in issue


def test_structural_issue_rejects_top_level_break_in_decoding_loop():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = "open"
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0:
    if phase == "open":
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = "answer"
    elif phase == "answer" and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        phase = "close"
    elif phase == "close" and helpers.IsComplete(generated):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        phase = "done"
    break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "bare top-level `break`" in issue


def test_structural_issue_allows_dangling_if_chain_with_step_snapshot_guard():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = "open"
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0:
    stepsLeftBeforeIteration = stepsLeft
    if phase == "open":
        generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
        phase = "answer"
    elif phase == "answer" and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
    elif phase == "answer" and helpers.IsComplete(generated):
        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        break
    elif phase == "answer":
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
    if stepsLeft >= stepsLeftBeforeIteration:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is None


def test_autofix_inserts_decreases_guard_for_stepsleft_loop():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = "reason"
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0:
    if phase == "reason":
        phase = "answer"
    elif phase == "answer":
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
"""

    fixed = generator._autofix_python_strategy(strategy)

    assert "stepsLeftBeforeIteration = stepsLeft" in fixed
    assert "if stepsLeft >= stepsLeftBeforeIteration:" in fixed


def test_autofix_does_not_duplicate_decreases_guard_when_present():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = "reason"
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0:
    stepsLeftBeforeIteration = stepsLeft
    if phase == "reason":
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
    if stepsLeft >= stepsLeftBeforeIteration:
        break
"""

    fixed = generator._autofix_python_strategy(strategy)

    assert fixed.count("stepsLeftBeforeIteration = stepsLeft") == 1
    assert fixed.count("if stepsLeft >= stepsLeftBeforeIteration:") == 1


def test_autofix_adds_else_break_for_dangling_if_chain():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = "open"
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0:
    if phase == "open":
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        phase = "answer"
    elif phase == "answer" and helpers.CanConstrain(generated):
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
"""

    fixed = generator._autofix_python_strategy(strategy)

    assert "else:" in fixed
    assert "\n        break" in fixed
    wrapped = "def _strategy():\n" + "\n".join(f"    {line}" for line in fixed.splitlines())
    ast.parse(wrapped)


def test_autofix_rewrites_suffix_string_cue_scan_patterns():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = "reason"
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0:
    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
    suffix = "".join(helpers.LongestValidSuffix(generated)).lower()
    if ("final" in suffix) or ("answer" in suffix):
        phase = "done"
"""

    fixed = generator._autofix_python_strategy(strategy)

    assert '".join(helpers.LongestValidSuffix(generated)).lower()' not in fixed
    assert "in suffix" not in fixed
    assert "if False:" in fixed


def test_autofix_does_not_add_else_break_to_snapshot_guard():
    generator = StrategyGenerator.__new__(StrategyGenerator)
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0:
    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
    if stepsLeft >= stepsLeftBeforeIteration:
        break
"""

    fixed = generator._autofix_python_strategy(strategy)

    assert "if stepsLeft >= stepsLeftBeforeIteration:\n        break\n    else:" not in fixed


def test_structural_issue_rejects_natural_phase_break_before_opening_helper(monkeypatch):
    generator = StrategyGenerator.__new__(StrategyGenerator)
    monkeypatch.setenv("CSD_REQUIRE_NATURAL_DELIMITERS", "1")
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = "reason"
closed_spans = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and closed_spans == 0:
    stepsLeftBeforeIteration = stepsLeft
    if phase == "reason":
        if helpers.EndsWithRightDelimiter(generated):
            closed_spans = closed_spans + 1
            break
        elif helpers.EndsWithLeftDelimiter(generated):
            phase = "span"
            break
        elif helpers.IsComplete(generated):
            phase = "open"
            break
        else:
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            if helpers.EndsWithLeftDelimiter(generated):
                phase = "span"
    elif phase == "open":
        if helpers.EndsWithLeftDelimiter(generated):
            phase = "span"
        else:
            generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
            if helpers.EndsWithLeftDelimiter(generated):
                phase = "span"
    elif phase == "span":
        if helpers.EndsWithRightDelimiter(generated):
            closed_spans = closed_spans + 1
            break
        elif helpers.IsComplete(generated) or helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
            if helpers.EndsWithRightDelimiter(generated):
                closed_spans = closed_spans + 1
                break
        else:
            break
    else:
        break
    if stepsLeft >= stepsLeftBeforeIteration:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "immediately `break`" in issue


def test_structural_issue_rejects_early_parser_readiness_final_ready(monkeypatch):
    generator = StrategyGenerator.__new__(StrategyGenerator)
    monkeypatch.setenv("CSD_REQUIRE_NATURAL_DELIMITERS", "1")
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = "setup"
in_span = False
closed_spans = 0
setup_steps = 0
nudge_steps = 0
answer_ready = False
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and closed_spans == 0:
    stepsLeftBeforeIteration = stepsLeft
    if in_span:
        if helpers.EndsWithRightDelimiter(generated):
            closed_spans = closed_spans + 1
            in_span = False
            break
        elif helpers.IsComplete(generated) or helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
            if helpers.EndsWithRightDelimiter(generated):
                closed_spans = closed_spans + 1
                in_span = False
                break
        else:
            break
    elif phase == "setup":
        if not answer_ready:
            if setup_steps >= 20 and helpers.ValidContinuationCount(generated) > 0 and helpers.ParserDistanceToComplete(generated) <= 1:
                answer_ready = True
        if answer_ready:
            phase = "nudge"
            generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
            nudge_steps = nudge_steps + 1
            if helpers.EndsWithLeftDelimiter(generated):
                in_span = True
                phase = "span"
        else:
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            setup_steps = setup_steps + 1
            if helpers.EndsWithLeftDelimiter(generated):
                in_span = True
                phase = "span"
    elif phase == "nudge":
        generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
        nudge_steps = nudge_steps + 1
        if helpers.EndsWithLeftDelimiter(generated):
            in_span = True
            phase = "span"
    else:
        break
    if stepsLeft >= stepsLeftBeforeIteration:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "parser-distance or valid-continuation" in issue


def test_structural_issue_rejects_mid_counter_answer_ready(monkeypatch):
    generator = StrategyGenerator.__new__(StrategyGenerator)
    monkeypatch.setenv("CSD_REQUIRE_NATURAL_DELIMITERS", "1")
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = "reason"
in_span = False
closed_spans = 0
reason_steps = 0
answer_ready = False
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and closed_spans == 0:
    stepsLeftBeforeIteration = stepsLeft
    if in_span:
        if helpers.EndsWithRightDelimiter(generated):
            in_span = False
            closed_spans = closed_spans + 1
            break
        elif helpers.IsComplete(generated) or helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
            if helpers.EndsWithRightDelimiter(generated):
                in_span = False
                closed_spans = closed_spans + 1
                break
        else:
            break
    elif answer_ready:
        generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
        if helpers.EndsWithLeftDelimiter(generated):
            in_span = True
            phase = "span"
    else:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        reason_steps = reason_steps + 1
        if reason_steps >= 24:
            answer_ready = True
            phase = "answer"
        if helpers.EndsWithLeftDelimiter(generated):
            in_span = True
            phase = "span"
    if stepsLeft >= stepsLeftBeforeIteration:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "forty-plus" in issue


def test_structural_issue_rejects_mid_counter_nudge(monkeypatch):
    generator = StrategyGenerator.__new__(StrategyGenerator)
    monkeypatch.setenv("CSD_REQUIRE_NATURAL_DELIMITERS", "1")
    strategy = """# CSD_RATIONALE_BEGIN
# test
# CSD_RATIONALE_END
phase = "reason"
in_span = False
closed_spans = 0
reason_steps = 0
nudge_steps = 0
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and closed_spans == 0:
    stepsLeftBeforeIteration = stepsLeft
    if in_span:
        if helpers.EndsWithRightDelimiter(generated):
            in_span = False
            closed_spans = closed_spans + 1
            break
        elif helpers.IsComplete(generated) or helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
            if helpers.EndsWithRightDelimiter(generated):
                in_span = False
                closed_spans = closed_spans + 1
                break
        else:
            break
    elif reason_steps >= 24:
        generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
        nudge_steps = nudge_steps + 1
        if helpers.EndsWithLeftDelimiter(generated):
            in_span = True
            phase = "span"
    else:
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        reason_steps = reason_steps + 1
        if helpers.EndsWithLeftDelimiter(generated):
            in_span = True
            phase = "span"
    if stepsLeft >= stepsLeftBeforeIteration:
        break
"""

    issue = generator._structural_issue(strategy)

    assert issue is not None
    assert "roughly forty-plus" in issue
