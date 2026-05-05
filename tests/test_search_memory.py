from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from synthesis.evaluator import EvaluationResult
from synthesis.feedback_loop import SynthesisAttempt, SynthesisPipeline
from synthesis.prompts import (
    build_compilation_error_prompt,
    build_evaluation_failure_prompt,
    build_format_repair_prompt,
    build_verification_error_prompt,
)


BASE_SYMBOL_CORE = """
// CSD_RATIONALE_BEGIN
// Chunked outside, observed entry, explicit cues, symbol core.
// CSD_RATIONALE_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
var steps: nat := 0;
if parser.IsCompletePrefix(currentConstrainedOut) {
  var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
    lm, parser, generated, currentConstrainedOut
  );
  generated := closedGenerated;
  insideConstrainedOut := closedInside;
  currentConstrainedOut := closedCurrent;
}
var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
  lm, parser, generated, currentConstrainedOut, next
);
var symbolGenerated, symbolInside, symbolCurrent, stepsUsed, hitEos :=
  helpers.ConstrainedSymbolInGenerated(
    lm, parser, generated, constrainedPrompt, currentConstrainedOut, 8, eosToken
  );
var chunkGenerated, chunkHitEos, chunkSteps := helpers.UnconstrainedChunk(
  lm, generated, 16, eosToken, "<<"
);
var observedGenerated, observedInside, observedCurrent := helpers.EnterObservedConstrainedSpan(
  lm, parser, chunkGenerated, currentConstrainedOut
);
var openGenerated, openInside, openCurrent := helpers.OpenConstrainedSpan(
  lm, parser, observedGenerated
);
"""


GROUP_VARIANT = BASE_SYMBOL_CORE + """
var validCount := helpers.ValidTokenCount(parser, currentConstrainedOut);
if validCount <= 8 {
  var boosted := helpers.GroupBoostedConstrainedStep(
    lm, parser, constrainedPrompt, currentConstrainedOut, validTokenGroups, 4.0, eosToken
  );
}
"""


ROLLBACK_VARIANT = BASE_SYMBOL_CORE + """
var rollbackGenerated, rollbackInside, rollbackCurrent := helpers.RollbackConstrainedSuffix(
  lm, parser, generated, currentConstrainedOut, 4
);
"""


def _pipeline() -> SynthesisPipeline:
    pipeline = SynthesisPipeline.__new__(SynthesisPipeline)
    pipeline.min_accuracy = 0.40
    pipeline.min_syntax_rate = 0.95
    pipeline.require_delimiters = True
    pipeline.eval_max_seconds_per_example = 120.0
    return pipeline


def _attempt(number: int, strategy: str, accuracy: float, syntax: float, contains: bool) -> SynthesisAttempt:
    attempt = SynthesisAttempt(
        attempt_number=number,
        strategy_code=strategy,
        full_dafny_code=strategy,
        timestamp="2026-05-05T00:00:00",
    )
    attempt.eval_result = EvaluationResult(
        success=True,
        accuracy=accuracy,
        contains_delimiters=contains,
        syntax_rate=syntax,
        num_examples=25,
        num_correct=int(round(accuracy * 25)),
        total_time_seconds=10.0,
        max_sample_time_seconds=8.0,
        sample_outputs=[{"is_correct": False}],
    )
    return attempt


def test_outer_structure_groups_local_narrow_step_swaps_but_exact_profile_does_not():
    pipeline = _pipeline()

    broad_group = pipeline._get_outer_structure_signature(GROUP_VARIANT)
    plain_group = pipeline._get_outer_structure_signature(BASE_SYMBOL_CORE)
    rollback_group = pipeline._get_outer_structure_signature(ROLLBACK_VARIANT)

    assert broad_group == plain_group
    assert broad_group != rollback_group
    assert "chunked outside" in pipeline._describe_outer_structure_signature(broad_group)
    assert "symbol_chunk inside core" in pipeline._describe_outer_structure_signature(broad_group)

    _, exact_group = pipeline._get_strategy_profile_for_evaluation_history(GROUP_VARIANT)
    _, exact_plain = pipeline._get_strategy_profile_for_evaluation_history(BASE_SYMBOL_CORE)
    assert exact_group != exact_plain


def test_compact_search_memory_surfaces_outcome_traps_and_broad_family():
    pipeline = _pipeline()
    attempts = [
        _attempt(2, BASE_SYMBOL_CORE, 0.28, 0.92, False),
        _attempt(4, BASE_SYMBOL_CORE, 0.32, 0.84, True),
        _attempt(8, GROUP_VARIANT, 0.28, 0.92, False),
        _attempt(9, BASE_SYMBOL_CORE, 0.32, 0.84, True),
        _attempt(12, GROUP_VARIANT, 0.28, 0.92, False),
        _attempt(19, GROUP_VARIANT, 0.32, 0.84, True),
        _attempt(21, BASE_SYMBOL_CORE, 0.32, 0.84, True),
    ]

    memory = pipeline._get_compact_search_memory(attempts, current_attempt=attempts[-1])

    assert memory.startswith("Search memory:")
    assert "Balanced-best: attempt 4" in memory
    assert "Repeated outcome trap: 28% accuracy / 92% syntax / required delimiter absent" in memory
    assert "Repeated broad family:" in memory
    assert "attempts 4, 8, 9, 12, 19, 21" in memory
    assert len(memory.splitlines()) <= 12


def test_search_memory_is_near_top_of_refinement_and_repair_prompts():
    search_memory = "Search memory:\n- Balanced-best: attempt 9, 32.0% accuracy / 84.0% syntax / delimiter present."

    _, eval_prompt = build_evaluation_failure_prompt(
        "task",
        "body",
        "feedback",
        search_memory=search_memory,
    )
    _, verification_prompt = build_verification_error_prompt(
        "task",
        "body",
        "error",
        search_memory=search_memory,
    )
    _, compilation_prompt = build_compilation_error_prompt(
        "body",
        "compile error",
        search_memory=search_memory,
    )
    _, format_prompt = build_format_repair_prompt("body", search_memory=search_memory)

    assert eval_prompt.index("Search memory:") < eval_prompt.index("## Strategy Context")
    assert verification_prompt.index("Search memory:") < verification_prompt.index("Previous attempt:")
    assert compilation_prompt.index("Search memory:") < compilation_prompt.index("Previous attempt:")
    assert format_prompt.index("Search memory:") < format_prompt.index("Content to rewrite:")
