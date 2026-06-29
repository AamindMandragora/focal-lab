"""
Main synthesis pipeline with feedback-based refinement.

Orchestrates the generate -> verify -> compile -> run loop with
iterative refinement based on errors.
"""

import json
import math
import os
import re
import secrets
from difflib import unified_diff
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from ..verify.compiler import CompilationResult, DafnyCompiler
from .evaluator import Evaluator, EvaluationResult
from .goodness import (
    compute_goodness_from_attempt,
    evaluation_scalar_score,
    scalar_target,
)
from .rex_bandit import RexBandit
from .search_tree import SearchNode, SearchTree
from ..generate.generator import StrategyGenerator
from ..generate import prompts as generation_prompts
from ..generate.rationale import extract_rationale
from ..verify.verifier import DafnyVerifier, VerificationResult


def _delimiter_miss_hint(require_delimiters: bool, contains_delimiters: bool, sample_outputs=None) -> str:
    """Localized diagnostic when the eval required << >> spans but produced none.

    Two distinct root causes need different advice:
    1. Spans never opened: strategy waits for model to emit "<<" but weak model never does.
       Fix: force "<<" via a direct-control helper.
    2. Spans opened but never closed: strategy gets "<<" in the output but the step budget
       runs out before ">>" is produced — the span terminates the span incorrectly.
       Fix: improve how the strategy decides a span is complete and exits it.
    Returns "" when delimiters are present or not required.
    """
    if not require_delimiters or contains_delimiters:
        return ""

    # contains_delimiters is True only when ALL examples had spans (strict AND).
    # If sample data shows spans are being produced for most outputs, the "none
    # had spans" message is wrong — suppress the hint entirely so the author isn't
    # misled into forcing delimiters when they're already mostly working.
    if sample_outputs:
        n = len(sample_outputs)
        n_with = sum(1 for s in sample_outputs if s.get("contains_delimiters", False))
        if n > 0 and n_with / n >= 0.05:
            return ""

    # Determine which root cause applies by inspecting sample outputs.
    # n_open_not_closed: outputs where "<<" IS present but ">>" is absent —
    # the span was opened but the strategy never closed it.
    if sample_outputs:
        n = len(sample_outputs)
        n_open_not_closed = sum(
            1 for s in sample_outputs
            if not s.get("uses_hidden_chunks", False)
            and "<<" in (s.get("full_output") or "")
            and ">>" not in (s.get("full_output") or "")
        )
        if n > 0 and n_open_not_closed >= 0.2 * n:
            # Root cause 2: spans opened but never closed.
            return (
                f"\n  ⚠ Delimiter check FAILED: none of the evaluated outputs "
                f"contained a complete << >> span, but spans are required. "
                f"{n_open_not_closed}/{n} outputs show that a `<<` span was "
                f"opened but the strategy never produced the closing `>>` — the "
                f"step budget ran out before the span was exited. This means `<<` "
                f"IS being emitted, but the strategy does not successfully reach "
                f"and emit `>>` for any example.\n"
                f"    What to reconsider: this is a decoding-mechanism issue with "
                f"how the strategy behaves *inside* a span — specifically, how it "
                f"makes forward progress through the span content and how it "
                f"recognizes that a span is complete and should be closed. The "
                f"strategy needs a reliable path from span-open to span-close "
                f"within the step budget. (General direction only — the specific "
                f"mechanism is yours to design.)\n"
            )

    return (
        "\n  ⚠ Delimiter check FAILED: none of the evaluated outputs contained a "
        "<< >> span, but spans are required.\n"
        "    Likely cause: the strategy opens spans by WAITING for the model to "
        "emit \"<<\" (e.g. a `next == \"<<\"` trigger, or an unconstrained chunk "
        "that stops on \"<<\"). A weak eval model may never emit \"<<\" on its "
        "own, so the trigger never fires, no span opens, and \">>\" is never "
        "reached.\n"
        "    Fix to consider: FORCE the opening delimiter at span entry (append "
        "\"<<\" directly via a forced-delimiter / direct-control helper) instead "
        "of depending on the model to produce it, then force \">>\" at span exit. "
        "This is a decoding-mechanism change, not task guidance.\n"
    )


def _span_not_closed_hint(require_delimiters: bool, sample_outputs) -> str:
    """Nudge when spans OPEN but never CLOSE before the step budget runs out.

    Distinct from _delimiter_miss_hint (no span opened at all): here the model
    DID emit "<<" but the strategy never brought the span to a ">>", so it burns
    the whole step budget inside one open span and the eval records no usable
    answer (accuracy/syntax collapse toward 0). Common when the strategy advances
    many constrained tokens per step with no way to recognize/terminate a complete
    span. We deliberately name only the SYMPTOM and the AREA to reconsider (how the
    strategy progresses and decides it is done inside a span) and leave the specific
    mechanism to the author — a general decoding-mechanism nudge, not task guidance.
    Returns "" when not required, no samples, or the pattern is not prevalent.
    """
    if not require_delimiters or not sample_outputs:
        return ""
    n = len(sample_outputs)
    n_unterminated = sum(
        1 for s in sample_outputs
        if not s.get("uses_hidden_chunks", False)
        and "<<" in (s.get("full_output") or "")
        and ">>" not in (s.get("full_output") or "")
    )
    n_maxsteps_nodelim = sum(
        1 for s in sample_outputs
        if s.get("hit_max_steps") and not s.get("contains_delimiters", False)
    )
    n_affected = max(n_unterminated, n_maxsteps_nodelim)
    if n_affected == 0 or n_affected < 0.1 * n:
        return ""
    return (
        f"\n  ⚠ Span-closure check: {n_affected}/{n} evaluated outputs opened a "
        "`<<` span but never emitted the closing `>>` before the generation step "
        "budget ran out. When a span never closes, the eval records no usable "
        "answer for that example, so accuracy and syntax collapse toward zero even "
        "though the model did start a span.\n"
        "    What to reconsider: this is a decoding-mechanism issue with how the "
        "strategy behaves *inside* a span — how it makes forward progress and how "
        "it decides the span is finished — not the task or the prompt. Aim for a "
        "strategy that reliably reaches and emits `>>` within the step budget. "
        "(General direction only — the specific mechanism is yours to design.)\n"
    )


def _constraint_bypassed_hint(require_delimiters: bool, contains_delimiters: bool, sample_outputs=None) -> str:
    """Nudge when << >> spans APPEAR in the text but the constraint barely engaged.

    Distinct from the other two delimiter hints:
      - _delimiter_miss_hint  : no spans at all (delimiters absent).
      - _span_not_closed_hint : spans open but never reach ">>".
    Here the delimiters ARE present, yet the strategy's constrained branch ran on
    almost none of the examples (`used_constrained_chunk` low) — so the span content
    was produced UNCONSTRAINED and its syntax is at the raw model's mercy. This is the
    failure mode where a reactive span-entry trigger (e.g. `next == "<<"`) never matches
    because the model emits a space-prefixed delimiter token, so the constrained path is
    skipped even though "<<" still shows up in the output text.

    We name only the SYMPTOM (constraint engaged on few examples) and the AREA to
    reconsider (span ENTRY — how the strategy decides to enter the constrained branch)
    and leave the mechanism to the author. Decoding-mechanism nudge, not task guidance.
    Returns "" when not required, delimiters absent, no samples, or the field is unrecorded.
    """
    if not require_delimiters or not contains_delimiters or not sample_outputs:
        return ""
    # Only examples that (a) actually show delimiters, (b) record whether the
    # constrained branch ran, and (c) are not using a different (hidden-chunk)
    # constraint mechanism can tell us whether the span content was constrained.
    relevant = [
        s for s in sample_outputs
        if not s.get("uses_hidden_chunks", False)
        and s.get("contains_delimiters", False)
        and "used_constrained_chunk" in s
    ]
    n_rel = len(relevant)
    if n_rel == 0:
        return ""
    n_engaged = sum(1 for s in relevant if s.get("used_constrained_chunk"))
    n_bypassed = n_rel - n_engaged
    n = len(sample_outputs)
    if n_bypassed < 0.2 * n or n_engaged >= 0.5 * n_rel:
        return ""
    return (
        f"\n  ⚠ Constraint-engagement check: {n_engaged}/{n_rel} of the outputs that "
        f"show `<< >>` actually ran the strategy's constrained branch — the other "
        f"{n_bypassed} produced the span content UNCONSTRAINED. The delimiters appear "
        f"in the text, so the spans LOOK present, but the constraint did not shape what "
        f"went inside them, leaving the span syntax at the raw model's mercy.\n"
        f"    Likely cause: the strategy enters its constrained branch by WAITING for a "
        f"specific span-open signal (e.g. a `next == \"<<\"` trigger) that rarely "
        f"matches the model's actual output, so the constrained path is skipped even "
        f"though `<<` still appears.\n"
        f"    Fix to consider: FORCE span entry — append the opening `<<` directly via "
        f"a forced-delimiter / direct-control helper and then drive the span content "
        f"through the constrained branch, instead of depending on a reactive trigger to "
        f"detect span entry. This is a decoding-mechanism change at span ENTRY, not "
        f"task guidance. (General direction only — the specific mechanism is yours to "
        f"design.)\n"
    )


def _final_span_failure_hint(require_delimiters: bool, sample_outputs=None) -> str:
    """Classify delimiter-bearing syntax failures into machine-checkable categories.

    The three categories are:
      - ``final_span_unclosed``: output has ``<<`` after the last ``>>``, so the
        final answer span opened but the generation stopped before ``>>`` was
        emitted.  (rfind('<<') > rfind('>>'))
      - ``no_span_emitted``: output has no ``<<`` at all — the model never
        started a constrained span.
      - ``final_span_invalid``: the final span closed (a complete ``<< >>``
        block exists) but the block fails the CRANE GSM syntax check (e.g.
        contains ``{``, ``}``, or ``**``).

    Returns a non-empty hint string when ``require_delimiters`` is True, there
    are sample outputs, and at least one syntax-failing example exists.
    Matches the style of ``_delimiter_miss_hint`` / ``_span_not_closed_hint``
    (symptom + area, no task-specific guidance).
    """
    if not require_delimiters or not sample_outputs:
        return ""

    # Collect syntax-failing examples (ones where the final span was expected
    # but either absent or invalid).  We look at full_output for each sample
    # that is NOT marked is_syntax_valid and is NOT using hidden chunks.
    unclosed: list[str] = []
    no_span: list[str] = []
    invalid: list[str] = []

    for s in sample_outputs:
        if s.get("uses_hidden_chunks"):
            continue
        if s.get("is_syntax_valid"):
            continue
        full_output = s.get("full_output") or ""
        last_open = full_output.rfind("<<")
        last_close = full_output.rfind(">>")
        if last_open == -1:
            # No << at all
            no_span.append(full_output)
        elif last_open > last_close:
            # << exists but either >> is absent or << appears after the last >>
            unclosed.append(full_output)
        else:
            # Closed block exists but failed syntax — final_span_invalid
            invalid.append(full_output)

    total_classified = len(unclosed) + len(no_span) + len(invalid)
    if total_classified == 0:
        return ""

    def _tail(output: str, max_chars: int = 120) -> str:
        """Return the last ``max_chars`` characters of ``output``, trimmed."""
        trimmed = output.strip()
        if len(trimmed) <= max_chars:
            return trimmed
        return "..." + trimmed[-max_chars:]

    lines = [
        f"\n  Delimiter syntax-failure breakdown ({total_classified} examples):"
    ]

    if unclosed:
        lines.append(
            f"    final_span_unclosed: {len(unclosed)} example(s) — "
            f"the generation emitted `<<` at the end and then stopped "
            f"(EOS or dead-end) before producing any span content or `>>`."
        )
        for ex in unclosed[:2]:
            lines.append(f"      output tail: {_tail(ex)!r}")

    if no_span:
        lines.append(
            f"    no_span_emitted: {len(no_span)} example(s) — "
            f"no `<<` appeared anywhere in the output."
        )
        for ex in no_span[:2]:
            lines.append(f"      output tail: {_tail(ex)!r}")

    if invalid:
        lines.append(
            f"    final_span_invalid: {len(invalid)} example(s) — "
            f"a `<< >>` block was present but failed the syntax check "
            f"(e.g. contained `{{`, `}}`, `**`, or was otherwise unparseable)."
        )
        for ex in invalid[:2]:
            lines.append(f"      output tail: {_tail(ex)!r}")

    lines.append(
        "    What to reconsider: these are decoding-mechanism failures at "
        "span entry/exit — not task guidance issues. Each category points to "
        "a different span-lifecycle step to strengthen."
    )
    return "\n".join(lines) + "\n"


def _unit_rewind_hint(strategy_source: str, sample_outputs) -> str:
    """Nudge toward RegenerateUnitOnCheckFailure when semantic failures dominate.

    Fires when BOTH:
      (a) Failures are dominated by syntax-valid but semantically wrong outputs
          (syntax_rate - accuracy >= 0.25, or most failing examples pass syntax).
      (b) The strategy source does NOT reference RegenerateUnitOnCheckFailure.

    The hint text is mechanism-level and task-agnostic: it describes what the
    helper does (unit-level rewind and resample on check failure) without
    mentioning SQL, joins, grammar specifics, or any other task detail.
    Returns "" when conditions are not met.
    """
    if not sample_outputs:
        return ""
    src = strategy_source or ""
    if "RegenerateUnitOnCheckFailure" in src or "RegenerateUnitOnGroundingFailure" in src:
        return ""

    n = len(sample_outputs)
    if n == 0:
        return ""

    n_correct = sum(1 for s in sample_outputs if s.get("is_correct"))
    n_syntax_valid = sum(1 for s in sample_outputs if s.get("is_syntax_valid"))
    accuracy = n_correct / n
    syntax_rate = n_syntax_valid / n

    # Condition (a): failures dominated by well-formed-but-wrong outputs.
    # Either the gap between syntax rate and accuracy is large (>= 0.25),
    # or the majority of failing examples are syntax-valid.
    n_failing = n - n_correct
    if n_failing == 0:
        return ""
    n_syntax_valid_wrong = sum(
        1 for s in sample_outputs
        if s.get("is_syntax_valid") and not s.get("is_correct")
    )
    semantic_dominated = (
        (syntax_rate - accuracy) >= 0.25
        or (n_failing > 0 and n_syntax_valid_wrong / n_failing >= 0.5)
    )
    if not semantic_dominated:
        return ""

    return (
        "\n  Unit-rewind opportunity: most failing examples produced well-formed output "
        "that passed the syntax check but scored incorrect — the errors are semantic, "
        "not structural. The library helper RegenerateUnitOnCheckFailure can check each "
        "completed grammar unit during generation against a caller-supplied set of allowed "
        "units and rewind-and-resample the unit on mismatch, catching wrong choices before "
        "the full span is committed. Consider whether unit-level checking applies here.\n"
    )


class FailureStage(Enum):
    """Stage where synthesis attempt failed."""

    SEARCH_CONTRACT = "search_contract"
    VERIFICATION = "verification"
    COMPILATION = "compilation"
    RUNTIME = "runtime"
    EVALUATION = "evaluation"


def parse_strategy_type(strategy_code: str) -> dict:
    """
    Parse the generated strategy code to extract strategy type and parameters.
    Useful for research analysis comparing dynamic vs static strategies.

    Returns:
        dict with keys: strategy_name, parameters, category
    """
    import re

    # Strip any embedded rationale block so pattern matching reflects the actual Dafny statements.
    extracted = extract_rationale(strategy_code)
    strategy_code_for_match = (
        extracted.body_without_rationale.strip() if extracted.has_markers else strategy_code.strip()
    )

    # Pattern matching for each strategy type
    patterns = {
        "PureConstrainedGeneration": {
            "pattern": r"PureConstrainedGeneration|ConstrainedGeneration",
            "category": "fully_constrained",
            "comparable_to": "SynCode",
        },
        "TryUnconstrainedThenConstrained": {
            "pattern": r"TryUnconstrainedThenConstrained.*?(\d+)",
            "category": "optimistic_with_fallback",
            "comparable_to": "IterGen-like",
        },
        "HybridGeneration": {
            "pattern": r"HybridGeneration.*?(\d+)",
            "category": "interleaved",
            "comparable_to": "Novel",
        },
        "SpeculativeGeneration": {
            "pattern": r"SpeculativeGeneration.*?(\d+)",
            "category": "speculative",
            "comparable_to": "SpecDec-like",
        },
        "CraneGeneration": {
            "pattern": r"CraneGeneration",
            "category": "crane_style",
            "comparable_to": "CRANE",
        },
    }

    for name, info in patterns.items():
        match = re.search(info["pattern"], strategy_code_for_match)
        if match:
            params: dict[str, int] = {}
            if match.groups():
                if name == "TryUnconstrainedThenConstrained":
                    params["unconstrained_steps"] = int(match.group(1))
                elif name == "HybridGeneration":
                    params["interval"] = int(match.group(1))
                elif name == "SpeculativeGeneration":
                    params["window_size"] = int(match.group(1))

            return {
                "strategy_name": name,
                "parameters": params,
                "category": info["category"],
                "comparable_to": info["comparable_to"],
                "raw_code": strategy_code,
            }

    return {
        "strategy_name": "Unknown",
        "parameters": {},
        "category": "unknown",
        "comparable_to": "N/A",
        "raw_code": strategy_code,
    }


@dataclass
class SynthesisAttempt:
    """Record of a single synthesis attempt."""

    attempt_number: int
    strategy_code: str
    full_dafny_code: str
    timestamp: str

    # Results from each stage (None if stage not reached)
    verification_result: Optional[VerificationResult] = None
    compilation_result: Optional[CompilationResult] = None
    eval_result: Optional[EvaluationResult] = None

    # Failure information
    failed_at: Optional[FailureStage] = None
    error_summary: str = ""

    # REx search-tree linkage
    node_id: Optional[int] = None
    parent_node_id: Optional[int] = None
    goodness: float = 0.0
    met_threshold: bool = False

    def succeeded(self) -> bool:
        """Check if this attempt passed verify, compile, and evaluation."""
        if self.failed_at is not None:
            return False
        return (
            self.verification_result is not None
            and self.verification_result.success
            and self.compilation_result is not None
            and self.compilation_result.success
            and self.eval_result is not None
            and self.eval_result.success
        )

    def get_strategy_analysis(self) -> dict:
        """Get parsed strategy information for research analysis."""
        return parse_strategy_type(self.strategy_code)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        strategy_analysis = self.get_strategy_analysis()
        return {
            "attempt_number": self.attempt_number,
            "strategy_code": self.strategy_code,
            "strategy_analysis": strategy_analysis,  # For research comparison
            "timestamp": self.timestamp,
            "succeeded": self.succeeded(),
            "failed_at": self.failed_at.value if self.failed_at else None,
            "error_summary": self.error_summary,
            "node_id": self.node_id,
            "parent_node_id": self.parent_node_id,
            "goodness": self.goodness,
            "met_threshold": self.met_threshold,
            "verification": {
                "success": self.verification_result.success if self.verification_result else None,
                "error_count": len(self.verification_result.errors) if self.verification_result else 0,
            }
            if self.verification_result
            else None,
            "compilation": {
                "success": self.compilation_result.success if self.compilation_result else None,
                "output_dir": str(self.compilation_result.output_dir)
                if self.compilation_result and self.compilation_result.output_dir
                else None,
            }
            if self.compilation_result
            else None,
            "evaluation": self.eval_result.to_dict() if self.eval_result else None,
        }


class SynthesisExhaustionError(Exception):
    """
    Raised when synthesis fails after exhausting all attempts.

    Contains detailed information about all attempts for debugging.
    """

    def __init__(
        self,
        message: str,
        attempts: list[SynthesisAttempt],
        report_path: Optional[Path] = None,
    ):
        super().__init__(message)
        self.attempts = attempts
        self.report_path = report_path

    def get_failure_summary(self) -> str:
        """Get a summary of failure patterns across attempts."""
        if not self.attempts:
            return "No attempts were made"

        lines = [f"Synthesis failed after {len(self.attempts)} attempt(s):", ""]

        # Count failures by stage
        stage_counts = {stage: 0 for stage in FailureStage}
        for attempt in self.attempts:
            if attempt.failed_at:
                stage_counts[attempt.failed_at] += 1

        lines.append("Failure breakdown by stage:")
        for stage, count in stage_counts.items():
            if count > 0:
                lines.append(f"  - {stage.value}: {count}")

        lines.append("")
        lines.append("Individual attempt summaries:")

        for attempt in self.attempts:
            status = (
                "✓ SUCCESS"
                if attempt.succeeded()
                else f"✗ Failed at {attempt.failed_at.value if attempt.failed_at else 'unknown'}"
            )
            lines.append(f"  Attempt {attempt.attempt_number}: {status}")
            if attempt.error_summary:
                # Truncate long error messages
                error_preview = attempt.error_summary[:200]
                if len(attempt.error_summary) > 200:
                    error_preview += "..."
                lines.append(f"    Error: {error_preview}")

        if self.report_path:
            lines.append("")
            lines.append(f"Full report saved to: {self.report_path}")

        return "\n".join(lines)


@dataclass
class SynthesisResult:
    """Result of a synthesis run (best-goodness node returned regardless of success)."""

    success: bool
    strategy_code: str
    full_dafny_code: str
    compiled_module_path: Optional[Path]
    output_dir: Optional[Path]
    run_dir: Optional[Path]
    attempts: list[SynthesisAttempt]
    total_time_ms: float
    best_node_id: Optional[int] = None
    best_goodness: float = 0.0
    met_threshold: bool = False
    search_tree: list[dict] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "strategy_code": self.strategy_code,
            "compiled_module_path": str(self.compiled_module_path)
            if self.compiled_module_path
            else None,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "num_attempts": len(self.attempts),
            "total_time_ms": self.total_time_ms,
            "best_node_id": self.best_node_id,
            "best_goodness": self.best_goodness,
            "met_threshold": self.met_threshold,
            "search_tree": self.search_tree,
        }


class SynthesisPipeline:
    """
    Main pipeline for synthesizing CSD strategies.

    Orchestrates REx search over an explicit strategy tree:
    1. Bootstrap root strategy (author model)
    2. Thompson-sampled arm selection (REx)
    3. Refinement pull -> verify -> compile -> evaluate
    4. Return argmax-goodness node after the full iteration budget
    """

    DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "generated"
    NON_PRUNABLE_HELPERS = {
        "AppendTaskGuidance",
        "UnconstrainedStep",
        "ConstrainedStep",
        "AppendConstrainedToken",
        "OpenConstrainedSpan",
        "EnterObservedConstrainedSpan",
        "CloseConstrainedSpan",
        "CloseSpanIfComplete",
        "Contains",
        "RenderPrefix",
        "GenerateLogits",
        "ChooseNextToken",
        "ChooseNextTokenUnconstrained",
        "GenerateUnconstrainedChunk",
        "MaskValidNextAndEos",
        "BoostValidNextAndEos",
        "IdToToken",
        "TokenToId",
        "TokenToIdRecursive",
        "IdToLogit",
        "TokenToLogit",
        "TokensToLogits",
        "IdsToLogits",
        "MaskToken",
        "MaskTokens",
        "MaskTokensExcept",
        "IsMasked",
        "HasUnmaskedToken",
        "IsValidPrefix",
        "IsCompletePrefix",
        "IsDeadPrefix",
        "ValidNextTokenCount",
        "ValidNextToken",
        "ValidNextTokens",
        "ParseG",
        "IsTokenValidNext",
        "ValidTokenCount",
        "DeadEndDetection",
        "TopValidCandidates",
        "RollbackConstrainedSuffix",
        "LastTokenBefore",
        "RegenerateUnitOnGroundingFailure",
        "CloseSpanWithinBudget",
    }
    PRUNABLE_HELPERS = {
        "UnconstrainedGeneration",
        "ConstrainedGeneration",
        "CraneGeneration",
        "UnconstrainedChunk",
        "ConstrainedSymbol",
        "ConstrainedSymbolInGenerated",
        "ConfidenceGatedStep",
        "SafeBoostedConstrainedStep",
        "SafePenalizedConstrainedStep",
        "SafeRepetitionPenaltyStep",
        "SafeTemperatureConstrainedStep",
        "SafeSoftConstrainedStep",
        "GroupBoostedConstrainedStep",
        "GroupHasValidMember",
        "BoostValidGroups",
        "DeadEndAvoidingStep",
        "RollbackAndRegenerate",
        "RegenerateUnitOnCheckFailure",
        "RollbackAndContinue",
        "RollbackConstrainedToComplete",
        "RollbackToCompletePrefix",
        "AdaptiveConstrainedStep",
        "AdaptiveConstrainedStepWithPenalties",
        "PenalizedConstrainedStep",
        "BoostedConstrainedStep",
        "SoftConstrainedStep",
        "BoostTokenLogits",
        "PenalizeTokenLogits",
        "SafeBoostTokenLogits",
        "SafePenalizeTokenLogits",
        "MaskTokensInPrefix",
        "GetHighestLogitToken",
        "GetLogitGap",
        "GetTopKTokens",
        "GetTokenLogit",
        "ScaleAllLogits",
        "SaveLogitsSnapshot",
        "RestoreLogitsSnapshot",
        "SpeculativeConstrainedRollout",
        "RolloutConstrainedWithPenalties",
        "RepetitionPenaltyStep",
        "TemperatureConstrainedStep",
        "RollbackConstrainedSpan",
        "ExtractAfterKeyword",
        "IntersectTokenSets",
        "SubtractTokenSets",
        "RollbackToValidPrefix",
        "FlattenTokenGroups",
        "GroupContaining",
        "PrefixToString",
        "ExtractContentBetweenDelimiters",
        "CountSubstring",
        "CountTokenOccurrences",
        "OccurrencesInRange",
        "TokensSinceLastOccurrence",
    }
    def __init__(
        self,
        evaluator: Evaluator,
        generator: Optional[StrategyGenerator] = None,
        verifier: Optional[DafnyVerifier] = None,
        compiler: Optional[DafnyCompiler] = None,
        max_iterations: int = 5,
        output_dir: Optional[Path] = None,
        save_reports: bool = True,
        # Evaluation thresholds
        min_accuracy: float = 0.0,
        min_syntax_rate: float = 0.0,
        require_delimiters: bool = True,
        eval_sample_size: int = 10,
        eval_max_seconds_per_example: Optional[float] = 90.0,
        min_examples_before_threshold_stop: Optional[int] = 15,
        adaptive_helper_mask: bool = True,
        helper_selection_policy: str = "bandit",
        helper_mask_min_evals: int = 4,
        helper_mask_min_uses: int = 2,
        helper_mask_margin: float = 0.25,
        helper_mask_max_disabled: int = 6,
        helper_bandit_min_evals: int = 3,
        helper_bandit_top_k: int = 12,
        helper_bandit_ucb_c: float = 0.35,
        helper_bandit_explore_untried: int = 1,
        refinement_beam_size: int = 2,
        local_neighborhood_refinement: bool = True,
        max_local_edit_ratio: float = 0.65,
        beam_verify_candidates: bool = True,
        # Restart-from-scratch mechanism: escape local-search basins when
        # anchor-based refinement gets stuck.
        restart_after_stuck_iters: int = 0,
        restart_cooldown_iters: int = 0,
        rex_temperature: float = 2.0,
    ):
        """
        Initialize the synthesis pipeline.

        Args:
            evaluator: Evaluator for dataset-based feedback (required)
            generator: Strategy generator (creates default if None)
            verifier: Dafny verifier (creates default if None)
            compiler: Dafny compiler (creates default if None)
            max_iterations: Maximum refinement iterations
            output_dir: Directory for outputs and reports
            save_reports: Whether to save failure reports to disk
            min_accuracy: Minimum accuracy threshold for evaluation
            min_syntax_rate: Minimum syntax validity rate threshold
            require_delimiters: Whether evaluated outputs must contain << >> spans
            eval_sample_size: Number of examples to evaluate on
            eval_max_seconds_per_example: Optional runtime budget per example in seconds
            min_examples_before_threshold_stop: Minimum number of examples that
                must be evaluated before threshold-impossible early stops can
                fire. Decouples the synthesis feedback budget from the
                acceptance threshold so the synthesizer always sees a usable
                amount of evaluation data. None means no minimum (legacy).
            adaptive_helper_mask: Enable empirical helper pruning contract
            helper_selection_policy: Helper selection policy (`bandit` only; UCB-style)
            helper_mask_min_evals: Evaluated attempts before pruning can start
            helper_mask_min_uses: Minimum helper usage count before pruning
            helper_mask_margin: Margin below run-wide mean utility to prune a helper
            helper_mask_max_disabled: Maximum helpers disabled in one run
            helper_bandit_min_evals: Evaluated attempts before bandit selection starts
            helper_bandit_top_k: Number of prunable helpers to keep active under bandit
            helper_bandit_ucb_c: UCB exploration coefficient
            helper_bandit_explore_untried: Number of unseen helpers to force-explore
            refinement_beam_size: Number of refinement candidates to sample per step
            local_neighborhood_refinement: Prefer local edits during refinement
            max_local_edit_ratio: Soft bound on changed-line ratio for local edits
            beam_verify_candidates: Verify beam candidates before selecting one
        """
        self.evaluator = evaluator
        self.generator = generator or StrategyGenerator()
        self.verifier = verifier or DafnyVerifier()
        self.compiler = compiler or DafnyCompiler()
        self.max_iterations = max_iterations
        self.output_dir = output_dir or self.DEFAULT_OUTPUT_DIR
        self.save_reports = save_reports

        # Restart-from-scratch mechanism (replaces two-phase). When the
        # Pareto-best anchor has not advanced for N consecutive iterations,
        # the next refinement call switches into restart mode (drops anchor,
        # asks for a structurally different family). Counter resets when the
        # anchor advances. Optional cooldown after a restart prevents
        # back-to-back restarts.
        self.restart_after_stuck_iters = max(0, int(restart_after_stuck_iters))
        self.restart_cooldown_iters = max(0, int(restart_cooldown_iters))
        self.rex_temperature = max(0.0, rex_temperature)
        self._scalar_target = scalar_target(
            min_accuracy=self.min_accuracy,
            min_syntax_rate=self.min_syntax_rate,
            require_delimiters=self.require_delimiters,
            eval_max_seconds_per_example=self.eval_max_seconds_per_example,
        )
        self._anchor_attempt_number: int | None = None
        self._iters_since_anchor_changed: int = 0

        self.min_accuracy = min_accuracy
        self.min_syntax_rate = min_syntax_rate
        self.require_delimiters = require_delimiters
        self.eval_sample_size = eval_sample_size
        self.eval_max_seconds_per_example = eval_max_seconds_per_example
        self.min_examples_before_threshold_stop = min_examples_before_threshold_stop
        self.adaptive_helper_mask = adaptive_helper_mask
        normalized_policy = helper_selection_policy.strip().lower()
        if normalized_policy != "bandit":
            raise ValueError(
                "helper_selection_policy must be 'bandit' (UCB/bandit only)"
            )
        self.helper_selection_policy = normalized_policy
        self.helper_mask_min_evals = max(1, helper_mask_min_evals)
        self.helper_mask_min_uses = max(1, helper_mask_min_uses)
        self.helper_mask_margin = helper_mask_margin
        self.helper_mask_max_disabled = max(0, helper_mask_max_disabled)
        self.helper_bandit_min_evals = max(1, helper_bandit_min_evals)
        self.helper_bandit_top_k = max(1, helper_bandit_top_k)
        self.helper_bandit_ucb_c = max(0.0, helper_bandit_ucb_c)
        self.helper_bandit_explore_untried = max(0, helper_bandit_explore_untried)
        self.refinement_beam_size = max(1, refinement_beam_size)
        self.local_neighborhood_refinement = local_neighborhood_refinement
        self.max_local_edit_ratio = max(0.0, max_local_edit_ratio)
        self.beam_verify_candidates = beam_verify_candidates
        self._helper_universe = self._extract_helper_universe_from_prompts()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _run_configuration_metadata(self, task_description: str, output_name: str) -> dict:
        """Return run-level provenance that is useful for experiment analysis."""
        evaluator = self.evaluator
        generator = self.generator
        return {
            "task_description": task_description,
            "output_name": output_name,
            "max_iterations": self.max_iterations,
            "thresholds": {
                "min_accuracy": self.min_accuracy,
                "min_syntax_rate": self.min_syntax_rate,
                "require_delimiters": self.require_delimiters,
            },
            "author_model": {
                "backend": getattr(generator, "backend", None),
                "model": getattr(generator, "model_name", None),
                "max_new_tokens": getattr(generator, "max_new_tokens", None),
                "anthropic_thinking": getattr(generator, "anthropic_thinking", None),
                "anthropic_effort": getattr(generator, "anthropic_effort", None),
                "anthropic_thinking_display": getattr(
                    generator, "anthropic_thinking_display", None
                ),
            },
            "evaluation": {
                "dataset": getattr(evaluator, "dataset_name", None),
                "eval_model": getattr(evaluator, "model_name", None),
                "eval_backend": getattr(evaluator, "backend", None),
                "eval_sample_size": self.eval_sample_size,
                "eval_max_steps": getattr(evaluator, "max_steps", None),
                "eval_step_token_budget": getattr(evaluator, "step_token_budget", None),
                "eval_max_seconds_per_example": self.eval_max_seconds_per_example,
                "min_examples_before_threshold_stop": self.min_examples_before_threshold_stop,
            },
            "synthesis_controls": {
                "restart_after_stuck_iters": self.restart_after_stuck_iters,
                "restart_cooldown_iters": self.restart_cooldown_iters,
                "adaptive_helper_mask": self.adaptive_helper_mask,
                "helper_selection_policy": self.helper_selection_policy,
                "helper_bandit_min_evals": self.helper_bandit_min_evals,
                "helper_bandit_top_k": self.helper_bandit_top_k,
                "helper_bandit_ucb_c": self.helper_bandit_ucb_c,
                "helper_bandit_explore_untried": self.helper_bandit_explore_untried,
                "refinement_beam_size": self.refinement_beam_size,
                "local_neighborhood_refinement": self.local_neighborhood_refinement,
                "max_local_edit_ratio": self.max_local_edit_ratio,
                "beam_verify_candidates": self.beam_verify_candidates,
                "search_policy": "rex",
                "rex_temperature": self.rex_temperature,
                "scalar_target": self._scalar_target,
            },
        }

    def _unload_evaluator_runtime_before_refinement(self) -> None:
        """Release evaluator GPU runtime only when the author model also needs it."""
        generator_backend = getattr(self.generator, "backend", "")
        if generator_backend not in {"huggingface", "vllm"}:
            print(
                "  Evaluator runtime kept warm; generation backend is hosted "
                f"({generator_backend or 'unknown'})"
            )
            return
        self.evaluator.unload_runtime()
        print("  Evaluator runtime unloaded to free GPU memory")

    def _get_recent_behavioral_context(self, attempts: list[SynthesisAttempt]) -> str:
        """Return a compact behavior summary from the most recent evaluated attempt."""
        for attempt in reversed(attempts):
            if attempt.eval_result is not None:
                summary = attempt.eval_result.get_behavioral_context_summary(
                    require_delimiters=self.require_delimiters
                )
                if summary:
                    return summary
        return ""

    def _get_verification_history_summary(self, attempts: list[SynthesisAttempt], max_attempts: int = 3) -> str:
        """Summarize recent verification failures so refinement can avoid oscillation."""
        recent_failures = [
            attempt
            for attempt in attempts
            if attempt.failed_at == FailureStage.VERIFICATION and attempt.verification_result is not None
        ][-max_attempts:]
        if not recent_failures:
            return ""

        lines = []
        for attempt in recent_failures:
            diagnostics = attempt.verification_result.diagnostics or []
            if not diagnostics:
                preview = attempt.error_summary.splitlines()[0] if attempt.error_summary else "Unknown verification failure"
                lines.append(f"Attempt {attempt.attempt_number}: {preview}")
                continue

            primary = diagnostics[0]
            line = f"Attempt {attempt.attempt_number}: {primary.obligation_kind} at line {primary.line}"
            if primary.call_name:
                line += f" around {primary.call_name}(...)"
            if primary.failing_text:
                line += f" | failing code: {primary.failing_text}"
            if primary.related_file and primary.related_line:
                line += f" | related contract: {Path(primary.related_file).name}:{primary.related_line}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _remove_marked_comment_block(text: str, begin_marker: str, end_marker: str) -> str:
        """Remove a generated comment block while leaving the strategy code intact."""
        lines = text.splitlines()
        output: list[str] = []
        i = 0
        while i < len(lines):
            if lines[i].strip() == begin_marker:
                j = i + 1
                while j < len(lines) and lines[j].strip() != end_marker:
                    j += 1
                if j < len(lines):
                    i = j + 1
                    continue
            output.append(lines[i])
            i += 1
        return "\n".join(output).strip()

    def _get_strategy_body_for_evaluation_history(self, strategy_code: str) -> str:
        """Return strategy code without rationale/proof-sketch prose."""
        body = extract_rationale(strategy_code).body_without_rationale
        body = self._remove_marked_comment_block(
            body,
            "// CSD_PROOF_SKETCH_BEGIN",
            "// CSD_PROOF_SKETCH_END",
        )
        return body.strip()

    def _get_helper_calls_for_evaluation_history(self, strategy_code: str) -> list[str]:
        """Return model-facing helper/CSDHelpers calls used by a strategy body."""
        body = self._get_strategy_body_for_evaluation_history(strategy_code)
        calls = re.findall(r"\b(?:helpers|CSDHelpers)\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", body)
        return sorted(set(calls))

    @staticmethod
    def _extract_helper_universe_from_prompts() -> set[str]:
        """Extract helper method names referenced in the prompt tool API docs."""
        prompt_helper_names = getattr(generation_prompts, "_ALL_HELPER_NAMES", None)
        if prompt_helper_names is not None:
            return set(prompt_helper_names)
        tool_reference = getattr(
            generation_prompts,
            "TOOL_REFERENCE",
            generation_prompts.SYSTEM_PROMPT,
        )
        calls = re.findall(
            r"\b(?:helpers|CSDHelpers)\.([A-Za-z_][A-Za-z0-9_]*)\b",
            tool_reference,
        )
        return set(calls)

    def _evaluation_scalar_score(self, result: EvaluationResult) -> float:
        """Scalar score used for helper utility estimates and goodness."""
        return evaluation_scalar_score(
            result,
            require_delimiters=self.require_delimiters,
            eval_max_seconds_per_example=self.eval_max_seconds_per_example,
        )

    def _compute_attempt_goodness(self, attempt: SynthesisAttempt) -> float:
        return compute_goodness_from_attempt(
            attempt,
            min_accuracy=self.min_accuracy,
            min_syntax_rate=self.min_syntax_rate,
            require_delimiters=self.require_delimiters,
            eval_max_seconds_per_example=self.eval_max_seconds_per_example,
        )

    def _attempt_met_threshold(self, attempt: SynthesisAttempt) -> bool:
        if attempt.eval_result is None:
            return False
        if attempt.eval_result.early_stopped:
            return False
        return attempt.eval_result.meets_threshold(
            min_accuracy=self.min_accuracy,
            min_syntax_rate=self.min_syntax_rate,
            require_delimiters=self.require_delimiters,
            max_seconds_per_example=self.eval_max_seconds_per_example,
        )

    def _compute_prunable_helper_marginals(
        self,
        evaluated_attempts: list[SynthesisAttempt],
    ) -> dict[str, tuple[float, int, bool]]:
        """Counterfactual marginal credit for each prunable helper.

        For each prunable helper that appeared in at least one attempt, credit
        it by `mean(scalar | helper used) - mean(scalar | helper NOT used)`.
        This isolates the helper's own contribution instead of appending the
        full composite scalar of every attempt it co-occurred in (which let a
        slow/bad co-occurring helper drag down a helper that actually drove
        accuracy).

        Returns {helper: (marginal, n_with, ubiquitous)} where:
          - marginal: with_mean - without_mean (0.0 when ubiquitous)
          - n_with: number of attempts that used the helper (UCB pull count)
          - ubiquitous: True when EVERY attempt used the helper (no "without"
            set; marginal is undefined, so the bandit protects rather than
            prunes it)
        Untried prunable helpers (n_with == 0) are omitted; the bandit keeps
        every untried helper on the menu separately.
        """
        per_attempt: list[tuple[float, set[str]]] = []
        for attempt in evaluated_attempts:
            score = self._evaluation_scalar_score(attempt.eval_result)
            used = set(self._get_helper_calls_for_evaluation_history(attempt.strategy_code))
            per_attempt.append((score, used))

        marginals: dict[str, tuple[float, int, bool]] = {}
        for helper in (self.PRUNABLE_HELPERS & self._helper_universe):
            with_scores = [s for s, used in per_attempt if helper in used]
            without_scores = [s for s, used in per_attempt if helper not in used]
            if not with_scores:
                continue
            if not without_scores:
                marginals[helper] = (0.0, len(with_scores), True)
                continue
            with_mean = sum(with_scores) / len(with_scores)
            without_mean = sum(without_scores) / len(without_scores)
            marginals[helper] = (with_mean - without_mean, len(with_scores), False)
        return marginals

    def _pareto_best_prunable_helpers(
        self,
        evaluated_attempts: list[SynthesisAttempt],
        prunable_pool: list[str],
    ) -> set[str]:
        """Prunable helpers used by the pareto-best (refinement-anchor) attempt.

        Seeds the bandit's keep-set from the SAME anchor the author is told to
        beat (the threshold-shortfall pareto-best), so the protected helpers
        belong to the branch being chased rather than to whichever attempt
        happened to win the composite scalar. Falls back to the scalar-best
        attempt when no pareto anchor exists (e.g. nothing evaluated on >=1
        example yet).
        """
        pareto_n, _, _ = self._compute_pareto_best(evaluated_attempts)
        best_attempt = next(
            (a for a in evaluated_attempts if a.attempt_number == pareto_n),
            None,
        )
        if best_attempt is None:
            best_attempt = max(
                evaluated_attempts,
                key=lambda attempt: self._evaluation_scalar_score(attempt.eval_result),
            )
        best_helpers = set(self._get_helper_calls_for_evaluation_history(best_attempt.strategy_code))
        return best_helpers & set(prunable_pool)

    def _compute_allowed_helpers_utility(
        self,
        evaluated_attempts: list[SynthesisAttempt],
    ) -> tuple[list[str], str]:
        """With-vs-without comparative helper pruning.

        For each prunable helper, compare the mean evaluation score of
        attempts that USED the helper to the mean score of attempts that DID
        NOT use it. Prune the helper when `without_mean - with_mean >=
        helper_mask_margin` — i.e., not using the helper is empirically
        better than using it by at least the margin.

        This replaces the old "compare with-mean to run-wide baseline-mean"
        rule, which had a blind spot: if a harmful helper was used in MOST
        attempts, its with-mean and the baseline-mean would be very close
        (because the baseline mean was itself dragged down by the helper),
        so the helper never crossed the pruning threshold even when removing
        it would have improved scores. The with-vs-without rule isolates the
        helper's marginal effect and surfaces ubiquitous-but-harmful helpers
        as well as occasional-but-harmful ones.
        """
        allowed_helpers = set(self._helper_universe)
        if len(evaluated_attempts) < self.helper_mask_min_evals:
            return sorted(allowed_helpers), (
                f"helper mask warm-up ({len(evaluated_attempts)}/{self.helper_mask_min_evals} evaluated attempts)"
            )

        best_attempt = max(
            evaluated_attempts,
            key=lambda attempt: self._evaluation_scalar_score(attempt.eval_result),
        )
        best_helpers = set(self._get_helper_calls_for_evaluation_history(best_attempt.strategy_code))

        # Precompute (score, used_helpers) for every evaluated attempt once.
        per_attempt: list[tuple[float, set[str]]] = []
        for attempt in evaluated_attempts:
            score = self._evaluation_scalar_score(attempt.eval_result)
            used = set(self._get_helper_calls_for_evaluation_history(attempt.strategy_code))
            per_attempt.append((score, used))

        # Universe of prunable helpers: prunable ∩ helper_universe, minus
        # NON_PRUNABLE (locked-on protections) and minus the helpers used by
        # the current best attempt (never prune what's working).
        prunable_candidates = (
            (self.PRUNABLE_HELPERS & self._helper_universe)
            - self.NON_PRUNABLE_HELPERS
            - best_helpers
        )

        harmful_helpers: list[tuple[float, int, int, str, float, float]] = []
        for helper in prunable_candidates:
            with_scores = [s for s, used in per_attempt if helper in used]
            without_scores = [s for s, used in per_attempt if helper not in used]
            if len(with_scores) < self.helper_mask_min_uses:
                # Not enough data on the "with" side to make a call.
                continue
            if not without_scores:
                # Every attempt used this helper; with-vs-without is
                # undefined. Keep helper enabled and wait for opus to drop it.
                continue
            with_mean = sum(with_scores) / len(with_scores)
            without_mean = sum(without_scores) / len(without_scores)
            delta = without_mean - with_mean  # positive = harmful
            if delta >= self.helper_mask_margin:
                # Sort key: most harmful (largest delta) first, then more
                # "with" usages (stronger evidence), then alphabetical.
                harmful_helpers.append(
                    (-delta, -len(with_scores), -len(without_scores), helper, with_mean, without_mean)
                )

        harmful_helpers.sort()
        disabled: list[str] = []
        disabled_with_evidence: list[str] = []
        for neg_delta, _n_with, _n_without, helper, with_mean, without_mean in harmful_helpers:
            if len(disabled) >= self.helper_mask_max_disabled:
                break
            disabled.append(helper)
            delta = -neg_delta
            disabled_with_evidence.append(
                f"{helper} (with={with_mean:.2f}, without={without_mean:.2f}, "
                f"Δ=+{delta:.2f})"
            )

        allowed_helpers -= set(disabled)
        if disabled:
            status = (
                "helper mask active (with-vs-without utility); disabled: "
                + ", ".join(disabled_with_evidence)
            )
        else:
            status = (
                "helper mask active (with-vs-without utility); "
                "no helpers showed a harmful with-vs-without delta yet"
            )
        return sorted(allowed_helpers), status

    def _compute_allowed_helpers_bandit(
        self,
        evaluated_attempts: list[SynthesisAttempt],
    ) -> tuple[list[str], str]:
        """Bandit-style helper selection using UCB over prunable helpers."""
        allowed_helpers = set(self._helper_universe)
        if len(evaluated_attempts) < self.helper_bandit_min_evals:
            return sorted(allowed_helpers), (
                f"helper mask warm-up ({len(evaluated_attempts)}/{self.helper_bandit_min_evals} evaluated attempts for bandit)"
            )

        prunable_pool = sorted(self.PRUNABLE_HELPERS & self._helper_universe)
        if not prunable_pool:
            return sorted(allowed_helpers), "helper bandit active; no prunable helpers in universe"

        marginals = self._compute_prunable_helper_marginals(evaluated_attempts)
        pulls = {helper: (marginals[helper][1] if helper in marginals else 0) for helper in prunable_pool}
        # UCB exploitation term is each helper's counterfactual marginal
        # (with-minus-without), not the absolute mean scalar of attempts using
        # it -- so a helper is not credited or penalised for what its
        # co-occurring helpers did.
        means = {helper: (marginals[helper][0] if helper in marginals else 0.0) for helper in prunable_pool}
        ubiquitous = {helper for helper in prunable_pool if helper in marginals and marginals[helper][2]}

        # Protect the helpers used by the pareto-best (refinement-anchor)
        # attempt -- the same branch the author is told to beat -- rather than
        # whichever attempt won the composite scalar.
        keep_prunable = set(self._pareto_best_prunable_helpers(evaluated_attempts, prunable_pool))
        # Helpers used by EVERY attempt have no counterfactual signal; protect
        # them from pruning rather than ranking them out on exploration alone.
        keep_prunable.update(ubiquitous)

        total_pulls = max(1, sum(pulls.values()))
        # Keep EVERY untried helper on the menu each iteration. An arm with zero
        # pulls has no evidence against it, so pruning it would hide a helper the
        # author never had the chance to try (the old policy kept only the single
        # alphabetically-first untried helper, which froze the rest out for the
        # whole run in low-adoption cases). The mask still prunes helpers that HAVE
        # been tried and did worse -- see the UCB ranking + target_size below --
        # it just never prunes untried ones. (User ruling 2026-06-19.)
        untried = [helper for helper in prunable_pool if pulls[helper] == 0 and helper not in keep_prunable]
        keep_prunable.update(untried)

        ranked_tried = sorted(
            (helper for helper in prunable_pool if pulls[helper] > 0 and helper not in keep_prunable),
            key=lambda helper: (
                means[helper]
                + self.helper_bandit_ucb_c
                * math.sqrt(math.log(total_pulls + 1.0) / pulls[helper])
            ),
            reverse=True,
        )

        target_size = min(max(self.helper_bandit_top_k, len(keep_prunable)), len(prunable_pool))
        for helper in ranked_tried:
            if len(keep_prunable) >= target_size:
                break
            keep_prunable.add(helper)

        disabled = sorted(set(prunable_pool) - keep_prunable)
        allowed_helpers -= set(disabled)
        n_untried_kept = sum(1 for helper in keep_prunable if pulls[helper] == 0)
        status = (
            "helper mask active (bandit/UCB); "
            f"kept {len(keep_prunable)}/{len(prunable_pool)} prunable helpers "
            f"(top_k={self.helper_bandit_top_k}, all {n_untried_kept} untried kept, "
            "only tried-and-worse helpers pruned)"
        )
        return sorted(allowed_helpers), status

    def _update_anchor_state(self, new_anchor_attempt_number: int | None) -> None:
        """Update the "iters since anchor moved" counter after an eval.

        Called once per eval, AFTER the new Pareto-best has been computed.
        Resets the counter when the anchor identity changes; otherwise ticks.
        First-ever anchor assignment counts as "moved" → counter starts at 0.
        """
        if new_anchor_attempt_number is None:
            return
        if new_anchor_attempt_number != self._anchor_attempt_number:
            self._anchor_attempt_number = new_anchor_attempt_number
            self._iters_since_anchor_changed = 0
        else:
            self._iters_since_anchor_changed += 1

    def _should_restart(self, attempts: list[SynthesisAttempt]) -> bool:
        """Predicate: should the next refinement use restart mode?

        True iff restart is enabled, at least one prior evaluated attempt
        exists (so the families-tried block has content), and the counter has
        reached the configured threshold.
        """
        if self.restart_after_stuck_iters <= 0:
            return False
        evaluated = [
            a for a in attempts
            if a.eval_result is not None
            and (a.eval_result.num_examples or 0) > 0
        ]
        if not evaluated:
            return False
        return self._iters_since_anchor_changed >= self.restart_after_stuck_iters

    def _apply_restart_cooldown(self) -> None:
        """Reset counter after a restart fires.

        With cooldown=0 we leave the counter alone — it keeps ticking on
        subsequent stuck iters so restart fires every iter until the anchor
        moves (true Option X / aggressive exploration). With cooldown=K we
        set the counter to -K so K refinement iters must run before another
        restart is eligible (Option Y / balanced rhythm).

        Earlier version unconditionally set counter to -cooldown, which with
        cooldown=0 forced counter to 0 — same effect as a 2-iter cooldown.
        This guard makes cooldown=0 do what the flag advertises.
        """
        if self.restart_cooldown_iters > 0:
            self._iters_since_anchor_changed = -self.restart_cooldown_iters

    def _compute_pareto_best(
        self, attempts: list[SynthesisAttempt]
    ) -> tuple[int | None, float | None, float | None]:
        """Identify the best refinement anchor by threshold shortfall.

        Only attempts with a completed evaluation on >=1 example are eligible.
        The anchor should be the evaluated attempt closest to satisfying the
        configured thresholds, not simply the highest-accuracy attempt. This
        keeps refinement anchored on branches that are closest to a full win
        when accuracy and syntax trade off.
        """
        candidates = [
            a for a in attempts
            if a.eval_result is not None
            and (a.eval_result.num_examples or 0) > 0
        ]
        if not candidates:
            return None, None, None

        def anchor_key(attempt: SynthesisAttempt) -> tuple[float, float, float, int]:
            result = attempt.eval_result
            accuracy = float(result.accuracy or 0.0)
            syntax_rate = float(result.syntax_rate or 0.0)
            accuracy_shortfall = max(0.0, self.min_accuracy - accuracy)
            syntax_shortfall = max(0.0, self.min_syntax_rate - syntax_rate)
            total_shortfall = accuracy_shortfall + syntax_shortfall
            return (
                total_shortfall,
                -accuracy,
                -syntax_rate,
                attempt.attempt_number,
            )

        best = min(
            candidates,
            key=anchor_key,
        )
        return (
            best.attempt_number,
            best.eval_result.accuracy,
            best.eval_result.syntax_rate,
        )

    def _lookup_best_so_far(
        self,
        attempts: list[SynthesisAttempt],
        anchor_attempt_number: int | None,
        current_attempt: SynthesisAttempt,
        current_strategy_code: str,
        current_eval_result: EvaluationResult,
    ) -> tuple[str, float, float]:
        """Resolve the best-so-far (strategy_code, accuracy, syntax_rate).

        Falls back to the current attempt's values when no prior Pareto-best
        exists yet (e.g. first failed evaluation).
        """
        if anchor_attempt_number is not None:
            best = next(
                (a for a in attempts if a.attempt_number == anchor_attempt_number),
                None,
            )
            if (
                best is not None
                and best.eval_result is not None
                and (best.eval_result.num_examples or 0) > 0
            ):
                return (
                    best.strategy_code or current_strategy_code,
                    best.eval_result.accuracy or 0.0,
                    best.eval_result.syntax_rate or 0.0,
                )
        return (
            current_strategy_code,
            current_eval_result.accuracy or 0.0,
            current_eval_result.syntax_rate or 0.0,
        )

    def _compute_allowed_helpers(self, attempts: list[SynthesisAttempt]) -> tuple[list[str] | None, str]:
        """
        Build a per-attempt helper-call contract from empirical policy.

        Returns:
            (allowed_helpers, status_text). `allowed_helpers=None` disables the
            contract block in prompts.
        """
        if not self.adaptive_helper_mask or not self._helper_universe:
            return None, ""

        # Attempts whose evaluation completed zero examples carry no real signal
        # for the helper mask: their scalar score reflects defaults (0 accuracy,
        # 0 syntax_rate) rather than actual helper behavior. Excluding them keeps
        # the utility/bandit policies from training on phantom rewards.
        with_eval = [attempt for attempt in attempts if attempt.eval_result is not None]
        evaluated = [
            attempt for attempt in with_eval
            if (attempt.eval_result.num_examples or 0) > 0
        ]
        skipped_zero_n = len(with_eval) - len(evaluated)
        allowed, status = self._compute_allowed_helpers_bandit(evaluated)
        if skipped_zero_n:
            suffix = (
                f"; excluded {skipped_zero_n} attempt(s) with zero evaluated "
                "examples from helper-mask scoring"
            )
            status = (status + suffix) if status else suffix.lstrip("; ")
        return allowed, status

    def _get_disallowed_helper_calls(
        self,
        strategy_code: str,
        allowed_helpers: list[str] | None,
    ) -> list[str]:
        """Return helper calls in strategy_code that violate the active contract."""
        if not allowed_helpers:
            return []
        allowed = set(allowed_helpers)
        used = set(self._get_helper_calls_for_evaluation_history(strategy_code))
        return sorted(used - allowed)

    def _build_attempt_outcome_ledger(
        self,
        attempts: list[SynthesisAttempt],
        best_attempt_number: int | None,
    ) -> str:
        """Build compact empirical context from evaluated attempts."""
        evaluated = [
            attempt for attempt in attempts
            if attempt.eval_result is not None
            and (attempt.eval_result.num_examples or 0) > 0
        ]
        if not evaluated:
            return ""

        best_attempt = next(
            (attempt for attempt in evaluated if attempt.attempt_number == best_attempt_number),
            None,
        )
        if best_attempt is None:
            best_attempt = max(
                evaluated,
                key=lambda attempt: self._evaluation_scalar_score(attempt.eval_result),
            )
        best_result = best_attempt.eval_result
        if best_result is None:
            return ""

        recent = [
            attempt for attempt in evaluated
            if attempt.attempt_number != best_attempt.attempt_number
        ][-3:]

        def rationale_summary(attempt: SynthesisAttempt) -> str:
            rationale = extract_rationale(attempt.strategy_code).rationale
            if not rationale:
                return "none captured"
            summarizer = getattr(self.generator, "summarize_rationale_claim", None)
            if callable(summarizer):
                try:
                    return str(summarizer(rationale)).strip() or rationale
                except Exception:
                    return rationale
            return rationale

        def failure_locations(result: EvaluationResult) -> str:
            counts: dict[str, int] = {}
            for sample in result.sample_outputs or []:
                if sample.get("is_correct"):
                    continue
                location = str(sample.get("failure_location") or "unknown")
                counts[location] = counts.get(location, 0) + 1
            if not counts:
                return "none"
            return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))

        def attempt_line(attempt: SynthesisAttempt, *, include_delta: bool) -> list[str]:
            result = attempt.eval_result
            if result is None:
                return []
            lines = [
                (
                    f"- Attempt {attempt.attempt_number}: "
                    f"accuracy {result.accuracy:.1%}, syntax {result.syntax_rate:.1%}; "
                    f"rationale claim: {rationale_summary(attempt)}"
                )
            ]
            if include_delta:
                acc_delta = (result.accuracy - best_result.accuracy) * 100.0
                syn_delta = (result.syntax_rate - best_result.syntax_rate) * 100.0
                lines.append(
                    f"  measured effect vs best: accuracy {acc_delta:+.1f}pp, "
                    f"syntax {syn_delta:+.1f}pp"
                )
            lines.append(f"  failure locations: {failure_locations(result)}")
            return lines

        lines = [
            "Use this as empirical search context, not as a recipe.",
            "Best result:",
            *attempt_line(best_attempt, include_delta=False),
        ]
        if recent:
            lines.append("Recent evaluated branches:")
            for attempt in recent:
                lines.extend(attempt_line(attempt, include_delta=True))
        return "\n".join(lines)

    @staticmethod
    def _strategy_change_ratio(before: str, after: str) -> float:
        """Return an approximate changed-line ratio between two strategy bodies."""
        before_lines = before.splitlines()
        after_lines = after.splitlines()
        diff_lines = list(
            unified_diff(
                before_lines,
                after_lines,
                fromfile="before",
                tofile="after",
                lineterm="",
                n=0,
            )
        )
        changed = sum(
            1
            for line in diff_lines
            if line and line[0] in {"+", "-"} and not line.startswith(("+++", "---"))
        )
        denom = max(1, len(before_lines))
        return changed / denom

    @staticmethod
    def _normalized_strategy_key(strategy_code: str) -> str:
        """Normalize strategy text for duplicate suppression in beam search."""
        return re.sub(r"\s+", " ", strategy_code).strip()

    def _helper_overlap_ratio(self, before: str, after: str) -> float:
        """Jaccard overlap of helper call sets between strategies."""
        before_set = set(self._get_helper_calls_for_evaluation_history(before))
        after_set = set(self._get_helper_calls_for_evaluation_history(after))
        union = before_set | after_set
        if not union:
            return 1.0
        return len(before_set & after_set) / len(union)

    def _refine_with_beam(
        self,
        *,
        stage_label: str,
        previous_strategy: str,
        allowed_helpers: list[str] | None,
        refine_once: Callable[[], str],
    ) -> str:
        """
        Sample multiple local refinements and select the best candidate.

        Ranking preference:
        1) helper-call contract satisfaction,
        2) optional pre-verification success (if enabled),
        3) local-neighborhood preference,
        4) helper-set overlap with previous strategy,
        5) smaller edit ratio.
        """
        beam_size = max(1, self.refinement_beam_size)
        if beam_size == 1:
            return refine_once()

        candidates: list[dict] = []
        seen: set[str] = {self._normalized_strategy_key(previous_strategy)}

        for _ in range(beam_size):
            candidate = refine_once()
            key = self._normalized_strategy_key(candidate)
            if not key or key in seen:
                continue
            seen.add(key)

            disallowed = self._get_disallowed_helper_calls(candidate, allowed_helpers)
            contract_ok = len(disallowed) == 0
            change_ratio = self._strategy_change_ratio(previous_strategy, candidate)
            locality_ok = (not self.local_neighborhood_refinement) or (
                change_ratio <= self.max_local_edit_ratio
            )
            overlap = self._helper_overlap_ratio(previous_strategy, candidate)

            verification_ok = False
            if self.beam_verify_candidates:
                try:
                    verification_ok = self.verifier.verify(
                        self.generator.inject_strategy(candidate)
                    ).success
                except Exception:
                    verification_ok = False

            score = (
                1 if contract_ok else 0,
                1 if verification_ok else 0,
                1 if locality_ok else 0,
                overlap,
                -change_ratio,
                -len(disallowed),
            )
            candidates.append(
                {
                    "strategy": candidate,
                    "score": score,
                    "contract_ok": contract_ok,
                    "verification_ok": verification_ok,
                    "locality_ok": locality_ok,
                    "change_ratio": change_ratio,
                    "disallowed": disallowed,
                }
            )

        if not candidates:
            return refine_once()

        best = max(candidates, key=lambda item: item["score"])
        print(
            f"  Beam {stage_label}: {len(candidates)} candidate(s), "
            f"selected contract_ok={best['contract_ok']} "
            f"verify_ok={best['verification_ok']} "
            f"local={best['locality_ok']} "
            f"edit_ratio={best['change_ratio']:.2f}"
        )
        if best["disallowed"]:
            print(
                "  Beam selected candidate still violates helper contract: "
                + ", ".join(best["disallowed"])
            )
        return best["strategy"]

    def _run_attempt_pipeline(
        self,
        *,
        strategy_code: str,
        attempt_num: int,
        node_id: int,
        parent_node_id: int | None,
        allowed_helpers: list[str] | None,
        compiler: DafnyCompiler,
        output_name: str,
    ) -> SynthesisAttempt:
        """Verify, compile, and evaluate one strategy snapshot without refining."""
        full_code = self.generator.inject_strategy(strategy_code)
        attempt = SynthesisAttempt(
            attempt_number=attempt_num,
            strategy_code=strategy_code,
            full_dafny_code=full_code,
            timestamp=datetime.now().isoformat(),
            node_id=node_id,
            parent_node_id=parent_node_id,
        )

        disallowed_helpers = self._get_disallowed_helper_calls(strategy_code, allowed_helpers)
        if disallowed_helpers:
            print("  ✗ Strategy contract violation")
            attempt.failed_at = FailureStage.SEARCH_CONTRACT
            attempt.error_summary = (
                "Strategy contract violation.\n"
                f"Violations: {', '.join(disallowed_helpers)}"
            )
            attempt.goodness = 0.0
            return attempt

        print("\n[1/4] Verifying with Dafny...")
        verification_result = self.verifier.verify(full_code)
        attempt.verification_result = verification_result
        if not verification_result.success:
            print("  ✗ Verification failed")
            print(f"  Error: {verification_result.get_error_summary()[:300]}")
            attempt.failed_at = FailureStage.VERIFICATION
            attempt.error_summary = verification_result.get_error_summary()
            attempt.goodness = 0.0
            return attempt
        print("  ✓ Verification passed")

        print("\n[2/4] Compiling to Python...")
        compilation_result = compiler.compile(full_code, output_name)
        attempt.compilation_result = compilation_result
        if not compilation_result.success:
            print("  ✗ Compilation failed")
            attempt.failed_at = FailureStage.COMPILATION
            attempt.error_summary = compilation_result.get_error_summary()
            attempt.goodness = 0.0
            return attempt
        print(f"  ✓ Compiled to {compilation_result.output_dir}")

        if compilation_result.main_module_path is None:
            print("  ✗ No main module found")
            attempt.failed_at = FailureStage.RUNTIME
            attempt.error_summary = "No main module path in compilation result"
            attempt.goodness = 0.0
            return attempt

        print("\n[3/4] Evaluating compiled strategy (runtime smoke test removed).")
        print("\n[4/4] Evaluating on dataset sample...")
        if self.generator._model is not None:
            del self.generator._model
            self.generator._model = None
            import gc
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            print("  Generator model (HF) unloaded to free GPU memory")
        if getattr(self.generator, "_vllm", None) is not None:
            import gc
            import torch
            vllm_obj = self.generator._vllm
            self.generator._vllm = None
            try:
                vllm_obj._run_engine = None
            except Exception:
                pass
            del vllm_obj
            try:
                from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            print("  Generator vllm engine unloaded to free GPU memory")

        print(f"  [synthesis] eval seed for this iteration: {self.evaluator.sample_seed}")
        eval_result = self.evaluator.evaluate_sample(
            compiled_module_path=compilation_result.main_module_path,
            sample_size=self.eval_sample_size,
            early_stop_min_accuracy=self.min_accuracy
            if os.environ.get("CSD_SYNTHESIS_EVAL_EARLY_STOP", "1") != "0"
            else None,
            early_stop_min_syntax_rate=self.min_syntax_rate
            if os.environ.get("CSD_SYNTHESIS_EVAL_EARLY_STOP", "1") != "0"
            else None,
            early_stop_runtime_failures=(
                int(os.environ.get("CSD_SYNTHESIS_EVAL_EARLY_STOP_RUNTIME_FAILURES", "3"))
                if os.environ.get("CSD_SYNTHESIS_EVAL_EARLY_STOP", "1") != "0"
                else None
            ),
            min_examples_before_threshold_stop=self.min_examples_before_threshold_stop,
        )
        if not hasattr(self, "_failure_ledger"):
            try:
                from synthesis.failure_taxonomy import make_persistent_ledger
            except ImportError:
                from failure_taxonomy import make_persistent_ledger
            self._failure_ledger = make_persistent_ledger()
        eval_result._failure_ledger = self._failure_ledger
        eval_result._attempt_index = attempt.attempt_number
        attempt.eval_result = eval_result

        smiles_trial = (eval_result.aux_metrics or {}).get("smiles_paper_trial", {})
        if isinstance(smiles_trial, dict) and smiles_trial:
            membership = smiles_trial.get("membership")
            validity = smiles_trial.get("validity_rdkit")
            samples_to_target = smiles_trial.get("samples_to_target_unique_valid")
            unique_valid = smiles_trial.get("unique_valid_count")
            sample_count = smiles_trial.get("sample_count")
            print("  [smiles] paper-aligned metrics:")
            if membership is not None:
                print(f"    Membership: {float(membership):.1%}")
            if validity is not None:
                print(f"    RDKit Validity: {float(validity):.1%}")
            print(
                "    Samples to 100 unique valid (cap 1000): "
                f"{samples_to_target}"
            )
            print(
                "    Unique valid molecules this eval: "
                f"{unique_valid}/{sample_count}"
            )

        if not eval_result.success:
            print(f"  ✗ Evaluation failed: {eval_result.error}")
            attempt.failed_at = FailureStage.EVALUATION
            attempt.error_summary = eval_result.error or "Evaluation failed"
            self._unload_evaluator_runtime_before_refinement()
            attempt.goodness = self._compute_attempt_goodness(attempt)
            return attempt

        attempt.met_threshold = self._attempt_met_threshold(attempt)
        if not attempt.met_threshold:
            print("  ✗ Evaluation below threshold:")
            print(f"    Accuracy: {eval_result.accuracy:.1%} (min: {self.min_accuracy:.1%})")
            if self.require_delimiters:
                print(
                    "    Contains << >>: "
                    f"{'yes' if eval_result.contains_delimiters else 'no'} "
                    f"(required: {'yes' if self.require_delimiters else 'no'})"
                )
            print(f"    Syntax: {eval_result.syntax_rate:.1%} (min: {self.min_syntax_rate:.1%})")
            attempt.failed_at = FailureStage.EVALUATION
            attempt.error_summary = eval_result.get_feedback_summary(self.require_delimiters)
            self._unload_evaluator_runtime_before_refinement()
        else:
            print("  ✓ Evaluation passed:")
            print(f"    Accuracy: {eval_result.accuracy:.1%}")
            print(f"    Contains << >>: {'yes' if eval_result.contains_delimiters else 'no'}")
            print(f"    Syntax: {eval_result.syntax_rate:.1%}")

        attempt.goodness = self._compute_attempt_goodness(attempt)
        return attempt

    def _produce_child_code(
        self,
        parent: SearchNode,
        attempts: list[SynthesisAttempt],
        task_description: str,
        allowed_helpers: list[str] | None,
    ) -> str:
        """Refine a selected tree arm into a child strategy body."""
        strategy_code = parent.strategy_code
        anchor_n, anchor_acc, anchor_syn = self._compute_pareto_best(attempts)
        if anchor_n is not None:
            print(
                f"  [synthesis] anchor for refinement: attempt {anchor_n} "
                f"(acc={anchor_acc:.1%}, syn={anchor_syn:.1%})"
            )
        attempt_outcome_ledger = self._build_attempt_outcome_ledger(attempts, anchor_n)

        if parent.failed_at == FailureStage.SEARCH_CONTRACT:
            error_msg = parent.error_summary
            return self._refine_with_beam(
                stage_label="search_contract",
                previous_strategy=strategy_code,
                allowed_helpers=allowed_helpers,
                refine_once=lambda: self.generator.refine_after_verification_error(
                    strategy_code,
                    error_msg,
                    allowed_helpers=allowed_helpers,
                ),
            )

        if parent.failed_at == FailureStage.VERIFICATION:
            error_msg = parent.error_summary
            structured_feedback = (
                parent.verification_result.get_structured_feedback()
                if parent.verification_result is not None
                else None
            )
            error_history = self._get_verification_history_summary(attempts)
            behavioral_context = self._get_recent_behavioral_context(attempts)
            return self._refine_with_beam(
                stage_label="verification",
                previous_strategy=strategy_code,
                allowed_helpers=allowed_helpers,
                refine_once=lambda: self.generator.refine_after_verification_error(
                    strategy_code,
                    error_msg,
                    behavioral_context=behavioral_context,
                    structured_feedback=structured_feedback,
                    error_history=error_history,
                    allowed_helpers=allowed_helpers,
                ),
            )

        if parent.failed_at == FailureStage.COMPILATION:
            error_msg = (
                parent.compilation_result.get_error_summary()
                if parent.compilation_result is not None
                else parent.error_summary
            )
            return self._refine_with_beam(
                stage_label="compilation",
                previous_strategy=strategy_code,
                allowed_helpers=allowed_helpers,
                refine_once=lambda: self.generator.refine_after_compilation_error(
                    strategy_code,
                    error_msg,
                    allowed_helpers=allowed_helpers,
                ),
            )

        if parent.failed_at == FailureStage.RUNTIME:
            return self._refine_with_beam(
                stage_label="runtime",
                previous_strategy=strategy_code,
                allowed_helpers=allowed_helpers,
                refine_once=lambda: self.generator.refine_after_runtime_error(
                    strategy_code,
                    parent.error_summary or "Runtime failure",
                    allowed_helpers=allowed_helpers,
                ),
            )

        eval_result = parent.eval_result
        if eval_result is None:
            return self.generator.generate_initial(
                task_description,
                allowed_helpers=allowed_helpers,
            )

        best_strategy_code, best_acc_val, best_syn_val = self._lookup_best_so_far(
            attempts,
            anchor_n,
            SynthesisAttempt(
                attempt_number=parent.attempt_number,
                strategy_code=strategy_code,
                full_dafny_code=parent.full_dafny_code,
                timestamp=parent.timestamp,
                eval_result=eval_result,
            ),
            strategy_code,
            eval_result,
        )
        refine_best_strategy = (
            best_strategy_code if best_strategy_code != strategy_code else None
        )
        refine_best_acc = best_acc_val if refine_best_strategy is not None else None
        refine_best_syn = best_syn_val if refine_best_strategy is not None else None
        prev_acc = eval_result.accuracy or 0.0
        prev_syn = eval_result.syntax_rate or 0.0
        prev_n = eval_result.num_examples or 0

        if not eval_result.success:
            evaluation_feedback = (
                eval_result.get_feedback_summary(self.require_delimiters)
                + _final_span_failure_hint(
                    self.require_delimiters, eval_result.sample_outputs
                )
                + _unit_rewind_hint(strategy_code, eval_result.sample_outputs)
            )
            mode_examples = eval_result._render_mode_examples()
            return self._refine_with_beam(
                stage_label="evaluation_error",
                previous_strategy=strategy_code,
                allowed_helpers=allowed_helpers,
                refine_once=lambda: self.generator.refine_after_evaluation_failure(
                    previous_strategy=strategy_code,
                    previous_accuracy=prev_acc,
                    previous_syntax_rate=prev_syn,
                    num_examples=prev_n,
                    goal_accuracy=self.min_accuracy,
                    goal_syntax_rate=self.min_syntax_rate,
                    evaluation_feedback=evaluation_feedback,
                    best_strategy=refine_best_strategy,
                    best_accuracy=refine_best_acc,
                    best_syntax_rate=refine_best_syn,
                    allowed_helpers=allowed_helpers,
                    eval_max_seconds_per_example=self.eval_max_seconds_per_example,
                    mode_examples=mode_examples,
                    attempt_outcome_ledger=attempt_outcome_ledger,
                ),
            )

        threshold_feedback = (
            "Required thresholds:\n"
            f"  Accuracy: {self.min_accuracy:.1%}\n"
            f"  Syntax Rate: {self.min_syntax_rate:.1%}\n\n"
            + ("  Contains << >>: required\n" if self.require_delimiters else "")
            + (
                f"  Max Runtime / Example: {self.eval_max_seconds_per_example:.2f}s\n"
                if self.eval_max_seconds_per_example is not None
                else ""
            )
            + _delimiter_miss_hint(
                self.require_delimiters, eval_result.contains_delimiters,
                eval_result.sample_outputs
            )
            + _span_not_closed_hint(
                self.require_delimiters, eval_result.sample_outputs
            )
            + _constraint_bypassed_hint(
                self.require_delimiters, eval_result.contains_delimiters,
                eval_result.sample_outputs
            )
            + _final_span_failure_hint(
                self.require_delimiters, eval_result.sample_outputs
            )
            + _unit_rewind_hint(strategy_code, eval_result.sample_outputs)
            + "\n"
            + eval_result.get_feedback_summary(self.require_delimiters)
        )
        mode_examples = eval_result._render_mode_examples()
        stage_label = "evaluation_threshold" if not parent.met_threshold else "evaluation_improve"
        return self._refine_with_beam(
            stage_label=stage_label,
            previous_strategy=strategy_code,
            allowed_helpers=allowed_helpers,
            refine_once=lambda: self.generator.refine_after_evaluation_failure(
                previous_strategy=strategy_code,
                previous_accuracy=prev_acc,
                previous_syntax_rate=prev_syn,
                num_examples=prev_n,
                goal_accuracy=self.min_accuracy,
                goal_syntax_rate=self.min_syntax_rate,
                evaluation_feedback=threshold_feedback,
                best_strategy=refine_best_strategy,
                best_accuracy=refine_best_acc,
                best_syntax_rate=refine_best_syn,
                allowed_helpers=allowed_helpers,
                eval_max_seconds_per_example=self.eval_max_seconds_per_example,
                mode_examples=mode_examples,
                attempt_outcome_ledger=attempt_outcome_ledger,
            ),
        )

    def _write_fallback_winner(
        self,
        attempts: list[SynthesisAttempt],
        run_results_dir: Path,
    ) -> None:
        fallback_winner = None
        for att in attempts:
            if (
                att.eval_result is not None
                and not att.eval_result.early_stopped
                and att.eval_result.accuracy >= self.min_accuracy
                and (
                    fallback_winner is None
                    or att.eval_result.accuracy > fallback_winner.eval_result.accuracy
                )
            ):
                fallback_winner = att

        if fallback_winner is None:
            print(
                "[FALLBACK] No accuracy-only candidate found either "
                "(no attempt met the min_accuracy threshold); no fallback saved."
            )
            return

        fb_path = run_results_dir / "fallback_winner.json"
        try:
            with open(fb_path, "w") as f:
                json.dump(
                    {
                        "fallback_reason": "accuracy_met_but_syntax_below_threshold",
                        "min_accuracy": self.min_accuracy,
                        "min_syntax_rate": self.min_syntax_rate,
                        "winner_attempt_number": fallback_winner.attempt_number,
                        "winner_node_id": fallback_winner.node_id,
                        "winner_accuracy": fallback_winner.eval_result.accuracy,
                        "winner_syntax_rate": fallback_winner.eval_result.syntax_rate,
                        "winner_goodness": fallback_winner.goodness,
                        "winner_strategy_code": fallback_winner.strategy_code,
                        "winner_full_dafny_code": fallback_winner.full_dafny_code,
                        "evaluation": fallback_winner.eval_result.to_dict(),
                    },
                    f,
                    indent=2,
                )
            print(
                f"[FALLBACK] accuracy-only winner: attempt "
                f"{fallback_winner.attempt_number}, "
                f"acc={fallback_winner.eval_result.accuracy:.1%} "
                f"(syntax {fallback_winner.eval_result.syntax_rate:.1%} below "
                f"required {self.min_syntax_rate:.1%}). Saved to {fb_path}."
            )
        except Exception as fb_err:
            print(f"[FALLBACK] Failed to write fallback_winner.json: {fb_err}")

    def synthesize(
        self,
        task_description: str,
        output_name: str = "generated_csd",
        initial_strategy_code: str | None = None,
        initial_attempt_offset: int = 0,
    ) -> SynthesisResult:
        """
        Synthesize a CSD strategy using REx search over an explicit strategy tree.

        Returns the best-goodness node after the full iteration budget is consumed.
        """
        import time

        start_time = time.time()
        attempts: list[SynthesisAttempt] = []
        tree = SearchTree()
        rex = RexBandit(temperature=self.rex_temperature)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
        run_dir = self.output_dir / f"{output_name}_{run_id}"
        run_dafny_dir = run_dir / "dafny"
        run_python_dir = run_dir / "python"
        run_results_dir = run_dir / "results"
        run_dafny_dir.mkdir(parents=True, exist_ok=True)
        run_python_dir.mkdir(parents=True, exist_ok=True)
        run_results_dir.mkdir(parents=True, exist_ok=True)

        from synthesis.project_defaults import synthesis_prompt_log_dir

        prompt_log_dir = synthesis_prompt_log_dir(output_name, run_id)
        prompt_log_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CSD_PROMPT_LOG_DIR"] = str(prompt_log_dir)

        try:
            (self.output_dir / "latest_run.txt").write_text(str(run_dir) + "\n")
        except Exception:
            pass

        compiler = DafnyCompiler(
            dafny_path=self.compiler.dafny_path,
            output_dir=run_python_dir,
            timeout=self.compiler.timeout,
            extra_args=list(self.compiler.extra_args),
        )

        self.generator.set_synthesis_context(
            eval_model=self.evaluator.model_name,
            dataset=self.evaluator.dataset_name,
            max_steps=self.evaluator.max_steps,
            step_token_budget=self.evaluator.step_token_budget,
        )

        allowed_helpers, helper_status = self._compute_allowed_helpers(attempts)
        if helper_status:
            print(f"Helper policy: {helper_status}")

        if initial_strategy_code is not None:
            print("Using caller-provided initial strategy seed")
            root_strategy = initial_strategy_code
        else:
            print(f"Generating initial strategy for: {task_description}")
            root_strategy = self.generator.generate_initial(
                task_description,
                allowed_helpers=allowed_helpers,
            )

        attempt_total = initial_attempt_offset + self.max_iterations

        def _register_attempt(
            attempt: SynthesisAttempt,
            *,
            parent_id: int | None,
        ) -> SearchNode:
            node = tree.add_node(
                parent_id=parent_id,
                attempt_number=attempt.attempt_number,
                strategy_code=attempt.strategy_code,
                full_dafny_code=attempt.full_dafny_code,
                timestamp=attempt.timestamp,
                goodness=attempt.goodness,
                met_threshold=attempt.met_threshold,
                failed_at=attempt.failed_at,
                error_summary=attempt.error_summary,
                verification_result=attempt.verification_result,
                compilation_result=attempt.compilation_result,
                eval_result=attempt.eval_result,
            )
            attempt.node_id = node.node_id
            attempt.parent_node_id = parent_id
            return node

        # Bootstrap root (attempt 1)
        attempt_num = initial_attempt_offset + 1
        print(f"\n{'='*60}")
        print(f"Attempt {attempt_num}/{attempt_total} [bootstrap root]")
        print(f"{'='*60}")
        if helper_status:
            print(f"Helper policy: {helper_status}")
        print(f"Strategy: {root_strategy}")
        root_attempt = self._run_attempt_pipeline(
            strategy_code=root_strategy,
            attempt_num=attempt_num,
            node_id=tree._next_id,
            parent_node_id=None,
            allowed_helpers=allowed_helpers,
            compiler=compiler,
            output_name=output_name,
        )
        attempts.append(root_attempt)
        _register_attempt(root_attempt, parent_id=None)
        print(f"  Goodness: {root_attempt.goodness:.3f}")

        # REx pulls for remaining budget
        for iteration in range(1, self.max_iterations):
            attempt_num = initial_attempt_offset + iteration + 1
            allowed_helpers, helper_status = self._compute_allowed_helpers(attempts)
            parent = rex.select_arm(tree.all_nodes())
            print(f"\n{'='*60}")
            print(
                f"Attempt {attempt_num}/{attempt_total} "
                f"[REx pull from node {parent.node_id}, goodness={parent.goodness:.3f}]"
            )
            print(f"{'='*60}")
            if helper_status:
                print(f"Helper policy: {helper_status}")
            child_strategy = self._produce_child_code(
                parent,
                attempts,
                task_description,
                allowed_helpers,
            )
            print(f"Strategy: {child_strategy[:120]}{'...' if len(child_strategy) > 120 else ''}")
            child_attempt = self._run_attempt_pipeline(
                strategy_code=child_strategy,
                attempt_num=attempt_num,
                node_id=tree._next_id,
                parent_node_id=parent.node_id,
                allowed_helpers=allowed_helpers,
                compiler=compiler,
                output_name=output_name,
            )
            attempts.append(child_attempt)
            _register_attempt(child_attempt, parent_id=parent.node_id)
            rex.record_pull(parent, child_attempt.met_threshold)
            print(f"  Goodness: {child_attempt.goodness:.3f}")

        total_time = (time.time() - start_time) * 1000
        best_node = tree.best_by_goodness()
        best_attempt = next(
            a for a in attempts if a.node_id == best_node.node_id
        )
        met_threshold = best_attempt.met_threshold
        success = met_threshold

        print(f"\n{'='*60}")
        print(
            f"REx search complete after {len(attempts)} attempt(s); "
            f"best node {best_node.node_id} goodness={best_node.goodness:.3f}"
        )
        print(f"Total time: {total_time:.1f}ms")
        print(f"{'='*60}")

        compiled_module_path = None
        compilation_result = best_attempt.compilation_result
        if (
            compilation_result is not None
            and compilation_result.success
            and compilation_result.main_module_path is not None
        ):
            compiled_module_path = compilation_result.main_module_path
        elif best_attempt.verification_result is not None and best_attempt.verification_result.success:
            recompile = compiler.compile(best_attempt.full_dafny_code, output_name)
            if recompile.success and recompile.main_module_path is not None:
                compilation_result = recompile
                compiled_module_path = recompile.main_module_path
                best_attempt.compilation_result = recompile

        output_dir = (
            compilation_result.output_dir
            if compilation_result is not None and compilation_result.success
            else run_python_dir
        )

        search_tree_export = tree.export()
        if self.save_reports:
            if success and best_attempt.eval_result is not None and compilation_result is not None:
                self._save_success_report(
                    best_attempt.strategy_code,
                    best_attempt.full_dafny_code,
                    compilation_result,
                    attempts,
                    task_description,
                    output_name,
                    run_dir,
                    run_dafny_dir,
                    run_results_dir,
                    best_attempt.eval_result,
                    search_tree=search_tree_export,
                    best_node_id=best_node.node_id,
                    best_goodness=best_node.goodness,
                )
            else:
                self._save_best_effort_report(
                    best_attempt=best_attempt,
                    attempts=attempts,
                    task_description=task_description,
                    output_name=output_name,
                    run_dir=run_dir,
                    run_dafny_dir=run_dafny_dir,
                    run_results_dir=run_results_dir,
                    search_tree=search_tree_export,
                    best_node_id=best_node.node_id,
                    best_goodness=best_node.goodness,
                    met_threshold=met_threshold,
                )
                self._write_fallback_winner(attempts, run_results_dir)

        return SynthesisResult(
            success=success,
            strategy_code=best_attempt.strategy_code,
            full_dafny_code=best_attempt.full_dafny_code,
            compiled_module_path=compiled_module_path,
            output_dir=output_dir,
            run_dir=run_dir,
            attempts=attempts,
            total_time_ms=total_time,
            best_node_id=best_node.node_id,
            best_goodness=best_node.goodness,
            met_threshold=met_threshold,
            search_tree=search_tree_export,
        )

    def _save_failure_report(
        self,
        attempts: list[SynthesisAttempt],
        task_description: str,
        output_name: str,
        run_dir: Path,
        results_dir: Path,
        *,
        search_tree: list[dict] | None = None,
        best_node_id: int | None = None,
        best_goodness: float | None = None,
    ) -> Path:
        """Save a detailed failure report to disk."""
        report_path = results_dir / "failure_report.json"

        report = {
            "run_configuration": self._run_configuration_metadata(task_description, output_name),
            "task_description": task_description,
            "total_attempts": len(attempts),
            "timestamp": datetime.now().isoformat(),
            "attempts": [attempt.to_dict() for attempt in attempts],
            "failure_patterns": self._analyze_failure_patterns(attempts),
            "search_tree": search_tree,
            "best_node_id": best_node_id,
            "best_goodness": best_goodness,
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Failure report saved to: {report_path}")

        # Create 'latest' symlink in the generated directory even on failure
        try:
            latest_link = self.output_dir / "latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(run_dir.name, target_is_directory=True)
            print(f"Latest run link (failed) updated: {latest_link}")
        except Exception as e:
            print(f"Warning: Could not create 'latest' symlink: {e}")

        return report_path

    def _save_best_effort_report(
        self,
        *,
        best_attempt: SynthesisAttempt,
        attempts: list[SynthesisAttempt],
        task_description: str,
        output_name: str,
        run_dir: Path,
        run_dafny_dir: Path,
        run_results_dir: Path,
        search_tree: list[dict],
        best_node_id: int,
        best_goodness: float,
        met_threshold: bool,
    ) -> None:
        """Persist the argmax-goodness node when thresholds were not met."""
        dafny_path = run_dafny_dir / f"{output_name}.dfy"
        with open(dafny_path, "w") as f:
            f.write(best_attempt.full_dafny_code)
        canonical_dafny_path = run_dafny_dir / "GeneratedCSD.dfy"
        with open(canonical_dafny_path, "w") as f:
            f.write(best_attempt.full_dafny_code)

        report_path = run_results_dir / "best_effort_report.json"
        report = {
            "run_configuration": self._run_configuration_metadata(
                task_description=task_description,
                output_name=output_name,
            ),
            "strategy_code": best_attempt.strategy_code,
            "dafny_file": str(dafny_path),
            "dafny_file_canonical": str(canonical_dafny_path),
            "total_attempts": len(attempts),
            "timestamp": datetime.now().isoformat(),
            "best_node_id": best_node_id,
            "best_goodness": best_goodness,
            "met_threshold": met_threshold,
            "search_tree": search_tree,
            "evaluation": (
                best_attempt.eval_result.to_dict()
                if best_attempt.eval_result is not None
                else None
            ),
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        self._save_failure_report(
            attempts,
            task_description,
            output_name,
            run_dir,
            run_results_dir,
            search_tree=search_tree,
            best_node_id=best_node_id,
            best_goodness=best_goodness,
        )
        print(f"Best-effort strategy saved to: {dafny_path}")
        print(f"Best-effort report saved to: {report_path}")

    def _save_success_report(
        self,
        strategy_code: str,
        full_code: str,
        compilation_result: CompilationResult,
        attempts: list[SynthesisAttempt],
        task_description: str,
        output_name: str,
        run_dir: Path,
        dafny_dir: Path,
        results_dir: Path,
        evaluation_result: EvaluationResult,
        *,
        search_tree: list[dict] | None = None,
        best_node_id: int | None = None,
        best_goodness: float | None = None,
    ) -> None:
        """Save a success report and the final strategy."""
        # Save the Dafny source
        dafny_path = dafny_dir / f"{output_name}.dfy"
        with open(dafny_path, "w") as f:
            f.write(full_code)
        canonical_dafny_path = dafny_dir / "GeneratedCSD.dfy"
        with open(canonical_dafny_path, "w") as f:
            f.write(full_code)

        # NOTE: We do NOT overwrite synthesis/verify/library/GeneratedCSD.dfy
        # here because it contains
        # the template markers (QWEN_INSERT_STRATEGY_HERE) needed for future runs.
        # The final Dafny code is saved in the run directory instead.

        rationale_extracted = extract_rationale(strategy_code)

        # Save a report
        report_path = results_dir / "success_report.json"
        report = {
            "run_configuration": self._run_configuration_metadata(
                task_description=task_description,
                output_name=output_name,
            ),
            "strategy_code": strategy_code,
            "tool_choice_rationale": rationale_extracted.rationale,
            "dafny_file": str(dafny_path),
            "dafny_file_canonical": str(canonical_dafny_path),
            "compiled_dir": str(compilation_result.output_dir),
            "total_attempts": len(attempts),
            "timestamp": datetime.now().isoformat(),
            "evaluation_result": evaluation_result.to_dict(),
            "sample_outputs": evaluation_result.sample_outputs,
            "search_tree": search_tree,
            "best_node_id": best_node_id,
            "best_goodness": best_goodness,
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Strategy saved to: {dafny_path}")
        print(f"Success report saved to: {report_path}")

        # Create 'latest' symlink in the generated directory
        try:
            latest_link = self.output_dir / "latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(run_dir.name, target_is_directory=True)
            print(f"Latest run link updated: {latest_link}")
        except Exception as e:
            print(f"Warning: Could not create 'latest' symlink: {e}")

    def _analyze_failure_patterns(self, attempts: list[SynthesisAttempt]) -> dict:
        """Analyze common failure patterns across attempts."""
        patterns = {
            "search_contract_failures": 0,
            "verification_failures": 0,
            "compilation_failures": 0,
            "runtime_failures": 0,
            "common_errors": [],
        }

        error_counts: dict[str, int] = {}

        for attempt in attempts:
            if attempt.failed_at == FailureStage.SEARCH_CONTRACT:
                patterns["search_contract_failures"] += 1
            elif attempt.failed_at == FailureStage.VERIFICATION:
                patterns["verification_failures"] += 1
            elif attempt.failed_at == FailureStage.COMPILATION:
                patterns["compilation_failures"] += 1
            elif attempt.failed_at == FailureStage.RUNTIME:
                patterns["runtime_failures"] += 1

            # Extract key error phrases
            if attempt.error_summary:
                if "GuaranteesValidOutput" in attempt.error_summary:
                    error_counts["GuaranteesValidOutput lemma failed"] = error_counts.get(
                        "GuaranteesValidOutput lemma failed", 0
                    ) + 1
                if "Free" in attempt.error_summary:
                    error_counts["Uses Free without fallback"] = error_counts.get(
                        "Uses Free without fallback", 0
                    ) + 1
                if "type" in attempt.error_summary.lower():
                    error_counts["Type error"] = error_counts.get("Type error", 0) + 1

        patterns["common_errors"] = [
            {"error": error, "count": count}
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        ]

        return patterns
