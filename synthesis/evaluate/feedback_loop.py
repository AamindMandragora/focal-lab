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
from ..generate.generator import StrategyGenerator
from ..generate import prompts as generation_prompts
from ..generate.rationale import extract_rationale
from ..verify.verifier import DafnyVerifier, VerificationResult


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
    """Result of a successful synthesis."""

    success: bool
    strategy_code: str
    full_dafny_code: str
    compiled_module_path: Optional[Path]
    output_dir: Optional[Path]
    run_dir: Optional[Path]
    attempts: list[SynthesisAttempt]
    total_time_ms: float

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
        }


class SynthesisPipeline:
    """
    Main pipeline for synthesizing CSD strategies.

    Orchestrates:
    1. Initial strategy generation with Qwen
    2. Dafny verification
    3. Compilation to Python
    4. Runtime testing
    5. Evaluation on dataset sample (optional)
    6. Feedback-based refinement on failure
    """

    DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "generated"
    NON_PRUNABLE_HELPERS = {
        "UnconstrainedStep",
        "ConstrainedStep",
        "AppendConstrainedToken",
        "OpenConstrainedSpan",
        "EnterObservedConstrainedSpan",
        "CloseConstrainedSpan",
        "IsTokenValidNext",
        "ValidTokenCount",
        "DeadEndDetection",
        "TopValidCandidates",
        "RollbackConstrainedSuffix",
        "LastTokenBefore",
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
                summary = attempt.eval_result.get_behavioral_context_summary()
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
        """Scalar score used for helper utility estimates."""
        delimiter_score = 1.0 if (result.contains_delimiters or not self.require_delimiters) else 0.0
        runtime_score = (
            1.0
            if self.eval_max_seconds_per_example is None
            or result.max_sample_time_seconds <= self.eval_max_seconds_per_example
            else 0.0
        )
        return result.accuracy + result.syntax_rate + delimiter_score + runtime_score

    def _collect_prunable_helper_scores(
        self,
        evaluated_attempts: list[SynthesisAttempt],
    ) -> dict[str, list[float]]:
        """Collect scalar rewards for each prunable helper across evaluated attempts."""
        helper_scores: dict[str, list[float]] = {}
        for attempt in evaluated_attempts:
            score = self._evaluation_scalar_score(attempt.eval_result)
            used_helpers = set(self._get_helper_calls_for_evaluation_history(attempt.strategy_code))
            for helper in used_helpers:
                if helper not in self.PRUNABLE_HELPERS:
                    continue
                if helper not in self._helper_universe:
                    continue
                helper_scores.setdefault(helper, []).append(score)
        return helper_scores

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

        helper_scores = self._collect_prunable_helper_scores(evaluated_attempts)
        pulls = {helper: len(helper_scores.get(helper, [])) for helper in prunable_pool}
        means = {
            helper: (
                sum(helper_scores.get(helper, [])) / pulls[helper]
                if pulls[helper] > 0
                else 0.0
            )
            for helper in prunable_pool
        }

        best_attempt = max(
            evaluated_attempts,
            key=lambda attempt: self._evaluation_scalar_score(attempt.eval_result),
        )
        best_helpers = set(self._get_helper_calls_for_evaluation_history(best_attempt.strategy_code))
        keep_prunable = set(best_helpers & set(prunable_pool))

        total_pulls = max(1, sum(pulls.values()))
        untried = [helper for helper in prunable_pool if pulls[helper] == 0 and helper not in keep_prunable]
        explore_count = min(self.helper_bandit_explore_untried, len(untried))
        keep_prunable.update(untried[:explore_count])

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
        status = (
            "helper mask active (bandit/UCB); "
            f"kept {len(keep_prunable)}/{len(prunable_pool)} prunable helpers "
            f"(top_k={self.helper_bandit_top_k}, explore_untried={self.helper_bandit_explore_untried})"
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
        attempts: list["SynthesisAttempt"],
        anchor_n: int | None,
        current_attempt: "SynthesisAttempt",
        current_strategy_code: str,
        current_eval_result,
    ) -> tuple[str, float, float]:
        """Resolve (best_strategy_code, best_accuracy, best_syntax_rate).

        Falls back to the current attempt when no prior evaluated attempt
        wins on (accuracy, syntax_rate). The fallback keeps the restart
        prompt's "score to beat" block well-defined even on the very first
        eval failure.
        """
        if anchor_n is not None:
            match = next(
                (
                    a for a in attempts
                    if a.attempt_number == anchor_n
                    and a is not current_attempt
                    and a.eval_result is not None
                    and (a.eval_result.num_examples or 0) > 0
                ),
                None,
            )
            if match is not None:
                return (
                    match.strategy_code,
                    match.eval_result.accuracy or 0.0,
                    match.eval_result.syntax_rate or 0.0,
                )
        return (
            current_strategy_code,
            (current_eval_result.accuracy if current_eval_result is not None else 0.0) or 0.0,
            (current_eval_result.syntax_rate if current_eval_result is not None else 0.0) or 0.0,
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

    def synthesize(
        self,
        task_description: str,
        output_name: str = "generated_csd",
        initial_strategy_code: str | None = None,
        initial_attempt_offset: int = 0,
    ) -> SynthesisResult:
        """
        Synthesize a CSD strategy for the given task.

        Args:
            task_description: Description of what the strategy should accomplish
            output_name: Name for the output module

        Returns:
            SynthesisResult on success

        Raises:
            SynthesisExhaustionError: If all attempts fail
        """
        import time

        start_time = time.time()
        attempts: list[SynthesisAttempt] = []

        # Create an isolated output directory for this run. The directory layout is:
        #   outputs/generated/<output_name>_<run_id>/
        #     - dafny/
        #     - python/
        #     - results/
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
        run_dir = self.output_dir / f"{output_name}_{run_id}"
        run_dafny_dir = run_dir / "dafny"
        run_python_dir = run_dir / "python"
        run_results_dir = run_dir / "results"
        run_dafny_dir.mkdir(parents=True, exist_ok=True)
        run_python_dir.mkdir(parents=True, exist_ok=True)
        run_results_dir.mkdir(parents=True, exist_ok=True)

        # Persist exact prompt/response records under the repo's single logs tree.
        from synthesis.project_defaults import synthesis_prompt_log_dir

        prompt_log_dir = synthesis_prompt_log_dir(output_name, run_id)
        prompt_log_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CSD_PROMPT_LOG_DIR"] = str(prompt_log_dir)

        # Update a convenience pointer to the most recent run
        try:
            (self.output_dir / "latest_run.txt").write_text(str(run_dir) + "\n")
        except Exception:
            pass

        # Use a per-run compiler output directory.
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

        # Initial generation, or a caller-provided recovery seed.
        if initial_strategy_code is not None:
            print("Using caller-provided initial strategy seed")
            strategy_code = initial_strategy_code
        else:
            print(f"Generating initial strategy for: {task_description}")
            strategy_code = self.generator.generate_initial(
                task_description,
                allowed_helpers=allowed_helpers,
            )

        # Index in `attempts` after which we last performed a fresh restart.
        # Used to bound the "consecutive verification failures since last restart"
        # counter so that a restart resets it.
        last_restart_index = 0

        for iteration in range(self.max_iterations):
            attempt_num = initial_attempt_offset + iteration + 1
            attempt_total = initial_attempt_offset + self.max_iterations
            allowed_helpers, helper_status = self._compute_allowed_helpers(attempts)
            print(f"\n{'='*60}")
            print(f"Attempt {attempt_num}/{attempt_total}")
            print(f"{'='*60}")
            if helper_status:
                print(f"Helper policy: {helper_status}")
            print(f"Strategy: {strategy_code}")

            # Create full Dafny code
            full_code = self.generator.inject_strategy(strategy_code)

            # Create attempt record
            attempt = SynthesisAttempt(
                attempt_number=attempt_num,
                strategy_code=strategy_code,
                full_dafny_code=full_code,
                timestamp=datetime.now().isoformat(),
            )

            disallowed_helpers = self._get_disallowed_helper_calls(strategy_code, allowed_helpers)
            if disallowed_helpers:
                print("  ✗ Strategy contract violation")
                error_msg = (
                    "Strategy contract violation.\n"
                    f"Violations: {', '.join(disallowed_helpers)}"
                )
                attempt.failed_at = FailureStage.SEARCH_CONTRACT
                attempt.error_summary = error_msg
                attempts.append(attempt)

                print("  Refining based on strategy contract violation...")
                next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
                if next_helper_status:
                    print(f"  Helper policy: {next_helper_status}")
                strategy_code = self._refine_with_beam(
                    stage_label="search_contract",
                    previous_strategy=strategy_code,
                    allowed_helpers=next_allowed_helpers,
                    refine_once=lambda: self.generator.refine_after_verification_error(
                        strategy_code,
                        error_msg,
                        allowed_helpers=next_allowed_helpers,
                    ),
                )
                continue

            # Stage 1: Verification
            print("\n[1/4] Verifying with Dafny...")
            verification_result = self.verifier.verify(full_code)
            attempt.verification_result = verification_result

            if not verification_result.success:
                print("  ✗ Verification failed")
                print(f"  Error: {verification_result.get_error_summary()[:300]}")
                attempt.failed_at = FailureStage.VERIFICATION
                attempt.error_summary = verification_result.get_error_summary()
                attempts.append(attempt)

                # Check if we're stuck on the same error repeatedly
                error_msg = verification_result.get_error_summary()
                consecutive_same = 0
                for prev in reversed(attempts[:-1]):
                    if prev.failed_at == FailureStage.VERIFICATION and prev.error_summary == error_msg:
                        consecutive_same += 1
                    else:
                        break

                if consecutive_same >= 2:
                    # After 3+ identical errors, abandon refinement and start fresh
                    print(f"  Stuck on same error for {consecutive_same + 1} attempts — restarting with fresh generation...")
                    next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
                    if next_helper_status:
                        print(f"  Helper policy: {next_helper_status}")
                    strategy_code = self.generator.generate_initial(
                        task_description,
                        allowed_helpers=next_allowed_helpers,
                    )
                    last_restart_index = len(attempts)
                    continue

                # Also restart if the last 3 attempts since the most recent
                # restart all failed verification, even when the specific
                # errors differ. This catches cases where the model is
                # rewriting the strategy every iteration and each broken
                # version surfaces a fresh error.
                post_restart_attempts = attempts[last_restart_index:]
                consecutive_verif_failures = 0
                for prev in reversed(post_restart_attempts):
                    if prev.failed_at == FailureStage.VERIFICATION:
                        consecutive_verif_failures += 1
                    else:
                        break

                if consecutive_verif_failures >= 3:
                    print(
                        f"  {consecutive_verif_failures} consecutive verification failures "
                        f"since last restart — restarting with fresh generation..."
                    )
                    next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
                    if next_helper_status:
                        print(f"  Helper policy: {next_helper_status}")
                    strategy_code = self.generator.generate_initial(
                        task_description,
                        allowed_helpers=next_allowed_helpers,
                    )
                    last_restart_index = len(attempts)
                    continue

                # Refine based on verification error
                print("  Refining based on verification error...")
                structured_feedback = verification_result.get_structured_feedback()
                error_history = self._get_verification_history_summary(attempts)
                behavioral_context = self._get_recent_behavioral_context(attempts)
                next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
                if next_helper_status:
                    print(f"  Helper policy: {next_helper_status}")
                strategy_code = self._refine_with_beam(
                    stage_label="verification",
                    previous_strategy=strategy_code,
                    allowed_helpers=next_allowed_helpers,
                    refine_once=lambda: self.generator.refine_after_verification_error(
                        strategy_code,
                        error_msg,
                        behavioral_context=behavioral_context,
                        structured_feedback=structured_feedback,
                        error_history=error_history,
                        allowed_helpers=next_allowed_helpers,
                    ),
                )
                continue

            print("  ✓ Verification passed")

            # Stage 2: Compilation
            print("\n[2/4] Compiling to Python...")
            compilation_result = compiler.compile(full_code, output_name)
            attempt.compilation_result = compilation_result

            if not compilation_result.success:
                print("  ✗ Compilation failed")
                attempt.failed_at = FailureStage.COMPILATION
                attempt.error_summary = compilation_result.get_error_summary()
                attempts.append(attempt)

                # Refine based on compilation error
                print("  Refining based on compilation error...")
                next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
                if next_helper_status:
                    print(f"  Helper policy: {next_helper_status}")
                strategy_code = self._refine_with_beam(
                    stage_label="compilation",
                    previous_strategy=strategy_code,
                    allowed_helpers=next_allowed_helpers,
                    refine_once=lambda: self.generator.refine_after_compilation_error(
                        strategy_code,
                        compilation_result.get_error_summary(),
                        allowed_helpers=next_allowed_helpers,
                    ),
                )
                continue

            print(f"  ✓ Compiled to {compilation_result.output_dir}")

            if compilation_result.main_module_path is None:
                print("  ✗ No main module found")
                attempt.failed_at = FailureStage.RUNTIME
                attempt.error_summary = "No main module path in compilation result"
                attempts.append(attempt)

                next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
                if next_helper_status:
                    print(f"  Helper policy: {next_helper_status}")
                strategy_code = self._refine_with_beam(
                    stage_label="runtime",
                    previous_strategy=strategy_code,
                    allowed_helpers=next_allowed_helpers,
                    refine_once=lambda: self.generator.refine_after_runtime_error(
                        strategy_code,
                        "Compilation succeeded but no Python module was generated",
                        allowed_helpers=next_allowed_helpers,
                    ),
                )
                continue

            print("\n[3/4] Evaluating compiled strategy (runtime smoke test removed).")

            # Stage 4: Evaluation
            print("\n[4/4] Evaluating on dataset sample...")
            # Unload generator model to free GPU memory for eval model
            if self.generator._model is not None:
                del self.generator._model
                self.generator._model = None
                import gc
                gc.collect()
                import torch
                torch.cuda.empty_cache()
                print("  Generator model (HF) unloaded to free GPU memory")
            # Also unload vllm engine if present (vllm backend keeps workers in subprocesses)
            if getattr(self.generator, '_vllm', None) is not None:
                import gc
                import torch
                vllm_obj = self.generator._vllm
                self.generator._vllm = None
                try:
                    vllm_obj._run_engine = None  # sever reference to engine
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

            # Eval seed is fixed across iterations so per-iter deltas are
            # statistically trustworthy and opus can anchor on best-so-far.
            # Overfitting concern is handled by the final held-out eval.
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
            # Change 2: stamp the cross-attempt cluster ledger on the result
            # so EvaluationResult.get_feedback_summary can emit persistent
            # mode IDs (mode_A appeared in attempts 1,3,5…).
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
                attempts.append(attempt)

                self._unload_evaluator_runtime_before_refinement()
                print("  Refining based on evaluation error...")
                evaluation_feedback = eval_result.get_feedback_summary()
                mode_examples = eval_result._render_mode_examples()
                next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
                if next_helper_status:
                    print(f"  Helper policy: {next_helper_status}")
                anchor_n, anchor_acc, anchor_syn = self._compute_pareto_best(attempts)
                self._update_anchor_state(anchor_n)
                use_restart = self._should_restart(attempts)
                if use_restart:
                    print(
                        f"  [synthesis] RESTART mode active for next refinement "
                        f"({self._iters_since_anchor_changed} iters since anchor moved)"
                    )
                elif anchor_n is not None:
                    print(
                        f"  [synthesis] anchor for next refinement: attempt {anchor_n} "
                        f"(acc={anchor_acc:.1%}, syn={anchor_syn:.1%})"
                    )
                best_strategy_code, best_acc_val, best_syn_val = (
                    self._lookup_best_so_far(attempts, anchor_n, attempt, strategy_code, eval_result)
                )
                attempt_outcome_ledger = self._build_attempt_outcome_ledger(attempts, anchor_n)
                prev_acc = eval_result.accuracy or 0.0
                prev_syn = eval_result.syntax_rate or 0.0
                prev_n = eval_result.num_examples or 0
                if use_restart:
                    strategy_code = self._refine_with_beam(
                        stage_label="evaluation_error_restart",
                        previous_strategy=strategy_code,
                        allowed_helpers=next_allowed_helpers,
                        refine_once=lambda: self.generator.generate_initial(
                            task_description,
                            allowed_helpers=next_allowed_helpers,
                        ),
                    )
                    self._apply_restart_cooldown()
                else:
                    refine_best_strategy = (
                        best_strategy_code if best_strategy_code != strategy_code else None
                    )
                    refine_best_acc = (
                        best_acc_val if refine_best_strategy is not None else None
                    )
                    refine_best_syn = (
                        best_syn_val if refine_best_strategy is not None else None
                    )
                    strategy_code = self._refine_with_beam(
                        stage_label="evaluation_error",
                        previous_strategy=strategy_code,
                        allowed_helpers=next_allowed_helpers,
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
                            allowed_helpers=next_allowed_helpers,
                            eval_max_seconds_per_example=self.eval_max_seconds_per_example,
                            mode_examples=mode_examples,
                            attempt_outcome_ledger=attempt_outcome_ledger,
                        ),
                    )
                continue

            # Check if evaluation meets thresholds
            if not eval_result.meets_threshold(
                min_accuracy=self.min_accuracy,
                min_syntax_rate=self.min_syntax_rate,
                require_delimiters=self.require_delimiters,
                max_seconds_per_example=self.eval_max_seconds_per_example,
            ):
                print(f"  ✗ Evaluation below threshold:")
                print(f"    Accuracy: {eval_result.accuracy:.1%} (min: {self.min_accuracy:.1%})")
                print(
                    "    Contains << >>: "
                    f"{'yes' if eval_result.contains_delimiters else 'no'} "
                    f"(required: {'yes' if self.require_delimiters else 'no'})"
                )
                print(f"    Syntax: {eval_result.syntax_rate:.1%} (min: {self.min_syntax_rate:.1%})")
                if self.eval_max_seconds_per_example is not None:
                    print(
                        f"    Slowest Example Time: {eval_result.max_sample_time_seconds:.2f}s "
                        f"(max: {self.eval_max_seconds_per_example:.2f}s)"
                    )
                attempt.failed_at = FailureStage.EVALUATION
                attempt.error_summary = eval_result.get_feedback_summary()
                attempts.append(attempt)

                self._unload_evaluator_runtime_before_refinement()
                print("  Refining based on evaluation results...")
                threshold_feedback = (
                    "Required thresholds:\n"
                    f"  Accuracy: {self.min_accuracy:.1%}\n"
                    f"  Syntax Rate: {self.min_syntax_rate:.1%}\n\n"
                    f"  Contains << >>: {'required' if self.require_delimiters else 'optional'}\n"
                    + (
                        f"  Max Runtime / Example: {self.eval_max_seconds_per_example:.2f}s\n"
                        if self.eval_max_seconds_per_example is not None
                        else ""
                    )
                    + "\n"
                    + eval_result.get_feedback_summary()
                )
                threshold_mode_examples = eval_result._render_mode_examples()
                next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
                if next_helper_status:
                    print(f"  Helper policy: {next_helper_status}")
                anchor_n, anchor_acc, anchor_syn = self._compute_pareto_best(attempts)
                self._update_anchor_state(anchor_n)
                use_restart = self._should_restart(attempts)
                if use_restart:
                    print(
                        f"  [synthesis] RESTART mode active for next refinement "
                        f"({self._iters_since_anchor_changed} iters since anchor moved)"
                    )
                elif anchor_n is not None:
                    print(
                        f"  [synthesis] anchor for next refinement: attempt {anchor_n} "
                        f"(acc={anchor_acc:.1%}, syn={anchor_syn:.1%})"
                    )
                best_strategy_code, best_acc_val, best_syn_val = (
                    self._lookup_best_so_far(attempts, anchor_n, attempt, strategy_code, eval_result)
                )
                attempt_outcome_ledger = self._build_attempt_outcome_ledger(attempts, anchor_n)
                prev_acc = eval_result.accuracy or 0.0
                prev_syn = eval_result.syntax_rate or 0.0
                prev_n = eval_result.num_examples or 0
                if use_restart:
                    strategy_code = self._refine_with_beam(
                        stage_label="evaluation_threshold_restart",
                        previous_strategy=strategy_code,
                        allowed_helpers=next_allowed_helpers,
                        refine_once=lambda: self.generator.generate_initial(
                            task_description,
                            allowed_helpers=next_allowed_helpers,
                        ),
                    )
                    self._apply_restart_cooldown()
                else:
                    refine_best_strategy = (
                        best_strategy_code if best_strategy_code != strategy_code else None
                    )
                    refine_best_acc = (
                        best_acc_val if refine_best_strategy is not None else None
                    )
                    refine_best_syn = (
                        best_syn_val if refine_best_strategy is not None else None
                    )
                    strategy_code = self._refine_with_beam(
                        stage_label="evaluation_threshold",
                        previous_strategy=strategy_code,
                        allowed_helpers=next_allowed_helpers,
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
                            allowed_helpers=next_allowed_helpers,
                            eval_max_seconds_per_example=self.eval_max_seconds_per_example,
                            mode_examples=threshold_mode_examples,
                            attempt_outcome_ledger=attempt_outcome_ledger,
                        ),
                    )
                continue

            print(f"  ✓ Evaluation passed:")
            print(f"    Accuracy: {eval_result.accuracy:.1%}")
            print(f"    Contains << >>: {'yes' if eval_result.contains_delimiters else 'no'}")
            print(f"    Syntax: {eval_result.syntax_rate:.1%}")

            # Success!
            attempts.append(attempt)
            total_time = (time.time() - start_time) * 1000

            print(f"\n{'='*60}")
            print(f"SUCCESS after {attempt_num} attempt(s)")
            print(f"Total time: {total_time:.1f}ms")
            print(f"{'='*60}")

            # Save successful strategy
            self._save_success_report(
                strategy_code,
                full_code,
                compilation_result,
                attempts,
                task_description,
                output_name,
                run_dir,
                run_dafny_dir,
                run_results_dir,
                eval_result,
            )

            return SynthesisResult(
                success=True,
                strategy_code=strategy_code,
                full_dafny_code=full_code,
                compiled_module_path=compilation_result.main_module_path,
                output_dir=compilation_result.output_dir,
                run_dir=run_dir,
                attempts=attempts,
                total_time_ms=total_time,
            )

        # All attempts exhausted
        total_time = (time.time() - start_time) * 1000

        print(f"\n{'='*60}")
        print(f"FAILED after {self.max_iterations} attempts")
        print(f"Total time: {total_time:.1f}ms")
        print(f"{'='*60}")

        # Save failure report
        report_path = None
        if self.save_reports:
            report_path = self._save_failure_report(
                attempts,
                task_description,
                output_name,
                run_dir,
                run_results_dir,
            )

        # Best-accuracy fallback: if any attempt's accuracy beat min_accuracy
        # (even if syntax fell short of min_syntax_rate), save a side-channel
        # `fallback_winner.json` so this cell can be harvested as an accuracy-only
        # win in post-hoc analysis. Does NOT change exception semantics — the
        # subprocess still exits non-zero so existing flows aren't affected.
        fallback_winner = None
        for att in attempts:
            if (
                att.eval_result is not None
                and att.eval_result.accuracy >= self.min_accuracy
                and (
                    fallback_winner is None
                    or att.eval_result.accuracy > fallback_winner.eval_result.accuracy
                )
            ):
                fallback_winner = att

        if fallback_winner is not None:
            fb_path = run_results_dir / "fallback_winner.json"
            try:
                with open(fb_path, "w") as f:
                    json.dump(
                        {
                            "fallback_reason": "accuracy_met_but_syntax_below_threshold",
                            "min_accuracy": self.min_accuracy,
                            "min_syntax_rate": self.min_syntax_rate,
                            "winner_attempt_number": fallback_winner.attempt_number,
                            "winner_accuracy": fallback_winner.eval_result.accuracy,
                            "winner_syntax_rate": fallback_winner.eval_result.syntax_rate,
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
        else:
            print(
                "[FALLBACK] No accuracy-only candidate found either "
                "(no attempt met the min_accuracy threshold); no fallback saved."
            )

        error = SynthesisExhaustionError(
            f"Synthesis failed after {self.max_iterations} attempts", attempts, report_path
        )

        print(error.get_failure_summary())
        raise error

    def _save_failure_report(
        self,
        attempts: list[SynthesisAttempt],
        task_description: str,
        output_name: str,
        run_dir: Path,
        results_dir: Path,
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
