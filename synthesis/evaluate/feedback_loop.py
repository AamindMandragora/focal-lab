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
        eval_max_seconds_per_example: Optional[float] = None,
        adaptive_helper_mask: bool = True,
        helper_selection_policy: str = "utility",
        helper_mask_min_evals: int = 4,
        helper_mask_min_uses: int = 2,
        helper_mask_margin: float = 0.25,
        helper_mask_max_disabled: int = 6,
        helper_bandit_min_evals: int = 3,
        helper_bandit_top_k: int = 6,
        helper_bandit_ucb_c: float = 0.35,
        helper_bandit_explore_untried: int = 1,
        refinement_beam_size: int = 1,
        local_neighborhood_refinement: bool = True,
        max_local_edit_ratio: float = 0.65,
        beam_verify_candidates: bool = True,
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
            adaptive_helper_mask: Enable empirical helper pruning contract
            helper_selection_policy: Helper selection policy (`utility` or `bandit`)
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

        # Evaluation thresholds
        self.min_accuracy = min_accuracy
        self.min_syntax_rate = min_syntax_rate
        self.require_delimiters = require_delimiters
        self.eval_sample_size = eval_sample_size
        self.eval_max_seconds_per_example = eval_max_seconds_per_example
        self.adaptive_helper_mask = adaptive_helper_mask
        normalized_policy = helper_selection_policy.strip().lower()
        if normalized_policy not in {"utility", "bandit"}:
            raise ValueError(
                "helper_selection_policy must be 'utility' or 'bandit'"
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
        """Empirical threshold-based helper pruning (existing policy)."""
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
        baseline_scores = [
            self._evaluation_scalar_score(attempt.eval_result)
            for attempt in evaluated_attempts
        ]
        baseline_mean = sum(baseline_scores) / max(1, len(baseline_scores))

        helper_scores = self._collect_prunable_helper_scores(evaluated_attempts)
        low_utility: list[tuple[float, int, str]] = []
        for helper, scores in helper_scores.items():
            if helper in self.NON_PRUNABLE_HELPERS or helper in best_helpers:
                continue
            if len(scores) < self.helper_mask_min_uses:
                continue
            mean_score = sum(scores) / len(scores)
            if mean_score <= baseline_mean - self.helper_mask_margin:
                low_utility.append((mean_score, len(scores), helper))

        low_utility.sort(key=lambda item: (item[0], -item[1], item[2]))
        disabled: list[str] = []
        for _mean_score, _uses, helper in low_utility:
            if len(disabled) >= self.helper_mask_max_disabled:
                break
            disabled.append(helper)

        allowed_helpers -= set(disabled)
        if disabled:
            status = (
                "helper mask active (utility); disabled low-utility helpers: "
                + ", ".join(disabled)
            )
        else:
            status = "helper mask active (utility); no helpers disabled by utility yet"
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

    def _compute_allowed_helpers(self, attempts: list[SynthesisAttempt]) -> tuple[list[str] | None, str]:
        """
        Build a per-attempt helper-call contract from empirical policy.

        Returns:
            (allowed_helpers, status_text). `allowed_helpers=None` disables the
            contract block in prompts.
        """
        if not self.adaptive_helper_mask or not self._helper_universe:
            return None, ""

        evaluated = [attempt for attempt in attempts if attempt.eval_result is not None]
        if self.helper_selection_policy == "bandit":
            return self._compute_allowed_helpers_bandit(evaluated)
        return self._compute_allowed_helpers_utility(evaluated)

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

    @staticmethod
    def _truncate_words(text: str, max_words: int) -> str:
        """Return a compact word-bounded single-line summary."""
        words = re.sub(r"\s+", " ", text).strip().split()
        if len(words) <= max_words:
            return " ".join(words)
        return " ".join(words[:max_words]) + " ..."

    def _get_strategy_behavior_summary(self, strategy_code: str) -> str:
        """Return a compact summary of what the strategy is trying to do."""
        extracted = extract_rationale(strategy_code)
        body = self._get_strategy_body_for_evaluation_history(strategy_code)
        helpers = self._get_helper_calls_for_evaluation_history(strategy_code)

        lines: list[str] = []
        if extracted.rationale:
            lines.append("rationale: " + self._truncate_words(extracted.rationale, 55))

        open_parts: list[str] = []
        if "EnterObservedConstrainedSpan" in helpers:
            open_parts.append("enters spans when observed delimiters appear in free output")
        if "OpenConstrainedSpan" in helpers:
            open_parts.append("can explicitly open constrained spans")
        if "LastTokenBefore" in helpers:
            open_parts.append("uses recent/generated token context before deciding span behavior")
        if "RollbackConstrainedSuffix" in helpers:
            open_parts.append("can roll back active constrained suffixes")
        if not open_parts and '<<"' in body:
            open_parts.append("checks or emits open delimiters directly")

        inside_parts: list[str] = []
        if "ConfidenceGatedStep" in helpers:
            inside_parts.append("uses confidence-gated constrained stepping")
        if "ConstrainedSymbolInGenerated" in helpers:
            inside_parts.append("generates constrained symbols while updating full output")
        if "ConstrainedSymbol" in helpers:
            inside_parts.append("generates constrained symbols")
        if "AdaptiveConstrainedStep" in helpers or "GroupBoostedConstrainedStep" in helpers:
            inside_parts.append("uses adaptive/group-biased constrained token stepping")
        if "ConstrainedStep" in helpers:
            inside_parts.append("uses hard parser-valid token stepping")
        safe_helpers = [name for name in helpers if name.startswith("Safe")]
        if safe_helpers:
            inside_parts.append("uses safe logit-shaped constrained stepping via " + ", ".join(safe_helpers))

        outside_parts: list[str] = []
        if "UnconstrainedChunk" in helpers:
            outside_parts.append("generates free text in chunks outside constrained spans")
        if "UnconstrainedStep" in helpers:
            outside_parts.append("generates free text token-by-token outside constrained spans")

        close_parts: list[str] = []
        if "CloseConstrainedSpan" in helpers:
            close_parts.append("closes spans through CloseConstrainedSpan")
        if "parser.IsCompletePrefix" in body:
            close_parts.append("uses parser completeness as a close/progress guard")

        mechanical = []
        if outside_parts:
            mechanical.append("outside: " + "; ".join(outside_parts))
        if open_parts:
            mechanical.append("span entry: " + "; ".join(open_parts))
        if inside_parts:
            mechanical.append("inside spans: " + "; ".join(inside_parts))
        if close_parts:
            mechanical.append("closing: " + "; ".join(close_parts))
        if mechanical:
            lines.append("mechanical sketch: " + " | ".join(mechanical))

        return "\n  ".join(lines) if lines else "(no compact behavior summary available)"

    def _get_strategy_profile_for_evaluation_history(self, strategy_code: str) -> tuple[str, tuple]:
        """Return a factual helper/control profile for repeated-shape detection."""
        body = self._get_strategy_body_for_evaluation_history(strategy_code)
        helpers = self._get_helper_calls_for_evaluation_history(strategy_code)
        counts = {
            name: len(re.findall(rf"\bhelpers\.{re.escape(name)}\s*\(", body))
            for name in helpers
        }
        control_facts = {
            "uses_parser_completion_guard": "parser.IsCompletePrefix" in body,
            "uses_parser_validity_guard": "parser.IsValidPrefix" in body,
            "tracks_inside_state": "insideConstrainedOut" in body,
            "tracks_current_constrained": "currentConstrainedOut" in body,
            "uses_generated_suffix_slice": "[|generated|" in body or "generated[|" in body,
            "mentions_open_delimiter_literal": '"<<"' in body,
            "mentions_close_delimiter_literal": '">>"' in body,
        }
        constrained_helpers = [
            name
            for name in helpers
            if "Constrained" in name or name in {"ConfidenceGatedStep"}
        ]
        logit_helpers = [
            name
            for name in helpers
            if name.startswith("Safe")
            or "Boost" in name
            or "Penal" in name
            or "Temperature" in name
            or "Repetition" in name
        ]
        opening_helpers = [
            name
            for name in helpers
            if name in {"OpenConstrainedSpan", "EnterObservedConstrainedSpan", "RollbackConstrainedSuffix"}
        ]

        description_parts = [
            "helpers: " + (", ".join(helpers) if helpers else "(none)"),
        ]
        if constrained_helpers:
            description_parts.append("constrained helpers: " + ", ".join(constrained_helpers))
        if logit_helpers:
            description_parts.append("logit helpers: " + ", ".join(logit_helpers))
        if opening_helpers:
            description_parts.append("span-state helpers: " + ", ".join(opening_helpers))
        description_parts.append(
            "control facts: "
            + ", ".join(
                key
                for key, value in control_facts.items()
                if value
            )
        )

        role_facts = {
            "uses_confidence_gated": "ConfidenceGatedStep" in helpers,
            "uses_safe_logit_step": any(name.startswith("Safe") for name in helpers),
            "uses_group_or_adaptive_step": any(
                name in helpers
                for name in {"AdaptiveConstrainedStep", "GroupBoostedConstrainedStep"}
            ),
            "uses_symbol_generated_helper": "ConstrainedSymbolInGenerated" in helpers,
            "uses_symbol_helper": "ConstrainedSymbol" in helpers,
            "uses_token_constrained_step": any(
                name in helpers
                for name in {"ConstrainedStep", "SafeSoftConstrainedStep"}
            ),
            "uses_observed_span_entry": "EnterObservedConstrainedSpan" in helpers,
            "uses_explicit_open_span": "OpenConstrainedSpan" in helpers,
            "uses_rollback": "RollbackConstrainedSuffix" in helpers,
            "uses_unconstrained_step": "UnconstrainedStep" in helpers,
            "uses_unconstrained_chunk": "UnconstrainedChunk" in helpers,
        }
        description_parts.append(
            "role facts: "
            + ", ".join(
                key
                for key, value in role_facts.items()
                if value
            )
        )

        signature = (
            tuple(sorted(key for key, value in role_facts.items() if value)),
            tuple(
                sorted(
                    (role, sum(counts.get(name, 0) for name in names))
                    for role, names in {
                        "open_or_enter": {
                            "OpenConstrainedSpan",
                            "EnterObservedConstrainedSpan",
                            "RollbackConstrainedSuffix",
                        },
                        "hard_or_soft_constrained_step": {
                            "ConstrainedStep",
                            "SafeSoftConstrainedStep",
                            "AdaptiveConstrainedStep",
                            "GroupBoostedConstrainedStep",
                        },
                        "symbol_step": {
                            "ConstrainedSymbol",
                            "ConstrainedSymbolInGenerated",
                        },
                        "unconstrained": {
                            "UnconstrainedStep",
                            "UnconstrainedChunk",
                        },
                    }.items()
                    if sum(counts.get(name, 0) for name in names) > 1
                )
            ),
            tuple(sorted(key for key, value in control_facts.items() if value)),
        )
        return "; ".join(description_parts), signature

    def _get_outer_structure_signature(self, strategy_code: str) -> tuple[str, ...]:
        """Return a broad CSD-family signature that ignores local helper swaps."""
        body = self._get_strategy_body_for_evaluation_history(strategy_code)
        helpers = set(self._get_helper_calls_for_evaluation_history(strategy_code))

        uses_chunk = "UnconstrainedChunk" in helpers
        uses_step = "UnconstrainedStep" in helpers
        if uses_chunk and uses_step:
            outer_generation = "mixed"
        elif uses_chunk:
            outer_generation = "chunked"
        elif uses_step:
            outer_generation = "token"
        else:
            outer_generation = "none"

        observed_entry = "EnterObservedConstrainedSpan" in helpers
        explicit_entry = "OpenConstrainedSpan" in helpers
        if observed_entry and explicit_entry:
            span_entry = "observed+explicit"
        elif observed_entry:
            span_entry = "observed"
        elif explicit_entry:
            span_entry = "explicit"
        else:
            span_entry = "none"

        candidate_helpers = {"TopValidCandidates", "ValidNextTokens", "ValidNextTokenCandidates"}
        token_helpers = {
            "ConstrainedStep",
            "SafeSoftConstrainedStep",
            "AdaptiveConstrainedStep",
            "GroupBoostedConstrainedStep",
            "ConfidenceGatedStep",
        }
        safe_helpers = {name for name in helpers if name.startswith("Safe")}
        has_candidate_list = bool(helpers & candidate_helpers) or "TopValidCandidates" in body
        has_symbol = bool(helpers & {"ConstrainedSymbol", "ConstrainedSymbolInGenerated"})
        has_token_step = bool(helpers & token_helpers) or bool(safe_helpers)
        has_group_or_adaptive = bool(helpers & {"AdaptiveConstrainedStep", "GroupBoostedConstrainedStep"})
        has_confidence = "ConfidenceGatedStep" in helpers

        if has_candidate_list:
            inside_primary = "candidate_list"
        elif has_symbol:
            inside_primary = "symbol_chunk"
        elif safe_helpers:
            inside_primary = "soft_logit"
        elif has_token_step:
            inside_primary = "token_step"
        else:
            inside_primary = "free/none"

        if has_symbol and has_token_step:
            # Collapse local swaps such as GroupBoostedConstrainedStep vs ConstrainedStep
            # when they are secondary to the same symbol-level core.
            inside_secondary = "narrow_token_step"
        elif has_confidence:
            inside_secondary = "confidence_gated"
        elif safe_helpers:
            inside_secondary = "safe_logit"
        elif has_group_or_adaptive:
            inside_secondary = "group_or_adaptive"
        else:
            inside_secondary = "none"

        if "RollbackConstrainedSuffix" in helpers:
            repair_policy = "rollback"
        elif "[|generated|" in body or "generated[|" in body:
            repair_policy = "suffix_slice"
        else:
            repair_policy = "none"

        if "CloseConstrainedSpan" in helpers and "parser.IsCompletePrefix" in body:
            closing_policy = "parser_complete_close"
        elif "CloseConstrainedSpan" in helpers and ("maxSteps - steps" in body or "remaining" in body):
            closing_policy = "budget_final_close"
        elif '">>"' in body:
            closing_policy = "free_close"
        else:
            closing_policy = "absent"

        return (
            outer_generation,
            span_entry,
            inside_primary,
            inside_secondary,
            closing_policy,
            repair_policy,
        )

    def _describe_outer_structure_signature(self, signature: tuple[str, ...]) -> str:
        """Convert a broad CSD-family signature into compact prompt text."""
        (
            outer_generation,
            span_entry,
            inside_primary,
            inside_secondary,
            closing_policy,
            repair_policy,
        ) = signature

        parts = [
            f"{outer_generation} outside",
            f"{span_entry} entry",
            f"{inside_primary} inside core",
        ]
        if inside_secondary != "none":
            parts.append(f"{inside_secondary} secondary control")
        if closing_policy != "absent":
            parts.append(closing_policy.replace("_", " "))
        if repair_policy != "none":
            parts.append(f"{repair_policy} repair")
        return " + ".join(parts)

    def _get_repeated_outer_structure_summary(self, attempts: list[SynthesisAttempt]) -> list[str]:
        """Return compact repeated broad-family lines for search memory."""
        evaluated_attempts = [attempt for attempt in attempts if attempt.eval_result is not None]
        if len(evaluated_attempts) < 3:
            return []

        best_attempt = max(
            evaluated_attempts,
            key=lambda attempt: self._evaluation_progress_score(attempt.eval_result),
        )
        best_score = self._evaluation_progress_score(best_attempt.eval_result)
        groups: dict[tuple[str, ...], dict] = {}

        for attempt in evaluated_attempts:
            result = attempt.eval_result
            if result is None:
                continue
            signature = self._get_outer_structure_signature(attempt.strategy_code)
            group = groups.setdefault(
                signature,
                {
                    "attempts": [],
                    "best_attempt": None,
                    "best_result": None,
                    "best_score": None,
                    "last_attempt": None,
                },
            )
            score = self._evaluation_progress_score(result)
            group["attempts"].append(attempt.attempt_number)
            group["last_attempt"] = attempt.attempt_number
            if group["best_score"] is None or score > group["best_score"]:
                group["best_score"] = score
                group["best_attempt"] = attempt.attempt_number
                group["best_result"] = result

        repeated = []
        for signature, group in groups.items():
            numbers = group["attempts"]
            after_best = [number for number in numbers if number > best_attempt.attempt_number]
            if len(numbers) >= 3 or len(after_best) >= 2:
                repeated.append((signature, group, len(after_best)))

        repeated.sort(
            key=lambda item: (
                item[1]["best_score"] is not None and item[1]["best_score"] >= best_score,
                len(item[1]["attempts"]),
                item[1]["last_attempt"] or 0,
            ),
            reverse=True,
        )

        lines: list[str] = []
        for signature, group, after_best_count in repeated[:2]:
            best_result = group["best_result"]
            if best_result is None:
                continue
            numbers = ", ".join(str(number) for number in group["attempts"][-6:])
            outcome = (
                f"{best_result.accuracy:.1%} accuracy / {best_result.syntax_rate:.1%} syntax / "
                f"delimiter {'present' if best_result.contains_delimiters else 'absent'}"
            )
            suffix = (
                f"; {after_best_count} attempt(s) after balanced-best have not improved it"
                if after_best_count
                else ""
            )
            lines.append(
                "Repeated broad family: "
                f"{self._describe_outer_structure_signature(signature)} in attempts {numbers}; "
                f"best in family {outcome}{suffix}."
            )
        return lines

    def _get_useful_ingredients_summary(
        self,
        attempts: list[SynthesisAttempt],
        best_attempt: SynthesisAttempt,
    ) -> str | None:
        """Return one positive evidence line about features worth carrying forward."""
        best_result = best_attempt.eval_result
        if best_result is None:
            return None
        has_valid_balanced_best = not self.require_delimiters or best_result.contains_delimiters

        useful_parts = self._get_strategy_ingredient_parts(best_attempt)
        if not useful_parts:
            return None

        best_score = self._evaluation_progress_score(best_result)
        best_accuracy = best_result.accuracy
        best_syntax = best_result.syntax_rate
        regressed_patterns: list[str] = []

        for attempt in attempts:
            result = attempt.eval_result
            if result is None or attempt.attempt_number <= best_attempt.attempt_number:
                continue
            score = self._evaluation_progress_score(result)
            if score >= best_score:
                continue

            signature = self._get_outer_structure_signature(attempt.strategy_code)
            helpers = set(self._get_helper_calls_for_evaluation_history(attempt.strategy_code))
            worse_accuracy = result.accuracy + 0.04 < best_accuracy
            worse_syntax = result.syntax_rate + 0.08 < best_syntax
            worse_contract = self.require_delimiters and not result.contains_delimiters
            worse_runtime = (
                self.eval_max_seconds_per_example is not None
                and result.max_sample_time_seconds >= self.eval_max_seconds_per_example
            )
            if not (worse_accuracy or worse_syntax or worse_contract or worse_runtime):
                continue

            if signature[0] == "chunked" or (
                "UnconstrainedChunk" in helpers and "EnterObservedConstrainedSpan" in helpers
            ):
                regressed_patterns.append("broad observed/chunked takeover")
            if signature[1] == "observed":
                regressed_patterns.append("observed-only/no-explicit entry")
            if signature[1] == "explicit" and "EnterObservedConstrainedSpan" not in helpers:
                regressed_patterns.append("always-on explicit-only entry")
            if signature[2] == "token_step" or "ConfidenceGatedStep" in helpers:
                regressed_patterns.append("token-only or confidence-gated inside control")

        useful_text = ", ".join(useful_parts[:7])
        unique_regressions = list(dict.fromkeys(regressed_patterns))
        if not has_valid_balanced_best:
            return (
                "- Useful ingredients so far: no valid balanced-best yet; the strongest "
                f"attempt suggests {useful_text}, but restore delimiter/coverage before "
                "treating these as proven ingredients. A useful ingredient is a mechanism "
                "with evidence of contributing to a positive metric shift, not a whole "
                "strategy template."
            )
        if unique_regressions:
            return (
                "- Useful ingredients so far: best valid attempts preserve "
                f"{useful_text}; "
                f"{', '.join(unique_regressions[:4])} have repeatedly regressed accuracy, "
                "coverage, or runtime. Carry forward useful ingredients even in larger "
                "changes unless metric evidence shows that specific ingredient stopped helping."
            )
        return (
            "- Useful ingredients so far: best valid attempts preserve "
            f"{useful_text}; carry these forward as evidence-backed ingredients while "
            "changing the next measured failure source. Preserve the ingredients that "
            "contributed to positive metric movement; do not copy the whole strategy shape."
        )

    def _get_strategy_ingredient_parts(self, attempt: SynthesisAttempt) -> list[str]:
        """Return compact mechanism parts that can be used as evidence."""
        (
            outer_generation,
            span_entry,
            inside_primary,
            inside_secondary,
            closing_policy,
            repair_policy,
        ) = self._get_outer_structure_signature(attempt.strategy_code)

        useful_parts: list[str] = [
            "broad family " + self._describe_outer_structure_signature(
                (
                    outer_generation,
                    span_entry,
                    inside_primary,
                    inside_secondary,
                    closing_policy,
                    repair_policy,
                )
            )
        ]
        if outer_generation == "token":
            useful_parts.append("mostly free token-wise outside generation")
        elif outer_generation == "mixed":
            useful_parts.append("bounded mixed free outside generation")
        elif outer_generation == "chunked":
            useful_parts.append("bounded free chunks outside constrained regions")

        if span_entry == "observed+explicit":
            useful_parts.append("observed entry with selective explicit entry")
        elif span_entry == "explicit":
            useful_parts.append("selective explicit span entry")
        elif span_entry == "observed":
            useful_parts.append("observed span entry")

        if inside_primary == "symbol_chunk":
            useful_parts.append("symbol-level progress in wider constrained states")
        elif inside_primary == "token_step":
            useful_parts.append("hard token progression inside constrained states")
        elif inside_primary == "soft_logit":
            useful_parts.append("logit-shaped constrained progression")

        if inside_secondary == "narrow_token_step":
            useful_parts.append("hard token fallback in narrow states")
        elif inside_secondary == "confidence_gated":
            useful_parts.append("confidence-gated fallback in narrow states")
        elif inside_secondary == "safe_logit":
            useful_parts.append("safe logit fallback in narrow states")

        if closing_policy == "parser_complete_close":
            useful_parts.append("immediate close on parser-complete prefixes")
        elif closing_policy == "budget_final_close":
            useful_parts.append("budget-aware final close")

        if repair_policy == "rollback":
            useful_parts.append("rollback repair")

        return useful_parts

    def _get_near_win_refinement_summary(
        self,
        best_attempt: SynthesisAttempt,
    ) -> str | None:
        """Return a mode-selection line when balanced-best should be repaired surgically."""
        best_result = best_attempt.eval_result
        if best_result is None:
            return None

        accuracy_target = self.min_accuracy if self.min_accuracy > 0 else 1.0
        syntax_target = self.min_syntax_rate if self.min_syntax_rate > 0 else 1.0
        accuracy_progress = min(best_result.accuracy / accuracy_target, 1.0)
        syntax_progress = min(best_result.syntax_rate / syntax_target, 1.0)
        covered, total = self._delimiter_coverage_counts(best_result)
        coverage_progress = covered / total if total else 1.0
        runtime_ok = (
            self.eval_max_seconds_per_example is None
            or best_result.max_sample_time_seconds < self.eval_max_seconds_per_example
        )
        is_near_win = (
            accuracy_progress >= 0.9
            and syntax_progress >= 0.85
            and coverage_progress >= 0.85
            and runtime_ok
        )
        thresholds_met = (
            best_result.accuracy >= self.min_accuracy
            and best_result.syntax_rate >= self.min_syntax_rate
            and (best_result.contains_delimiters or not self.require_delimiters)
            and runtime_ok
        )
        if not is_near_win or thresholds_met:
            return None

        weak_points: list[str] = []
        if best_result.accuracy < self.min_accuracy:
            weak_points.append(f"accuracy {best_result.accuracy:.1%}/{self.min_accuracy:.1%}")
        if best_result.syntax_rate < self.min_syntax_rate:
            weak_points.append(f"syntax {best_result.syntax_rate:.1%}/{self.min_syntax_rate:.1%}")
        if self.require_delimiters and (not best_result.contains_delimiters or covered < total):
            weak_points.append(f"delimiter coverage {covered}/{total}")
        if (
            self.eval_max_seconds_per_example is not None
            and best_result.max_sample_time_seconds >= self.eval_max_seconds_per_example
        ):
            weak_points.append(
                f"runtime {best_result.max_sample_time_seconds:.2f}s/"
                f"{self.eval_max_seconds_per_example:.2f}s"
            )
        weak_text = ", ".join(weak_points) if weak_points else "threshold margin"
        family = self._describe_outer_structure_signature(
            self._get_outer_structure_signature(best_attempt.strategy_code)
        )
        return (
            "- Near-win refinement mode: balanced-best is close to target "
            f"(remaining weak points: {weak_text}). Preserve its broad family "
            f"({family}), span-entry policy, answer-production path, and completion behavior. "
            "Make the smallest localized repair that targets those weak points. Do not introduce "
            "a new span-entry policy, broad chunking, or stronger inside control unless evidence "
            "shows the near-win structure cannot repair them."
        )

    def _get_valid_basin_refinement_summary(
        self,
        best_attempt: SynthesisAttempt,
    ) -> str | None:
        """Return a mode-selection line for valid but not near-winning basins."""
        best_result = best_attempt.eval_result
        if best_result is None:
            return None

        covered, total = self._delimiter_coverage_counts(best_result)
        delimiter_ok = (
            not self.require_delimiters
            or (best_result.contains_delimiters and (not total or covered / total >= 0.9))
        )
        syntax_target = self.min_syntax_rate if self.min_syntax_rate > 0 else 1.0
        syntax_progress = min(best_result.syntax_rate / syntax_target, 1.0)
        runtime_ok = (
            self.eval_max_seconds_per_example is None
            or best_result.max_sample_time_seconds < self.eval_max_seconds_per_example
        )
        thresholds_met = (
            best_result.accuracy >= self.min_accuracy
            and best_result.syntax_rate >= self.min_syntax_rate
            and (best_result.contains_delimiters or not self.require_delimiters)
            and runtime_ok
        )
        if thresholds_met or not delimiter_ok or syntax_progress < 0.8 or not runtime_ok:
            return None

        family = self._describe_outer_structure_signature(
            self._get_outer_structure_signature(best_attempt.strategy_code)
        )
        coverage_text = f"{covered}/{total}" if total else "n/a"
        return (
            "- Valid-basin refinement mode: balanced-best is delimiter-valid with decent syntax "
            f"({best_result.syntax_rate:.1%} syntax, coverage {coverage_text}) but is not yet "
            "near target. Preserve useful ingredients as evidence-backed components that "
            "contributed to positive metric shifts, not as a full strategy template. Change "
            "one causal axis per attempt and name which useful ingredients are preserved "
            f"unchanged from the valid basin ({family}). Broad structural change is allowed "
            "only after failed single-axis repairs or when metrics show a specific useful "
            "ingredient is no longer helping; even then, carry forward the useful ingredients "
            "that have not been disproven."
        )

    def _get_repeated_strategy_profile_summary(self, attempts: list[SynthesisAttempt]) -> str:
        """
        Summarize repeated evaluated strategy profiles using factual code features.

        This intentionally avoids hand-authored task labels. The goal is to make
        repeated empirical basins visible without telling the model what GSM-specific
        behavior to implement next.
        """
        evaluated_attempts = [
            attempt
            for attempt in attempts
            if attempt.eval_result is not None
        ]
        if len(evaluated_attempts) < 3:
            return ""

        best_attempt = max(
            evaluated_attempts,
            key=lambda attempt: self._evaluation_progress_score(attempt.eval_result),
        )
        best_score = self._evaluation_progress_score(best_attempt.eval_result)
        groups: dict[tuple, dict] = {}
        previous_eval: SynthesisAttempt | None = None

        for attempt in evaluated_attempts:
            result = attempt.eval_result
            if result is None:
                continue
            description, signature = self._get_strategy_profile_for_evaluation_history(attempt.strategy_code)
            group = groups.setdefault(
                signature,
                {
                    "description": description,
                    "behavior_summary": "",
                    "attempts": [],
                    "best_score": None,
                    "best_attempt": None,
                    "best_result": None,
                    "last_attempt": None,
                },
            )
            delta_text = ""
            if previous_eval is not None and previous_eval.eval_result is not None:
                previous_result = previous_eval.eval_result
                delta_text = (
                    f", delta vs previous eval acc {result.accuracy - previous_result.accuracy:+.1%}, "
                    f"syntax {result.syntax_rate - previous_result.syntax_rate:+.1%}"
                )
            group["attempts"].append(
                (
                    attempt.attempt_number,
                    result.accuracy,
                    result.syntax_rate,
                    result.max_sample_time_seconds,
                    delta_text,
                )
            )
            attempt_score = self._evaluation_progress_score(result)
            if group["best_score"] is None or attempt_score > group["best_score"]:
                group["best_score"] = attempt_score
                group["best_attempt"] = attempt.attempt_number
                group["best_result"] = result
                group["description"] = description
                group["behavior_summary"] = self._get_strategy_behavior_summary(attempt.strategy_code)
            group["last_attempt"] = attempt.attempt_number
            previous_eval = attempt

        repeated = [
            group
            for group in groups.values()
            if len(group["attempts"]) >= 2
        ]
        if not repeated:
            return ""

        repeated.sort(
            key=lambda group: (
                group["best_score"] is None or group["best_score"] < best_score,
                len(group["attempts"]),
                group["last_attempt"] or 0,
            ),
            reverse=True,
        )

        lines = [
            "Repeated evaluated strategy profiles:",
            (
                "Secondary detail: exact helper/control profiles are narrower than the top-level Search memory. "
                "Use this only to check whether a local helper-level variant has already repeated."
            ),
            (
                f"Current balanced-best attempt is {best_attempt.attempt_number}: "
                f"accuracy {best_attempt.eval_result.accuracy:.1%}, syntax {best_attempt.eval_result.syntax_rate:.1%}."
            ),
        ]

        for idx, group in enumerate(repeated[:4], start=1):
            best_result = group["best_result"]
            if best_result is None:
                continue
            status = (
                "matched or exceeded balanced-best"
                if group["best_score"] is not None and group["best_score"] >= best_score
                else "did not match balanced-best"
            )
            attempt_bits = [
                (
                    f"{number}: acc {accuracy:.1%}, syntax {syntax:.1%}, "
                    f"slowest {slowest:.2f}s{delta_text}"
                )
                for number, accuracy, syntax, slowest, delta_text in group["attempts"]
            ]
            lines.extend(
                [
                    f"Profile {idx} ({status}; best within profile attempt {group['best_attempt']}):",
                    f"  behavior summary: {self._truncate_words(group['behavior_summary'], 45)}",
                    "  outcomes: " + " | ".join(attempt_bits),
                ]
            )

        return "\n".join(lines)

    def _get_search_coverage_summary(self, attempts: list[SynthesisAttempt]) -> str:
        """Return a neutral summary of empirical plateau and helper-role coverage."""
        evaluated_attempts = [attempt for attempt in attempts if attempt.eval_result is not None]
        if len(evaluated_attempts) < 5:
            return ""

        best_attempt = max(
            evaluated_attempts,
            key=lambda attempt: self._evaluation_progress_score(attempt.eval_result),
        )
        best_result = best_attempt.eval_result
        if best_result is None:
            return ""

        best_score = self._evaluation_progress_score(best_result)
        attempts_since_best = [
            attempt for attempt in evaluated_attempts if attempt.attempt_number > best_attempt.attempt_number
        ]
        matched_or_improved_after_best = [
            attempt
            for attempt in attempts_since_best
            if attempt.eval_result is not None
            and self._evaluation_progress_score(attempt.eval_result) >= best_score
        ]

        recent = evaluated_attempts[-8:]
        outcome_buckets: dict[tuple[int, int, bool], list[int]] = {}
        for attempt in evaluated_attempts:
            result = attempt.eval_result
            if result is None:
                continue
            key = (
                round(result.accuracy * 100),
                round(result.syntax_rate * 100),
                bool(result.contains_delimiters),
            )
            outcome_buckets.setdefault(key, []).append(attempt.attempt_number)
        repeated_outcomes = [
            (key, numbers) for key, numbers in outcome_buckets.items() if len(numbers) >= 2
        ]
        repeated_outcomes.sort(key=lambda item: (len(item[1]), item[1][-1]), reverse=True)

        all_helpers: set[str] = set()
        for attempt in evaluated_attempts:
            all_helpers.update(self._get_helper_calls_for_evaluation_history(attempt.strategy_code))

        helper_roles = {
            "confidence-gated constrained stepping": {"ConfidenceGatedStep"},
            "safe logit-shaped constrained stepping": {
                "SafeSoftConstrainedStep",
                "SafeRepetitionPenaltyStep",
                "SafeTemperatureConstrainedStep",
                "SafePenalizedConstrainedStep",
                "SafeBoostedConstrainedStep",
            },
            "symbol-level constrained progress": {
                "ConstrainedSymbol",
                "ConstrainedSymbolInGenerated",
            },
            "adaptive/group-biased constrained token stepping": {
                "AdaptiveConstrainedStep",
                "AdaptiveConstrainedStepWithPenalties",
                "GroupBoostedConstrainedStep",
                "GroupHasValidMember",
                "BoostValidGroups",
            },
            "rollback repair": {"RollbackConstrainedSuffix", "RollbackConstrainedSpan", "RollbackToValidPrefix"},
            "observed delimiter entry": {"EnterObservedConstrainedSpan"},
            "explicit delimiter entry": {"OpenConstrainedSpan"},
        }
        role_lines = []
        for role, helper_names in helper_roles.items():
            used = sorted(all_helpers & helper_names)
            if used:
                role_lines.append(f"  {role}: present via {', '.join(used)}")
            else:
                role_lines.append(f"  {role}: not present in evaluated strategies")

        lines = [
            "Search coverage and plateau summary:",
            (
                f"Best balanced attempt so far: attempt {best_attempt.attempt_number} "
                f"with accuracy {best_result.accuracy:.1%}, syntax {best_result.syntax_rate:.1%}, "
                f"contains << >> {'yes' if best_result.contains_delimiters else 'no'}, "
                f"slowest {best_result.max_sample_time_seconds:.2f}s."
            ),
            (
                f"Evaluated attempts since that best: {len(attempts_since_best)}; "
                f"matched-or-improved balanced score after best: {len(matched_or_improved_after_best)}."
            ),
            "Recent evaluated outcomes: "
            + " | ".join(
                f"{attempt.attempt_number}: acc {attempt.eval_result.accuracy:.1%}, "
                f"syntax {attempt.eval_result.syntax_rate:.1%}, "
                f"contains {'yes' if attempt.eval_result.contains_delimiters else 'no'}"
                for attempt in recent
                if attempt.eval_result is not None
            ),
            "Repeated exact outcome bands: "
            + (
                " | ".join(
                    f"acc {key[0]}%, syntax {key[1]}%, contains {'yes' if key[2] else 'no'}: "
                    f"attempts {', '.join(str(number) for number in numbers[-6:])}"
                    for key, numbers in repeated_outcomes[:5]
                )
                if repeated_outcomes
                else "none"
            ),
            "Helper-role coverage across evaluated strategies:",
            *role_lines,
        ]
        return "\n".join(lines)

    def _delimiter_coverage_counts(self, result: EvaluationResult) -> tuple[int, int]:
        """Return per-example required delimiter/span coverage for prompt feedback."""
        samples = result.sample_outputs or []
        if not samples:
            total = result.num_examples
            covered = total if result.contains_delimiters else 0
            return covered, total
        total = len(samples)
        covered = sum(1 for sample in samples if bool(sample.get("contains_delimiters")))
        return covered, total

    def _format_delimiter_contract_status(self, result: EvaluationResult) -> str:
        if not self.require_delimiters:
            return "delimiter optional"
        covered, total = self._delimiter_coverage_counts(result)
        return (
            f"delimiter {'present' if result.contains_delimiters else 'absent'} "
            f"({covered}/{total} examples with required complete span/chunk)"
        )

    def _get_dual_anchor_summary(
        self,
        evaluated_attempts: list[SynthesisAttempt],
    ) -> str | None:
        """Return a compact contract-vs-accuracy anchor line when they diverge."""
        if not evaluated_attempts:
            return None

        def coverage_fraction(result: EvaluationResult) -> float:
            covered, total = self._delimiter_coverage_counts(result)
            return covered / total if total else 1.0

        contract_candidates = [
            attempt
            for attempt in evaluated_attempts
            if attempt.eval_result is not None
            and (not self.require_delimiters or attempt.eval_result.contains_delimiters)
        ]
        contract_anchor = (
            max(
                contract_candidates,
                key=lambda attempt: (
                    coverage_fraction(attempt.eval_result),
                    attempt.eval_result.syntax_rate,
                    attempt.eval_result.accuracy,
                ),
            )
            if contract_candidates
            else None
        )
        accuracy_anchor = max(
            evaluated_attempts,
            key=lambda attempt: (
                attempt.eval_result.accuracy,
                attempt.eval_result.syntax_rate,
                coverage_fraction(attempt.eval_result),
                bool(attempt.eval_result.contains_delimiters),
            ),
        )
        accuracy_result = accuracy_anchor.eval_result
        if accuracy_result is None:
            return None
        accuracy_ingredients = ", ".join(
            self._get_strategy_ingredient_parts(accuracy_anchor)[:5]
        )

        accuracy_text = (
            f"accuracy anchor attempt {accuracy_anchor.attempt_number}: "
            f"{accuracy_result.accuracy:.1%} accuracy / {accuracy_result.syntax_rate:.1%} syntax / "
            f"{self._format_delimiter_contract_status(accuracy_result)}; "
            f"accuracy ingredients: {accuracy_ingredients}"
        )
        if contract_anchor is None:
            return (
                "- Dual-anchor evidence: no delimiter-valid contract anchor yet; "
                f"{accuracy_text}. Treat its answer-production behavior as tentative evidence "
                "while first restoring the hard delimiter/coverage contract."
            )

        contract_result = contract_anchor.eval_result
        if contract_result is None:
            return None
        contract_ingredients = ", ".join(
            self._get_strategy_ingredient_parts(contract_anchor)[:5]
        )
        if contract_anchor.attempt_number == accuracy_anchor.attempt_number:
            return (
                "- Dual-anchor evidence: contract/syntax and accuracy anchors are currently the same "
                f"attempt {contract_anchor.attempt_number}; shared ingredients: {contract_ingredients}. "
                "Preserve its contract mechanics while "
                "changing only the measured weak point."
            )

        return (
            "- Dual-anchor evidence: "
            f"contract/syntax anchor attempt {contract_anchor.attempt_number}: "
            f"{contract_result.accuracy:.1%} accuracy / {contract_result.syntax_rate:.1%} syntax / "
            f"{self._format_delimiter_contract_status(contract_result)}; "
            f"contract ingredients: {contract_ingredients}; "
            f"{accuracy_text}. Use the contract anchor for delimiter/syntax/runtime mechanics and "
            "the accuracy anchor for answer-production evidence. Preferred merge/repair: preserve "
            "the contract anchor's delimiter contract while importing only one accuracy-improving "
            "ingredient from the accuracy anchor; alternatively repair the accuracy anchor's "
            "contract if that is the smaller change. Name which operation you are doing."
        )

    def _get_compact_search_memory(
        self,
        attempts: list[SynthesisAttempt],
        current_attempt: SynthesisAttempt | None = None,
        repair_stage: str | None = None,
    ) -> str:
        """Return a short, high-salience search-state block for refinement prompts."""
        evaluated_attempts = [attempt for attempt in attempts if attempt.eval_result is not None]
        if not evaluated_attempts:
            return ""

        best_attempt = max(
            evaluated_attempts,
            key=lambda attempt: self._evaluation_progress_score(attempt.eval_result),
        )
        best_result = best_attempt.eval_result
        if best_result is None:
            return ""

        lines = [
            "Search memory:",
            (
                f"- Balanced-best: attempt {best_attempt.attempt_number}, "
                f"{best_result.accuracy:.1%} accuracy / {best_result.syntax_rate:.1%} syntax / "
                f"{self._format_delimiter_contract_status(best_result)}."
            ),
        ]
        if self.require_delimiters and not best_result.contains_delimiters:
            covered, total = self._delimiter_coverage_counts(best_result)
            lines.append(
                "- Hard delimiter contract: "
                f"balanced-best currently has required complete spans/chunks in only {covered}/{total} examples; "
                "do not treat it as sufficient until coverage is restored."
            )
        if (
            self.eval_max_seconds_per_example is not None
            and best_result.max_sample_time_seconds >= self.eval_max_seconds_per_example
        ):
            lines.append(
                "- Runtime contract: "
                f"balanced-best hit {best_result.max_sample_time_seconds:.2f}s "
                f"against the {self.eval_max_seconds_per_example:.2f}s per-example limit; "
                "restore completion/runtime before treating score changes as progress."
            )

        dual_anchor_line = self._get_dual_anchor_summary(evaluated_attempts)
        if dual_anchor_line:
            lines.append(dual_anchor_line)

        useful_ingredients_line = self._get_useful_ingredients_summary(attempts, best_attempt)
        if useful_ingredients_line:
            lines.append(useful_ingredients_line)
        near_win_line = self._get_near_win_refinement_summary(best_attempt)
        if near_win_line:
            lines.append(near_win_line)
        else:
            valid_basin_line = self._get_valid_basin_refinement_summary(best_attempt)
            if valid_basin_line:
                lines.append(valid_basin_line)

        previous_eval = evaluated_attempts[-1]
        previous_result = previous_eval.eval_result
        if previous_result is not None and previous_eval is not best_attempt:
            previous_score = self._evaluation_progress_score(previous_result)
            best_score = self._evaluation_progress_score(best_result)
            if previous_score > best_score:
                relation = "improved beyond balanced-best"
            elif previous_score == best_score:
                relation = "matched balanced-best"
            else:
                relation = "did not improve balanced-best"
            lines.append(
                f"- Previous evaluated: attempt {previous_eval.attempt_number}, "
                f"{previous_result.accuracy:.1%} accuracy / {previous_result.syntax_rate:.1%} syntax / "
                f"{self._format_delimiter_contract_status(previous_result)}; {relation}."
            )
            if self.require_delimiters and not previous_result.contains_delimiters:
                covered, total = self._delimiter_coverage_counts(previous_result)
                lines.append(
                    "- Hard delimiter contract: "
                    f"previous attempt produced required complete spans/chunks in only {covered}/{total} examples; "
                    "restore this coverage before treating accuracy-only gains as progress."
                )
            if (
                self.eval_max_seconds_per_example is not None
                and previous_result.max_sample_time_seconds >= self.eval_max_seconds_per_example
            ):
                lines.append(
                    "- Runtime contract: "
                    f"previous attempt hit {previous_result.max_sample_time_seconds:.2f}s "
                    f"against the {self.eval_max_seconds_per_example:.2f}s per-example limit; "
                    "do not repeat its control profile unless the next change directly targets runtime/completion."
                )

        best_score = self._evaluation_progress_score(best_result)
        outcome_buckets: dict[tuple[int, int, bool], list[SynthesisAttempt]] = {}
        for attempt in evaluated_attempts:
            result = attempt.eval_result
            if result is None:
                continue
            key = (
                round(result.accuracy * 100),
                round(result.syntax_rate * 100),
                bool(result.contains_delimiters),
            )
            outcome_buckets.setdefault(key, []).append(attempt)

        repeated_outcomes = []
        for key, bucket_attempts in outcome_buckets.items():
            after_best = [
                attempt
                for attempt in bucket_attempts
                if attempt.attempt_number > best_attempt.attempt_number
            ]
            bucket_best_score = max(
                self._evaluation_progress_score(attempt.eval_result)
                for attempt in bucket_attempts
                if attempt.eval_result is not None
            )
            if bucket_best_score >= best_score:
                continue
            if len(bucket_attempts) >= 3 or len(after_best) >= 2:
                repeated_outcomes.append((key, bucket_attempts, after_best))

        repeated_outcomes.sort(
            key=lambda item: (len(item[1]), item[1][-1].attempt_number),
            reverse=True,
        )
        for key, bucket_attempts, _after_best in repeated_outcomes[:2]:
            numbers = ", ".join(str(attempt.attempt_number) for attempt in bucket_attempts[-6:])
            lines.append(
                "- Repeated outcome trap: "
                f"{key[0]}% accuracy / {key[1]}% syntax / "
                f"required delimiter {'present' if key[2] else 'absent'} in attempts {numbers}."
            )

        for broad_family_line in self._get_repeated_outer_structure_summary(attempts):
            lines.append(f"- {broad_family_line}")

        if repair_stage:
            attempt_text = (
                f" attempt {current_attempt.attempt_number}"
                if current_attempt is not None
                else ""
            )
            lines.append(
                f"- Repair continuity: this {repair_stage} repair for{attempt_text} should preserve search memory; "
                "do not reset to a near-duplicate of a repeated non-winning outcome."
            )

        revision_check = (
            "- Revision check: first choose the refinement mode. Near target: use balanced-best "
            "as an anchor for a surgical repair. Valid basin but not near target: preserve useful "
            "ingredients, compare contract/syntax and accuracy anchors, and change one causal axis. "
            "When anchors differ, prefer merge/repair: preserve the contract anchor's delimiter "
            "contract while importing only one accuracy-improving ingredient from the accuracy "
            "anchor; or repair the accuracy anchor's contract if that is smaller. "
            "No valid basin, repeated failed single-axis repairs, or a disproven useful ingredient: "
            "make a causal structural change while carrying forward useful ingredients that still "
            "have positive metric evidence. Do not re-submit balanced-best or a balanced-best-like "
            "near-copy whose expected metric movement is unclear."
        )
        lines.append(revision_check)
        max_lines = 14
        if len(lines) <= max_lines:
            return "\n".join(lines)
        compact_lines = lines[: max_lines - 1]
        if revision_check not in compact_lines:
            compact_lines.append(revision_check)
        return "\n".join(compact_lines[:max_lines])

    def _evaluation_progress_score(self, result: EvaluationResult) -> tuple:
        """Rank evaluated attempts by balanced accuracy/syntax progress."""
        accuracy_target = self.min_accuracy if self.min_accuracy > 0 else 1.0
        syntax_target = self.min_syntax_rate if self.min_syntax_rate > 0 else 1.0
        accuracy_progress = min(result.accuracy / accuracy_target, 1.0)
        syntax_progress = min(result.syntax_rate / syntax_target, 1.0)
        balanced_progress = min(accuracy_progress, syntax_progress)
        delimiter_progress = 1.0 if result.contains_delimiters or not self.require_delimiters else 0.0
        runtime_progress = (
            1.0
            if self.eval_max_seconds_per_example is None
            or result.max_sample_time_seconds <= self.eval_max_seconds_per_example
            else 0.0
        )
        return (
            runtime_progress,
            delimiter_progress,
            balanced_progress,
            accuracy_progress + syntax_progress + delimiter_progress + runtime_progress,
            result.accuracy,
            result.syntax_rate,
            -result.max_sample_time_seconds,
        )

    @staticmethod
    def _short_unified_diff(before: str, after: str, before_label: str, after_label: str, max_lines: int = 80) -> str:
        """Return a bounded unified diff for prompt feedback."""
        diff_lines = list(
            unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=before_label,
                tofile=after_label,
                lineterm="",
                n=2,
            )
        )
        if not diff_lines:
            return "(no textual changes)"
        if len(diff_lines) > max_lines:
            omitted = len(diff_lines) - max_lines
            diff_lines = diff_lines[:max_lines] + [f"... ({omitted} diff lines omitted)"]
        return "\n".join(diff_lines)

    @staticmethod
    def _format_execution_counts(label: str, result: EvaluationResult) -> list[str]:
        """Return compact output-run buckets for prompt comparisons."""
        samples = result.sample_outputs
        n = len(samples) or result.num_examples
        errors = [str(sample.get("error") or "") for sample in samples if sample.get("error")]
        timeouts = sum(1 for sample in samples if sample.get("runtime_budget_exceeded"))
        nonempty_outputs = sum(
            1
            for sample in samples
            if int(sample.get("token_count", 0) or 0) > 0
            or bool((sample.get("scored_output") or sample.get("full_output") or "").strip())
        )
        lines = [
            f"{label} output run:",
            f"  generated_token_outputs_nonempty {nonempty_outputs}/{n}",
        ]
        if not errors and timeouts == 0:
            lines.append("  all evaluated examples completed and returned generated output records")
            return lines

        if errors:
            lines.append(f"  examples_without_completed_output_records {len(errors)}/{n}")
        if timeouts:
            lines.append(f"  per_example_time_budget_exceeded {timeouts}/{n}")
        return lines

    @staticmethod
    def _format_execution_delta(
        label: str,
        current_result: EvaluationResult,
        baseline_result: EvaluationResult,
    ) -> list[str]:
        """Return current-minus-baseline deltas for execution buckets."""
        def counts(result: EvaluationResult) -> dict[str, int]:
            samples = result.sample_outputs
            errors = [str(sample.get("error") or "") for sample in samples if sample.get("error")]
            return {
                "incomplete_output_records": len(errors),
                "timeouts": sum(1 for sample in samples if sample.get("runtime_budget_exceeded")),
                "nonempty_outputs": sum(
                    1
                    for sample in samples
                    if int(sample.get("token_count", 0) or 0) > 0
                    or bool((sample.get("scored_output") or sample.get("full_output") or "").strip())
                ),
            }

        current = counts(current_result)
        baseline = counts(baseline_result)
        lines = [
            f"{label} output-run delta current minus baseline:",
            f"  nonempty_outputs {current['nonempty_outputs'] - baseline['nonempty_outputs']:+d}",
        ]
        if (
            current["incomplete_output_records"] == 0
            and baseline["incomplete_output_records"] == 0
            and current["timeouts"] == 0
            and baseline["timeouts"] == 0
        ):
            lines.append("  both attempts completed all evaluated examples")
            return lines

        anomaly_parts = []
        if current["incomplete_output_records"] or baseline["incomplete_output_records"]:
            anomaly_parts.append(
                "incomplete_output_records "
                f"{current['incomplete_output_records'] - baseline['incomplete_output_records']:+d}"
            )
        if current["timeouts"] or baseline["timeouts"]:
            anomaly_parts.append(f"time_budget_exceeded {current['timeouts'] - baseline['timeouts']:+d}")
        if anomaly_parts:
            lines.append("  " + ", ".join(anomaly_parts))
        return lines

    @staticmethod
    def _format_diagnostic_counts(label: str, result: EvaluationResult) -> list[str]:
        """Return compact evaluator diagnostic buckets for prompt comparisons."""
        counts = result.get_diagnostic_counts()
        n = counts.get("examples", result.num_examples)
        return [
            f"{label} diagnostic buckets:",
            (
                f"  syntax_valid_correct {counts['syntax_valid_correct']}/{n}, "
                f"syntax_valid_wrong {counts['syntax_valid_wrong']}/{n}, "
                f"syntax_invalid_wrong {counts['syntax_invalid_wrong']}/{n}, "
                f"no_complete_span_wrong {counts['no_complete_span_wrong']}/{n}"
            ),
            (
                f"  answer_source last_visible_span {counts['answer_from_last_visible_span']}/{n}, "
                f"text_fallback {counts['answer_from_text_fallback']}/{n}, "
                f"none {counts['no_extracted_answer']}/{n}"
            ),
            (
                f"  span_use final_answer_span {counts['examples_with_final_answer_span']}/{n}, "
                f"valid_nonfinal_only {counts['examples_with_valid_nonfinal_spans_only']}/{n}, "
                f"no_valid_span {counts['examples_with_no_valid_span']}/{n}"
            ),
            (
                f"  constrained_activity examples_with_activity {counts['examples_with_constrained_activity']}/{n}, "
                f"visible_span_without_activity {counts['visible_span_without_constrained_activity']}/{n}, "
                f"wrong_with_activity {counts['wrong_with_constrained_activity']}/{n}, "
                f"wrong_without_activity {counts['wrong_without_constrained_activity']}/{n}"
            ),
        ]

    @staticmethod
    def _format_diagnostic_delta(
        label: str,
        current_result: EvaluationResult,
        baseline_result: EvaluationResult,
    ) -> list[str]:
        """Return current-minus-baseline deltas for key diagnostic buckets."""
        current = current_result.get_diagnostic_counts()
        baseline = baseline_result.get_diagnostic_counts()
        keys = [
            "syntax_valid_wrong",
            "syntax_invalid_wrong",
            "no_complete_span_wrong",
            "answer_from_last_visible_span",
            "answer_from_text_fallback",
            "no_extracted_answer",
            "examples_with_final_answer_span",
            "examples_with_valid_nonfinal_spans_only",
            "examples_with_no_valid_span",
            "examples_with_constrained_activity",
            "visible_span_without_constrained_activity",
            "wrong_with_constrained_activity",
            "wrong_without_constrained_activity",
        ]
        parts = [
            f"{key} {current.get(key, 0) - baseline.get(key, 0):+d}"
            for key in keys
        ]
        return [f"{label} diagnostic delta current minus baseline:", "  " + ", ".join(parts)]

    @staticmethod
    def _format_named_counter(counter, denominator: int, max_items: int = 5) -> str:
        from synthesis.evaluate.benchmarks.common.formatting import format_named_counter

        return format_named_counter(
            counter,
            denominator,
            max_items=max_items,
            min_denominator=1,
        )

    @staticmethod
    def _format_counter_delta(current_counter, baseline_counter, max_items: int = 6) -> str:
        from synthesis.evaluate.benchmarks.common.formatting import format_counter_delta

        return format_counter_delta(
            current_counter,
            baseline_counter,
            max_items=max_items,
        )

    @staticmethod
    def _format_provenance_counts(label: str, result: EvaluationResult) -> list[str]:
        """Return compact provenance/localization buckets for prompt comparisons."""
        counts = result.get_provenance_counts()
        n = len(result.sample_outputs) or result.num_examples
        return [
            f"{label} provenance/localization buckets:",
            "  answer_provenance "
            + SynthesisPipeline._format_named_counter(counts["answer_provenance"], n),
            "  control_tags "
            + SynthesisPipeline._format_named_counter(counts["control_tags"], n),
            "  failure_location "
            + SynthesisPipeline._format_named_counter(counts["failure_location"], n),
        ]

    @staticmethod
    def _format_provenance_delta(
        label: str,
        current_result: EvaluationResult,
        baseline_result: EvaluationResult,
    ) -> list[str]:
        current = current_result.get_provenance_counts()
        baseline = baseline_result.get_provenance_counts()
        return [
            f"{label} provenance/localization delta current minus baseline:",
            "  answer_provenance "
            + SynthesisPipeline._format_counter_delta(
                current["answer_provenance"], baseline["answer_provenance"]
            ),
            "  control_tags "
            + SynthesisPipeline._format_counter_delta(
                current["control_tags"], baseline["control_tags"]
            ),
            "  failure_location "
            + SynthesisPipeline._format_counter_delta(
                current["failure_location"], baseline["failure_location"]
            ),
        ]

    def _format_change_impact_attribution(
        self,
        label: str,
        before_attempt: SynthesisAttempt,
        after_attempt: SynthesisAttempt,
    ) -> list[str]:
        """Summarize the factual behavior shift associated with a code revision."""
        before_result = before_attempt.eval_result
        after_result = after_attempt.eval_result
        if before_result is None or after_result is None:
            return []
        before_helpers = set(self._get_helper_calls_for_evaluation_history(before_attempt.strategy_code))
        after_helpers = set(self._get_helper_calls_for_evaluation_history(after_attempt.strategy_code))
        added = ", ".join(sorted(after_helpers - before_helpers)) or "none"
        removed = ", ".join(sorted(before_helpers - after_helpers)) or "none"
        current_diag = after_result.get_diagnostic_counts()
        before_diag = before_result.get_diagnostic_counts()
        diag_keys = [
            "syntax_valid_wrong",
            "syntax_invalid_wrong",
            "no_complete_span_wrong",
            "examples_with_constrained_activity",
            "visible_span_without_constrained_activity",
            "wrong_with_constrained_activity",
            "wrong_without_constrained_activity",
        ]
        ranked_diag = sorted(
            diag_keys,
            key=lambda key: (-abs(current_diag.get(key, 0) - before_diag.get(key, 0)), key),
        )
        diag_delta = ", ".join(
            f"{key} {current_diag.get(key, 0) - before_diag.get(key, 0):+d}"
            for key in ranked_diag[:5]
        )
        before_prov = before_result.get_provenance_counts()
        after_prov = after_result.get_provenance_counts()
        return [
            f"{label} change impact attribution:",
            f"  helper_set_added {added}; helper_set_removed {removed}",
            (
                "  score_delta "
                f"accuracy {after_result.accuracy - before_result.accuracy:+.1%}, "
                f"syntax {after_result.syntax_rate - before_result.syntax_rate:+.1%}, "
                f"slowest {after_result.max_sample_time_seconds - before_result.max_sample_time_seconds:+.2f}s"
            ),
            f"  largest_diagnostic_shifts {diag_delta}",
            "  largest_provenance_shifts "
            + SynthesisPipeline._format_counter_delta(
                after_prov["answer_provenance"], before_prov["answer_provenance"], max_items=4
            ),
            "  largest_localization_shifts "
            + SynthesisPipeline._format_counter_delta(
                after_prov["failure_location"], before_prov["failure_location"], max_items=4
            ),
        ]

    def _get_best_so_far_comparison(self, attempts: list[SynthesisAttempt], current_attempt: SynthesisAttempt) -> str:
        """Compare the current evaluated attempt against the best previous evaluated attempt."""
        current_result = current_attempt.eval_result
        if current_result is None:
            return ""

        previous_evaluated = [
            attempt
            for attempt in attempts
            if attempt.eval_result is not None and attempt is not current_attempt
        ]
        if not previous_evaluated:
            return (
                "Best-so-far comparison:\n"
                f"Current attempt {current_attempt.attempt_number} is the first evaluated attempt."
            )

        best_attempt = max(previous_evaluated, key=lambda attempt: self._evaluation_progress_score(attempt.eval_result))
        best_result = best_attempt.eval_result
        if best_result is None:
            return ""

        current_score = self._evaluation_progress_score(current_result)
        best_score = self._evaluation_progress_score(best_result)
        if current_score > best_score:
            verdict = "improved over best-so-far"
        elif current_score == best_score:
            verdict = "tied best-so-far"
        else:
            verdict = "regressed from best-so-far"

        body_diff = self._short_unified_diff(
            self._get_strategy_body_for_evaluation_history(best_attempt.strategy_code),
            self._get_strategy_body_for_evaluation_history(current_attempt.strategy_code),
            f"attempt_{best_attempt.attempt_number}_body",
            f"attempt_{current_attempt.attempt_number}_body",
            max_lines=90,
        )

        return "\n".join(
            [
                "Best-so-far comparison:",
                (
                    "Best-so-far is selected by balanced accuracy/syntax progress. "
                    "A lopsided result with high syntax but much lower accuracy, or high accuracy but much lower syntax, "
                    "is not treated as best merely because one metric is strong."
                ),
                (
                    f"Current attempt {current_attempt.attempt_number}: "
                    f"accuracy {current_result.accuracy:.1%} ({current_result.num_correct}/{current_result.num_examples}), "
                    f"syntax {current_result.syntax_rate:.1%}, "
                    f"{self._format_delimiter_contract_status(current_result)}, "
                    f"slowest {current_result.max_sample_time_seconds:.2f}s"
                ),
                (
                    f"Best previous attempt {best_attempt.attempt_number}: "
                    f"accuracy {best_result.accuracy:.1%} ({best_result.num_correct}/{best_result.num_examples}), "
                    f"syntax {best_result.syntax_rate:.1%}, "
                    f"{self._format_delimiter_contract_status(best_result)}, "
                    f"slowest {best_result.max_sample_time_seconds:.2f}s"
                ),
                (
                    "Delta current minus best: "
                    f"accuracy {current_result.accuracy - best_result.accuracy:+.1%}, "
                    f"syntax {current_result.syntax_rate - best_result.syntax_rate:+.1%}, "
                    f"slowest {current_result.max_sample_time_seconds - best_result.max_sample_time_seconds:+.2f}s"
                ),
                *self._format_execution_counts("Current attempt", current_result),
                *self._format_execution_counts("Best previous attempt", best_result),
                *self._format_execution_delta(
                    "Best-so-far comparison",
                    current_result,
                    best_result,
                ),
                *self._format_diagnostic_counts("Current attempt", current_result),
                *self._format_diagnostic_counts("Best previous attempt", best_result),
                *self._format_diagnostic_delta(
                    "Best-so-far comparison",
                    current_result,
                    best_result,
                ),
                *self._format_provenance_counts("Current attempt", current_result),
                *self._format_provenance_counts("Best previous attempt", best_result),
                *self._format_provenance_delta(
                    "Best-so-far comparison",
                    current_result,
                    best_result,
                ),
                f"Assessment: current attempt {verdict}.",
                "Strategy body diff versus best-so-far:",
                body_diff,
            ]
        )

    def _get_working_hypothesis_state(self, attempts: list[SynthesisAttempt], current_attempt: SynthesisAttempt) -> str:
        """Describe strategy lineage without prescribing the next edit."""
        current_result = current_attempt.eval_result
        if current_result is None:
            return ""

        evaluated_attempts = [
            attempt
            for attempt in attempts
            if attempt.eval_result is not None
        ]
        if not evaluated_attempts:
            return ""

        balanced_best = max(
            evaluated_attempts,
            key=lambda attempt: self._evaluation_progress_score(attempt.eval_result),
        )
        best_result = balanced_best.eval_result
        if best_result is None:
            return ""

        previous_evaluated = [
            attempt
            for attempt in evaluated_attempts
            if attempt is not current_attempt and attempt.attempt_number < current_attempt.attempt_number
        ]
        previous_eval = previous_evaluated[-1] if previous_evaluated else None
        previous_result = previous_eval.eval_result if previous_eval is not None else None

        def attempt_line(label: str, attempt: SynthesisAttempt, result: EvaluationResult) -> str:
            return (
                f"{label}: Attempt {attempt.attempt_number}: "
                f"accuracy {result.accuracy:.1%} ({result.num_correct}/{result.num_examples}), "
                f"syntax {result.syntax_rate:.1%}, "
                f"{self._format_delimiter_contract_status(result)}, "
                f"slowest {result.max_sample_time_seconds:.2f}s"
            )

        lines = [
            "Working hypothesis state:",
            (
                "Best-so-far is selected by balanced accuracy/syntax progress; "
                "a lopsided result is not best merely because one metric is high."
            ),
            attempt_line("Current evaluated attempt", current_attempt, current_result),
            attempt_line("Current balanced-best attempt", balanced_best, best_result),
            (
                "Relation current minus balanced-best: "
                f"accuracy {current_result.accuracy - best_result.accuracy:+.1%}, "
                f"syntax {current_result.syntax_rate - best_result.syntax_rate:+.1%}, "
                f"slowest {current_result.max_sample_time_seconds - best_result.max_sample_time_seconds:+.2f}s"
            ),
            *self._format_execution_counts("Current evaluated attempt", current_result),
            *self._format_execution_counts("Current balanced-best attempt", best_result),
            *self._format_execution_delta(
                "Relation to balanced-best",
                current_result,
                best_result,
            ),
            *self._format_diagnostic_counts("Current evaluated attempt", current_result),
            *self._format_diagnostic_counts("Current balanced-best attempt", best_result),
            *self._format_diagnostic_delta(
                "Relation to balanced-best",
                current_result,
                best_result,
            ),
            *self._format_provenance_counts("Current evaluated attempt", current_result),
            *self._format_provenance_counts("Current balanced-best attempt", best_result),
            *self._format_provenance_delta(
                "Relation to balanced-best",
                current_result,
                best_result,
            ),
        ]

        if previous_eval is not None and previous_result is not None:
            lines.extend(
                [
                    attempt_line("Immediately previous evaluated attempt", previous_eval, previous_result),
                    (
                        "Relation current minus previous evaluated: "
                        f"accuracy {current_result.accuracy - previous_result.accuracy:+.1%}, "
                        f"syntax {current_result.syntax_rate - previous_result.syntax_rate:+.1%}, "
                        f"slowest {current_result.max_sample_time_seconds - previous_result.max_sample_time_seconds:+.2f}s"
                    ),
                    *self._format_execution_counts("Immediately previous evaluated attempt", previous_result),
                    *self._format_execution_delta(
                        "Relation to previous evaluated",
                        current_result,
                        previous_result,
                    ),
                    *self._format_diagnostic_counts("Immediately previous evaluated attempt", previous_result),
                    *self._format_diagnostic_delta(
                        "Relation to previous evaluated",
                        current_result,
                        previous_result,
                    ),
                    *self._format_provenance_counts("Immediately previous evaluated attempt", previous_result),
                    *self._format_provenance_delta(
                        "Relation to previous evaluated",
                        current_result,
                        previous_result,
                    ),
                    *self._format_change_impact_attribution(
                        "Previous-to-current evaluated",
                        previous_eval,
                        current_attempt,
                    ),
                    "Most recent modification summary versus previous evaluated attempt:",
                    "Strategy body diff:",
                    self._short_unified_diff(
                        self._get_strategy_body_for_evaluation_history(previous_eval.strategy_code),
                        self._get_strategy_body_for_evaluation_history(current_attempt.strategy_code),
                        f"attempt_{previous_eval.attempt_number}_body",
                        f"attempt_{current_attempt.attempt_number}_body",
                        max_lines=80,
                    ),
                ]
            )

        repeated_profiles = self._get_repeated_strategy_profile_summary(attempts)
        if repeated_profiles:
            lines.extend(["", repeated_profiles])

        search_coverage = self._get_search_coverage_summary(attempts)
        if search_coverage:
            lines.extend(["", search_coverage])

        if balanced_best is not current_attempt:
            lines.extend(
                [
                    "Complete balanced-best strategy body without rationale/proof sketch:",
                    "```dafny",
                    self._get_strategy_body_for_evaluation_history(balanced_best.strategy_code),
                    "```",
                ]
            )

        return "\n".join(lines)

    def _get_verification_refinement_context(
        self,
        attempts: list[SynthesisAttempt],
        current_attempt: SynthesisAttempt,
    ) -> str:
        """Describe evaluated lineage when a verification-only attempt needs repair."""
        evaluated_attempts = [
            attempt
            for attempt in attempts
            if attempt.eval_result is not None
        ]
        if not evaluated_attempts:
            return ""

        balanced_best = max(
            evaluated_attempts,
            key=lambda attempt: self._evaluation_progress_score(attempt.eval_result),
        )
        best_result = balanced_best.eval_result
        if best_result is None:
            return ""

        previous_eval = evaluated_attempts[-1]
        previous_result = previous_eval.eval_result

        def attempt_line(label: str, attempt: SynthesisAttempt, result: EvaluationResult) -> str:
            return (
                f"{label}: Attempt {attempt.attempt_number}: "
                f"accuracy {result.accuracy:.1%} ({result.num_correct}/{result.num_examples}), "
                f"syntax {result.syntax_rate:.1%}, "
                f"{self._format_delimiter_contract_status(result)}, "
                f"slowest {result.max_sample_time_seconds:.2f}s"
            )

        lines = [
            "Strategy context before verification failure:",
            (
                f"Current attempt {current_attempt.attempt_number} failed verification and has not been evaluated. "
                "The evaluated attempts below are the current empirical context."
            ),
            (
                "Best-so-far is selected by balanced accuracy/syntax progress; "
                "a lopsided result is not best merely because one metric is high."
            ),
            attempt_line("Current balanced-best evaluated attempt", balanced_best, best_result),
            *self._format_diagnostic_counts("Current balanced-best evaluated attempt", best_result),
            *self._format_provenance_counts("Current balanced-best evaluated attempt", best_result),
        ]

        if previous_result is not None:
            lines.extend(
                [
                    attempt_line("Immediately previous evaluated attempt", previous_eval, previous_result),
                    *self._format_diagnostic_counts("Immediately previous evaluated attempt", previous_result),
                    *self._format_provenance_counts("Immediately previous evaluated attempt", previous_result),
                    *self._format_diagnostic_delta(
                        "Previous evaluated versus balanced-best",
                        previous_result,
                        best_result,
                    ),
                    *self._format_provenance_delta(
                        "Previous evaluated versus balanced-best",
                        previous_result,
                        best_result,
                    ),
                    "Modification diff from previous evaluated attempt to verification-failed attempt:",
                    self._short_unified_diff(
                        self._get_strategy_body_for_evaluation_history(previous_eval.strategy_code),
                        self._get_strategy_body_for_evaluation_history(current_attempt.strategy_code),
                        f"attempt_{previous_eval.attempt_number}_body",
                        f"attempt_{current_attempt.attempt_number}_body",
                        max_lines=80,
                    ),
                ]
            )

        repeated_profiles = self._get_repeated_strategy_profile_summary(attempts)
        if repeated_profiles:
            lines.extend(["", repeated_profiles])

        search_coverage = self._get_search_coverage_summary(attempts)
        if search_coverage:
            lines.extend(["", search_coverage])

        lines.extend(
            [
                "Complete balanced-best strategy body without rationale/proof sketch:",
                "```dafny",
                self._get_strategy_body_for_evaluation_history(balanced_best.strategy_code),
                "```",
            ]
        )

        return "\n".join(lines)

    def _get_evaluation_history_summary(self, attempts: list[SynthesisAttempt]) -> str:
        """Summarize evaluated attempts with factual metrics and strategy bodies."""
        evaluated_attempts = [
            attempt
            for attempt in attempts
            if attempt.eval_result is not None
        ]
        if not evaluated_attempts:
            return ""

        lines = [
            "Evaluation attempts only; verification-only attempts are omitted.",
            (
                "Each attempt includes compact approach/family notes instead of full bodies; "
                "full bodies are supplied separately for the current, previous, or balanced-best strategy when needed."
            ),
        ]
        previous = None
        for attempt in evaluated_attempts:
            result = attempt.eval_result
            if result is None:
                continue
            line = (
                f"Attempt {attempt.attempt_number}: "
                f"accuracy {result.accuracy:.1%} ({result.num_correct}/{result.num_examples}), "
                f"syntax {result.syntax_rate:.1%}, "
                f"{self._format_delimiter_contract_status(result)}, "
                f"slowest {result.max_sample_time_seconds:.2f}s"
            )
            if previous is not None:
                line += (
                    " | delta vs previous eval: "
                    f"accuracy {result.accuracy - previous.accuracy:+.1%}, "
                    f"syntax {result.syntax_rate - previous.syntax_rate:+.1%}, "
                    f"slowest {result.max_sample_time_seconds - previous.max_sample_time_seconds:+.2f}s"
                )
            lines.append(line)
            broad_family = self._describe_outer_structure_signature(
                self._get_outer_structure_signature(attempt.strategy_code)
            )
            approach = self._truncate_words(
                self._get_strategy_behavior_summary(attempt.strategy_code),
                70,
            )
            helpers = ", ".join(self._get_helper_calls_for_evaluation_history(attempt.strategy_code))
            lines.append(f"  broad_family: {broad_family}")
            lines.append(f"  approach: {approach}")
            lines.append(f"  helper_palette: {helpers if helpers else '(none)'}")
            previous = result

        return "\n".join(lines)

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
                print("  ✗ Helper-call contract violation")
                error_msg = (
                    "Helper-call contract violation.\n"
                    f"Disallowed helper calls: {', '.join(disallowed_helpers)}"
                )
                attempt.failed_at = FailureStage.SEARCH_CONTRACT
                attempt.error_summary = error_msg
                attempts.append(attempt)

                print("  Refining based on helper-call contract violation...")
                next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
                if next_helper_status:
                    print(f"  Helper policy: {next_helper_status}")
                search_memory = self._get_compact_search_memory(
                    attempts,
                    current_attempt=attempt,
                    repair_stage="search_contract",
                )
                strategy_code = self._refine_with_beam(
                    stage_label="search_contract",
                    previous_strategy=strategy_code,
                    allowed_helpers=next_allowed_helpers,
                    refine_once=lambda: self.generator.refine_after_verification_error(
                        strategy_code,
                        error_msg,
                        search_memory=search_memory,
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
                strategy_context = self._get_verification_refinement_context(attempts, attempt)
                behavioral_context = self._get_recent_behavioral_context(attempts)
                search_memory = self._get_compact_search_memory(
                    attempts,
                    current_attempt=attempt,
                    repair_stage="verification",
                )
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
                        strategy_context=strategy_context,
                        search_memory=search_memory,
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
                search_memory = self._get_compact_search_memory(
                    attempts,
                    current_attempt=attempt,
                    repair_stage="compilation",
                )
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
                        search_memory=search_memory,
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

                search_memory = self._get_compact_search_memory(
                    attempts,
                    current_attempt=attempt,
                    repair_stage="runtime",
                )
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
                        search_memory=search_memory,
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

            # Rotate eval seed each iteration so the gate moves and the
            # synthesis loop can't local-search a single sample's quirks.
            if not hasattr(self, "_eval_base_seed"):
                self._eval_base_seed = (
                    int(self.evaluator.sample_seed)
                    if self.evaluator.sample_seed is not None
                    else 0
                )
            self.evaluator.sample_seed = self._eval_base_seed + (attempt.attempt_number - 1)
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
                    int(os.environ.get("CSD_SYNTHESIS_EVAL_EARLY_STOP_RUNTIME_FAILURES", "1"))
                    if os.environ.get("CSD_SYNTHESIS_EVAL_EARLY_STOP", "1") != "0"
                    else None
                ),
            )
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

                self.evaluator.unload_runtime()
                print("  Evaluator runtime unloaded to free GPU memory")
                print("  Refining based on evaluation error...")
                eval_history = self._get_evaluation_history_summary(attempts)
                working_hypothesis = self._get_working_hypothesis_state(attempts, attempt)
                evaluation_feedback = eval_result.get_feedback_summary()
                search_memory = self._get_compact_search_memory(attempts, current_attempt=attempt)
                next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
                if next_helper_status:
                    print(f"  Helper policy: {next_helper_status}")
                strategy_code = self._refine_with_beam(
                    stage_label="evaluation_error",
                    previous_strategy=strategy_code,
                    allowed_helpers=next_allowed_helpers,
                    refine_once=lambda: self.generator.refine_after_evaluation_failure(
                        strategy_code,
                        evaluation_feedback,
                        evaluation_history=eval_history,
                        working_hypothesis=working_hypothesis,
                        search_memory=search_memory,
                        allowed_helpers=next_allowed_helpers,
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

                self.evaluator.unload_runtime()
                print("  Evaluator runtime unloaded to free GPU memory")
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
                eval_history = self._get_evaluation_history_summary(attempts)
                working_hypothesis = self._get_working_hypothesis_state(attempts, attempt)
                search_memory = self._get_compact_search_memory(attempts, current_attempt=attempt)
                next_allowed_helpers, next_helper_status = self._compute_allowed_helpers(attempts)
                if next_helper_status:
                    print(f"  Helper policy: {next_helper_status}")
                strategy_code = self._refine_with_beam(
                    stage_label="evaluation_threshold",
                    previous_strategy=strategy_code,
                    allowed_helpers=next_allowed_helpers,
                    refine_once=lambda: self.generator.refine_after_evaluation_failure(
                        strategy_code,
                        threshold_feedback,
                        evaluation_history=eval_history,
                        working_hypothesis=working_hypothesis,
                        search_memory=search_memory,
                        allowed_helpers=next_allowed_helpers,
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
                run_dir,
                run_results_dir,
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
        run_dir: Path,
        results_dir: Path,
    ) -> Path:
        """Save a detailed failure report to disk."""
        report_path = results_dir / "failure_report.json"

        report = {
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
