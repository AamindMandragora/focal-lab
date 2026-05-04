"""
Main synthesis pipeline with feedback-based refinement.

Orchestrates the generate -> verify -> compile -> run loop with
iterative refinement based on errors.
"""

import json
import re
import secrets
from difflib import unified_diff
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from .compiler import CompilationResult, DafnyCompiler
from .evaluator import Evaluator, EvaluationResult
from .generator import StrategyGenerator
from .rationale import extract_rationale
from .runner import RuntimeResult, StrategyRunner
from .verifier import DafnyVerifier, VerificationResult


class FailureStage(Enum):
    """Stage where synthesis attempt failed."""

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
    runtime_result: Optional[RuntimeResult] = None
    eval_result: Optional[EvaluationResult] = None

    # Failure information
    failed_at: Optional[FailureStage] = None
    error_summary: str = ""

    def succeeded(self) -> bool:
        """Check if this attempt succeeded completely."""
        return (
            self.verification_result is not None
            and self.verification_result.success
            and self.compilation_result is not None
            and self.compilation_result.success
            and self.runtime_result is not None
            and self.runtime_result.success
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
            "runtime": {
                "success": self.runtime_result.success if self.runtime_result else None,
                "output_length": len(self.runtime_result.output)
                if self.runtime_result and self.runtime_result.output
                else 0,
                "cost": self.runtime_result.cost if self.runtime_result else 0,
                "execution_time_ms": self.runtime_result.execution_time_ms if self.runtime_result else 0,
            }
            if self.runtime_result
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

    DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "generated-csd"

    def __init__(
        self,
        evaluator: Evaluator,
        generator: Optional[StrategyGenerator] = None,
        verifier: Optional[DafnyVerifier] = None,
        compiler: Optional[DafnyCompiler] = None,
        runner: Optional[StrategyRunner] = None,
        max_iterations: int = 5,
        output_dir: Optional[Path] = None,
        save_reports: bool = True,
        # Evaluation thresholds
        min_accuracy: float = 0.0,
        min_syntax_rate: float = 0.0,
        require_delimiters: bool = True,
        eval_sample_size: int = 10,
        eval_max_seconds_per_example: Optional[float] = None,
    ):
        """
        Initialize the synthesis pipeline.

        Args:
            evaluator: Evaluator for dataset-based feedback (required)
            generator: Strategy generator (creates default if None)
            verifier: Dafny verifier (creates default if None)
            compiler: Dafny compiler (creates default if None)
            runner: Strategy runner (creates default if None)
            max_iterations: Maximum refinement iterations
            output_dir: Directory for outputs and reports
            save_reports: Whether to save failure reports to disk
            min_accuracy: Minimum accuracy threshold for evaluation
            min_syntax_rate: Minimum syntax validity rate threshold
            require_delimiters: Whether evaluated outputs must contain << >> spans
            eval_sample_size: Number of examples to evaluate on
            eval_max_seconds_per_example: Optional runtime budget per example in seconds
        """
        self.evaluator = evaluator
        self.generator = generator or StrategyGenerator()
        self.verifier = verifier or DafnyVerifier()
        self.compiler = compiler or DafnyCompiler()
        self.runner = runner  # Will be created per-task in synthesize()
        self.max_iterations = max_iterations
        self.output_dir = output_dir or self.DEFAULT_OUTPUT_DIR
        self.save_reports = save_reports

        # Evaluation thresholds
        self.min_accuracy = min_accuracy
        self.min_syntax_rate = min_syntax_rate
        self.require_delimiters = require_delimiters
        self.eval_sample_size = eval_sample_size
        self.eval_max_seconds_per_example = eval_max_seconds_per_example

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
        """Return model-facing helper calls used by a strategy body."""
        body = self._get_strategy_body_for_evaluation_history(strategy_code)
        calls = re.findall(r"\bhelpers\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", body)
        return sorted(set(calls))

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
                "Profiles are grouped by observable helper usage and control facts, not by task-specific labels. "
                "If a repeated profile has not matched the balanced-best result, further revisions should make a real structural change instead of only retuning literals or thresholds inside that same profile."
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
                    f"  behavior summary: {group['behavior_summary']}",
                    f"  {group['description']}",
                    "  outcomes: " + " | ".join(attempt_bits),
                ]
            )

        return "\n".join(lines)

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
                    f"contains << >> {'yes' if current_result.contains_delimiters else 'no'}, "
                    f"slowest {current_result.max_sample_time_seconds:.2f}s"
                ),
                (
                    f"Best previous attempt {best_attempt.attempt_number}: "
                    f"accuracy {best_result.accuracy:.1%} ({best_result.num_correct}/{best_result.num_examples}), "
                    f"syntax {best_result.syntax_rate:.1%}, "
                    f"contains << >> {'yes' if best_result.contains_delimiters else 'no'}, "
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
                f"contains << >> {'yes' if result.contains_delimiters else 'no'}, "
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

    def _get_evaluation_history_summary(self, attempts: list[SynthesisAttempt]) -> str:
        """Summarize evaluated attempts with factual metrics and strategy bodies."""
        evaluated_attempts = [
            attempt
            for attempt in attempts
            if attempt.eval_result is not None
        ]
        if not evaluated_attempts:
            return ""

        lines = ["Evaluation attempts only; verification-only attempts are omitted."]
        previous = None
        for attempt in evaluated_attempts:
            result = attempt.eval_result
            if result is None:
                continue
            line = (
                f"Attempt {attempt.attempt_number}: "
                f"accuracy {result.accuracy:.1%} ({result.num_correct}/{result.num_examples}), "
                f"syntax {result.syntax_rate:.1%}, "
                f"contains << >> {'yes' if result.contains_delimiters else 'no'}, "
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
            lines.append("BEGIN STRATEGY BODY")
            lines.append(self._get_strategy_body_for_evaluation_history(attempt.strategy_code))
            lines.append("END STRATEGY BODY")
            previous = result

        return "\n".join(lines)

    def synthesize(
        self,
        task_description: str,
        output_name: str = "generated_csd",
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

        # Create runner if not already provided
        if self.runner is None:
            # Pull grammar from the evaluator so the smoke test exercises
            # the SAME grammar the real evaluation will use. This removes
            # the whole class of "TestParser missing method X" false negatives.
            grammar_source = None
            grammar_start = "start"
            try:
                grammar_source = str(self.evaluator._get_grammar_file())
            except Exception:
                grammar_source = None
            runner = StrategyRunner(
                parser_mode="permissive",
                grammar_source=grammar_source,
                grammar_start=grammar_start,
            )
        else:
            runner = self.runner

        # Create an isolated output directory for this run
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
        run_dir = self.output_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Update a convenience pointer to the most recent run
        try:
            (self.output_dir / "latest_run.txt").write_text(str(run_dir) + "\n")
        except Exception:
            pass

        # Use a per-run compiler output directory.
        compiler = DafnyCompiler(
            dafny_path=self.compiler.dafny_path,
            output_dir=run_dir,
            timeout=self.compiler.timeout,
            extra_args=list(self.compiler.extra_args),
        )

        # Initial generation
        print(f"Generating initial strategy for: {task_description}")
        strategy_code = self.generator.generate_initial(task_description)

        # Index in `attempts` after which we last performed a fresh restart.
        # Used to bound the "consecutive verification failures since last restart"
        # counter so that a restart resets it.
        last_restart_index = 0

        for iteration in range(self.max_iterations):
            attempt_num = iteration + 1
            print(f"\n{'='*60}")
            print(f"Attempt {attempt_num}/{self.max_iterations}")
            print(f"{'='*60}")
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
                    strategy_code = self.generator.generate_initial(task_description)
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
                    strategy_code = self.generator.generate_initial(task_description)
                    last_restart_index = len(attempts)
                    continue

                # Refine based on verification error
                print("  Refining based on verification error...")
                structured_feedback = verification_result.get_structured_feedback()
                error_history = self._get_verification_history_summary(attempts)
                strategy_code = self.generator.refine_after_verification_error(
                    strategy_code,
                    error_msg,
                    "",
                    structured_feedback,
                    error_history,
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
                strategy_code = self.generator.refine_after_compilation_error(
                    strategy_code, compilation_result.get_error_summary()
                )
                continue

            print(f"  ✓ Compiled to {compilation_result.output_dir}")

            if compilation_result.main_module_path is None:
                print("  ✗ No main module found")
                attempt.failed_at = FailureStage.RUNTIME
                attempt.error_summary = "No main module path in compilation result"
                attempts.append(attempt)

                strategy_code = self.generator.refine_after_runtime_error(
                    strategy_code,
                    "Compilation succeeded but no Python module was generated",
                )
                continue

            # Smoke-test stage removed (April 25). The TestLM stub in runner.py
            # diverged from the real _TensorizedLMBase API (e.g. _logits_tensor,
            # _token_indices_for_token), causing valid strategies to be marked
            # runtime-failed when they used helpers that the fastpath shims
            # vectorize. Real evaluation catches the same crash modes the smoke
            # test was meant to catch (the eval has its own per-example step
            # budget and any interface error surfaces there too). Skipping
            # straight from compile to eval.
            print("\n[3/4] Skipping runtime smoke test (removed; eval catches the same failures).")

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
            )
            attempt.eval_result = eval_result

            if not eval_result.success:
                print(f"  ✗ Evaluation failed: {eval_result.error}")
                attempt.failed_at = FailureStage.EVALUATION
                attempt.error_summary = eval_result.error or "Evaluation failed"
                attempts.append(attempt)

                print("  Refining based on evaluation error...")
                eval_history = self._get_evaluation_history_summary(attempts)
                working_hypothesis = self._get_working_hypothesis_state(attempts, attempt)
                evaluation_feedback = eval_result.get_feedback_summary()
                strategy_code = self.generator.refine_after_evaluation_failure(
                    strategy_code, evaluation_feedback, eval_history, working_hypothesis
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
                strategy_code = self.generator.refine_after_evaluation_failure(
                    strategy_code, threshold_feedback, eval_history, working_hypothesis
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
            report_path = self._save_failure_report(attempts, task_description, run_dir)

        error = SynthesisExhaustionError(
            f"Synthesis failed after {self.max_iterations} attempts", attempts, report_path
        )

        print(error.get_failure_summary())
        raise error

    def _save_failure_report(self, attempts: list[SynthesisAttempt], task_description: str, run_dir: Path) -> Path:
        """Save a detailed failure report to disk."""
        report_path = run_dir / "failure_report.json"

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

        # Create 'latest' symlink in the runs directory even on failure
        try:
            latest_link = run_dir.parent / "latest"
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
        evaluation_result: EvaluationResult,
    ) -> None:
        """Save a success report and the final strategy."""
        # Save the Dafny source
        dafny_path = run_dir / f"{output_name}.dfy"
        with open(dafny_path, "w") as f:
            f.write(full_code)

        # NOTE: We do NOT overwrite dafny/GeneratedCSD.dfy here because it contains
        # the template markers (QWEN_INSERT_STRATEGY_HERE) needed for future runs.
        # The final Dafny code is saved in the run directory instead.

        rationale_extracted = extract_rationale(strategy_code)

        # Save a report
        report_path = run_dir / "success_report.json"
        report = {
            "strategy_code": strategy_code,
            "tool_choice_rationale": rationale_extracted.rationale,
            "dafny_file": str(dafny_path),
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

        # Create 'latest' symlink in the runs directory
        try:
            latest_link = run_dir.parent / "latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(run_dir.name, target_is_directory=True)
            print(f"Latest run link updated: {latest_link}")
        except Exception as e:
            print(f"Warning: Could not create 'latest' symlink: {e}")

    def _analyze_failure_patterns(self, attempts: list[SynthesisAttempt]) -> dict:
        """Analyze common failure patterns across attempts."""
        patterns = {
            "verification_failures": 0,
            "compilation_failures": 0,
            "runtime_failures": 0,
            "common_errors": [],
        }

        error_counts: dict[str, int] = {}

        for attempt in attempts:
            if attempt.failed_at == FailureStage.VERIFICATION:
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
