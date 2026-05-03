"""
Main synthesis pipeline with feedback-based refinement.

Orchestrates the generate -> verify -> compile -> run loop with
iterative refinement based on errors.
"""

import json
import secrets
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

    PROOF_CRITIQUE = "proof_critique"
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

    # Which attempt this one was refined from (None for the initial generation)
    refined_from_attempt: Optional[int] = None

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
        min_format_rate: float = 0.0,
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
            min_format_rate: Minimum format/delimiter threshold for evaluation
            min_syntax_rate: Minimum syntax validity rate threshold
            require_delimiters: Whether evaluated outputs must contain << >> spans
            eval_sample_size: Number of examples to evaluate on
            eval_max_seconds_per_example: Optional runtime budget per example in seconds
        """
        self.evaluator = evaluator
        self.generator = generator or StrategyGenerator()
        self.verifier = verifier
        self.compiler = compiler or DafnyCompiler()
        self.runner = runner  # Will be created per-task in synthesize()
        self.max_iterations = max_iterations
        self.output_dir = output_dir or self.DEFAULT_OUTPUT_DIR
        self.save_reports = save_reports

        # Evaluation thresholds
        self.min_accuracy = min_accuracy
        self.min_format_rate = min_format_rate
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
    def _summarize_strategy_structure(strategy_code: str) -> str:
        """Build a concise structural summary of a strategy from its helper calls."""
        import re as _re
        helpers_used = []
        helper_names = [
            "UnconstrainedStep", "UnconstrainedChunk",
            "ConstrainedStep", "AdaptiveConstrainedStep",
            "PenalizedConstrainedStep", "GroupBoostedConstrainedStep",
            "OpenConstrainedSpan", "CloseConstrainedSpan",
            "RollbackToBoundary", "RollbackConstrainedSpan", "RollbackToValidPrefix",
            "CraneGeneration", "PureConstrainedGeneration",
            "TryUnconstrainedThenConstrained",
            "IntersectTokenSets", "FlattenTokenGroups",
        ]
        for name in helper_names:
            if _re.search(r'\b' + name + r'\b', strategy_code):
                helpers_used.append(name)

        unconstrained_mode = "none"
        if "UnconstrainedChunk" in helpers_used:
            unconstrained_mode = "UnconstrainedChunk (batched)"
        elif "UnconstrainedStep" in helpers_used:
            unconstrained_mode = "UnconstrainedStep (token-by-token)"

        constrained_mode = "none"
        for h in ["AdaptiveConstrainedStep", "GroupBoostedConstrainedStep",
                   "PenalizedConstrainedStep", "ConstrainedStep"]:
            if h in helpers_used:
                constrained_mode = h
                break

        has_open = "OpenConstrainedSpan" in helpers_used
        has_close = "CloseConstrainedSpan" in helpers_used
        has_rollback = any(h.startswith("Rollback") for h in helpers_used)
        has_crane = "CraneGeneration" in helpers_used

        checks_delimiter = bool(_re.search(r'==.*"<<"', strategy_code) or
                                _re.search(r'Contains\(.*"<<"', strategy_code))

        parts = []
        if has_crane:
            parts.append("CraneGeneration (one-shot)")
        else:
            parts.append(f"unconstrained={unconstrained_mode}")
            if checks_delimiter:
                parts.append("delimiter-triggered << entry")
            elif has_open:
                parts.append("forced OpenConstrainedSpan")
            parts.append(f"constrained={constrained_mode}")
            if has_close:
                parts.append("explicit close on IsCompletePrefix")
            if has_rollback:
                rollbacks = [h for h in helpers_used if h.startswith("Rollback")]
                parts.append(f"rollback={'+'.join(rollbacks)}")
            if "IntersectTokenSets" in helpers_used or "FlattenTokenGroups" in helpers_used:
                parts.append("uses token groups")

        return "; ".join(parts)

    @staticmethod
    def _compute_strategy_diff(old_code: str, new_code: str) -> str:
        """Compute a concise unified diff between two strategy codes, excluding
        comment-only and blank-line changes."""
        import difflib
        old_lines = [l for l in old_code.splitlines() if l.strip() and not l.strip().startswith("//")]
        new_lines = [l for l in new_code.splitlines() if l.strip() and not l.strip().startswith("//")]
        diff = list(difflib.unified_diff(old_lines, new_lines, lineterm="", n=1))
        if not diff:
            return "(no code changes — only comments/rationale changed)"
        # Skip the --- / +++ header lines
        return "\n".join(diff[2:])

    def _get_evaluation_history_summary(self, attempts: list[SynthesisAttempt]) -> str:
        """Chronological summary of all attempts — both evaluated and
        verification-failed — so the model sees the full timeline including
        approaches that were tried but couldn't pass Dafny."""
        relevant = [
            a for a in attempts
            if a.eval_result is not None or a.failed_at == FailureStage.VERIFICATION
        ]
        if not relevant:
            return ""

        eval_by_num = {a.attempt_number: a for a in attempts if a.eval_result is not None}

        lines = []
        best_acc = 0.0
        best_attempt = None
        for a in relevant:
            if a.failed_at == FailureStage.VERIFICATION:
                rationale = extract_rationale(a.strategy_code)
                rationale_text = rationale.rationale.strip() if rationale.has_markers else "(no rationale)"
                error_short = (a.error_summary or "unknown error")[:200]
                lines.append(f"--- Attempt {a.attempt_number}: VERIFICATION FAILED ---")
                lines.append(f"Approach: {rationale_text}")
                lines.append(f"Error: {error_short}")
                lines.append("")
                continue

            r = a.eval_result
            acc = r.accuracy
            syntax = f"{r.syntax_rate:.1%}"
            n = r.num_examples

            base = eval_by_num.get(a.refined_from_attempt) if a.refined_from_attempt else None

            rationale = extract_rationale(a.strategy_code)
            rationale_text = rationale.rationale.strip() if rationale.has_markers else None

            if base is None:
                lines.append(f"--- Attempt {a.attempt_number}: accuracy={acc:.1%} syntax={syntax} (n={n}) ---")
                lines.append("```dafny")
                lines.append(a.strategy_code)
                lines.append("```")
            else:
                base_acc = base.eval_result.accuracy
                delta = acc - base_acc
                delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
                outcome = "IMPROVED" if delta > 0 else ("REGRESSED" if delta < 0 else "NO CHANGE")

                lines.append(f"--- Attempt {a.attempt_number}: accuracy={acc:.1%} syntax={syntax} ({outcome}, {delta_str} from attempt {base.attempt_number}) ---")
                if rationale_text:
                    lines.append(f"Approach: {rationale_text}")
                diff = self._compute_strategy_diff(base.strategy_code, a.strategy_code)
                lines.append(f"Changes from attempt {base.attempt_number}:")
                lines.append("```diff")
                lines.append(diff)
                lines.append("```")
                if outcome == "REGRESSED":
                    lines.append(f"^^^ WARNING: this change caused a regression. DO NOT repeat it.")

            if acc > best_acc:
                best_acc = acc
                best_attempt = a.attempt_number
            lines.append("")
        if best_attempt is not None:
            lines.append(f"*** Best so far: attempt {best_attempt} ({best_acc:.1%} accuracy)")

        regressed = []
        for a in relevant:
            if a.eval_result is None:
                continue
            base = eval_by_num.get(a.refined_from_attempt) if a.refined_from_attempt else None
            if base is None:
                continue
            delta = a.eval_result.accuracy - base.eval_result.accuracy
            if delta < 0:
                rationale = extract_rationale(a.strategy_code)
                summary = rationale.rationale.strip() if rationale.has_markers else "(no rationale)"
                summary_oneline = " ".join(summary.split())[:200]
                regressed.append(
                    f"  - Attempt {a.attempt_number} ({delta:+.1%}): {summary_oneline}"
                )
        if regressed:
            lines.append("")
            lines.append("Approaches that caused regressions — DO NOT retry these or similar ideas:")
            lines.extend(regressed)

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
            grammar_source = None
            if self.evaluator.dataset_name in ("spider", "gsm_symbolic", "smiles", "folio"):
                grammars_dir = Path(__file__).parent.parent / "grammars"
                grammar_map = {
                    "spider": "sql.lark",
                    "gsm_symbolic": "gsm.lark",
                    "smiles": "smiles.lark",
                    "folio": "folio.lark",
                }
                grammar_path = grammars_dir / grammar_map[self.evaluator.dataset_name]
                if grammar_path.exists():
                    grammar_source = str(grammar_path)
            runner = StrategyRunner(parser_mode="permissive", grammar_source=grammar_source)
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
        if self.verifier is None:
            self.verifier = DafnyVerifier()

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

        best_eval_accuracy = 0.0
        best_eval_strategy = None
        best_eval_attempt_num = None
        derived_from_attempt = None  # attempt number the current strategy was derived from

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
                refined_from_attempt=derived_from_attempt,
            )

            # Stage 0: Proof-sketch critique — DISABLED.
            # The critic was rejecting plausible candidates for stylistic
            # reasons ("helper postcondition not cited explicitly", etc.) that
            # Dafny itself will verify or refute deterministically. Dafny's
            # structured diagnostics (obligation_kind + contract_excerpt +
            # failing_text) are a strictly richer refinement signal than the
            # critic's prose. We now go straight to Dafny.
            print("\n[0/4] Critic disabled — going straight to Dafny verification")

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
                    if best_eval_strategy is not None:
                        print(f"  Stuck on same error for {consecutive_same + 1} attempts — falling back to best evaluated strategy ({best_eval_accuracy:.1%})...")
                        strategy_code = best_eval_strategy
                        derived_from_attempt = best_eval_attempt_num
                    else:
                        print(f"  Stuck on same error for {consecutive_same + 1} attempts — restarting with fresh generation...")
                        strategy_code = self.generator.generate_initial(task_description)
                        derived_from_attempt = None
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
                    if best_eval_strategy is not None:
                        print(
                            f"  {consecutive_verif_failures} consecutive verification failures "
                            f"since last restart — falling back to best evaluated strategy ({best_eval_accuracy:.1%})..."
                        )
                        strategy_code = best_eval_strategy
                        derived_from_attempt = best_eval_attempt_num
                    else:
                        print(
                            f"  {consecutive_verif_failures} consecutive verification failures "
                            f"since last restart — restarting with fresh generation..."
                        )
                        strategy_code = self.generator.generate_initial(task_description)
                        derived_from_attempt = None
                    last_restart_index = len(attempts)
                    continue

                # Refine based on verification error
                print("  Refining based on verification error...")
                behavioral_context = self._get_recent_behavioral_context(attempts[:-1])
                structured_feedback = verification_result.get_structured_feedback()
                error_history = self._get_verification_history_summary(attempts)
                strategy_code = self.generator.refine_after_verification_error(
                    strategy_code,
                    error_msg,
                    behavioral_context,
                    structured_feedback,
                    error_history,
                )
                derived_from_attempt = attempt_num
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
                derived_from_attempt = attempt_num
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
                derived_from_attempt = attempt_num
                continue

            if self.evaluator.dataset_name != "gsm_symbolic":
                print("\n[3/4] Testing runtime execution...")

                runtime_result = runner.run(compilation_result.main_module_path)
                attempt.runtime_result = runtime_result

                if not runtime_result.success:
                    print(f"  ✗ Runtime error: {runtime_result.error_type}: {runtime_result.error_message}")
                    attempt.failed_at = FailureStage.RUNTIME
                    attempt.error_summary = runtime_result.get_error_summary()
                    attempts.append(attempt)

                    print("  Refining based on runtime error...")
                    strategy_code = self.generator.refine_after_runtime_error(
                        strategy_code, runtime_result.get_error_summary()
                    )
                    derived_from_attempt = attempt_num
                    continue

                print(f"  ✓ Execution successful ({runtime_result.execution_time_ms:.1f}ms)")
                print(f"  Output length: {len(runtime_result.output or [])} tokens")
            else:
                print("\n[3/4] Skipping toy runtime check for GSM-Symbolic...")

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
                strategy_code = self.generator.refine_after_evaluation_failure(
                    strategy_code, eval_result.get_feedback_summary(), task_description=task_description
                )
                derived_from_attempt = attempt_num
                continue

            # Check if evaluation meets thresholds
            if not eval_result.meets_threshold(
                min_accuracy=self.min_accuracy,
                min_format_rate=self.min_format_rate,
                min_syntax_rate=self.min_syntax_rate,
                require_delimiters=self.require_delimiters,
                max_seconds_per_example=self.eval_max_seconds_per_example,
            ):
                print(f"  ✗ Evaluation below threshold:")
                print(f"    Accuracy: {eval_result.accuracy:.1%} (min: {self.min_accuracy:.1%})")
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
                    + (
                        f"  Max Runtime / Example: {self.eval_max_seconds_per_example:.2f}s\n"
                        if self.eval_max_seconds_per_example is not None
                        else ""
                    )
                    + "\n"
                    + eval_result.get_feedback_summary()
                )
                eval_history = self._get_evaluation_history_summary(attempts)
                if eval_history:
                    threshold_feedback += "\n\nPrior evaluation attempts:\n" + eval_history
                print(threshold_feedback)
                print("--- END FEEDBACK ---")

                if eval_result.accuracy > best_eval_accuracy:
                    best_eval_accuracy = eval_result.accuracy
                    best_eval_strategy = strategy_code
                    best_eval_attempt_num = attempt_num

                refine_from = best_eval_strategy if best_eval_strategy else strategy_code
                refine_from_accuracy = best_eval_accuracy if best_eval_strategy else eval_result.accuracy
                strategy_code = self.generator.refine_after_evaluation_failure(
                    refine_from, threshold_feedback, task_description=task_description,
                    best_strategy=best_eval_strategy or "",
                    best_accuracy=best_eval_accuracy,
                    current_accuracy=refine_from_accuracy,
                    min_accuracy=self.min_accuracy,
                )
                derived_from_attempt = best_eval_attempt_num if best_eval_strategy else attempt_num
                continue

            print(f"  ✓ Evaluation passed:")
            print(f"    Accuracy: {eval_result.accuracy:.1%}")
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
            if attempt.failed_at == FailureStage.PROOF_CRITIQUE:
                patterns.setdefault("proof_critique_failures", 0)
                patterns["proof_critique_failures"] += 1
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
