"""
Evaluation module for synthesis feedback loop.

Provides quick evaluation of synthesized CSD strategies on dataset samples
to enable feedback-driven refinement based on actual performance metrics.
"""

from __future__ import annotations

import json
import re
from collections import Counter

# SQL keyword set for failure-preview anonymization. Identifiers outside this
# set are replaced with <id>; numbers with <num>; string literals with <str>.
# Punctuation and the keywords themselves pass through verbatim. The result
# preserves structural shape (SELECT-FROM-WHERE etc.) without leaking
# synthesis-sample schema/identifier text.
_SQL_KEYWORDS = {
    "SELECT","FROM","WHERE","JOIN","INNER","LEFT","RIGHT","FULL","OUTER","CROSS","ON","AS",
    "GROUP","BY","HAVING","ORDER","ASC","DESC","LIMIT","OFFSET","UNION","INTERSECT","EXCEPT",
    "ALL","DISTINCT","AND","OR","NOT","IN","IS","NULL","LIKE","BETWEEN","EXISTS",
    "COUNT","SUM","AVG","MIN","MAX","CASE","WHEN","THEN","ELSE","END","CAST","INT","REAL","TEXT",
    "TRUE","FALSE",
}
import re as _re

def _anonymize_sql_preview(s: str) -> str:
    """Replace identifiers/numbers/strings with placeholders, keep keywords + punctuation."""
    if not s:
        return s
    out_parts = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            out_parts.append(ch)
            i += 1
            continue
        if ch == "'":
            # consume string literal
            j = s.find("'", i + 1)
            j = j + 1 if j >= 0 else len(s)
            out_parts.append("<str>")
            i = j
            continue
        if ch.isdigit() or (ch == "." and i + 1 < len(s) and s[i + 1].isdigit()):
            m = _re.match(r"\d+(?:\.\d+)?", s[i:])
            out_parts.append("<num>")
            i += m.end() if m else 1
            continue
        if ch.isalpha() or ch == "_":
            m = _re.match(r"[A-Za-z_][A-Za-z0-9_]*", s[i:])
            tok = m.group(0) if m else ch
            out_parts.append(tok if tok.upper() in _SQL_KEYWORDS else "<id>")
            i += m.end() if m else 1
            continue
        out_parts.append(ch)
        i += 1
    return "".join(out_parts)

import time
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class PerExampleTimeout(Exception):
    """Raised when a single evaluation example exceeds its runtime budget."""


class _PerExampleTimer:
    """Unix wall-clock timer for interrupting a single long-running example."""

    def __init__(self, seconds: Optional[float]):
        self.seconds = seconds
        self._old_handler = None
        self._old_timer = None

    def __enter__(self):
        if self.seconds is None:
            return self
        if self.seconds <= 0:
            raise PerExampleTimeout(f"Example exceeded {self.seconds:.2f}s runtime budget")

        def _raise_timeout(signum, frame):
            raise PerExampleTimeout(f"Example exceeded {self.seconds:.2f}s runtime budget")

        self._old_handler = signal.getsignal(signal.SIGALRM)
        self._old_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds is not None:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)
            if self._old_timer and self._old_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, self._old_timer[0], self._old_timer[1])
        return False


@dataclass
class EvaluationResult:
    """
    Result of evaluating a CSD strategy on a dataset sample.

    Contains metrics and sample outputs for feedback to the generator.
    """
    success: bool
    accuracy: float  # 0.0 to 1.0
    contains_delimiters: bool
    syntax_rate: float  # 0.0 to 1.0
    num_examples: int
    num_correct: int
    total_time_seconds: float
    max_sample_time_seconds: float = 0.0

    # Sample outputs for feedback (question, expected, actual, is_correct)
    sample_outputs: List[Dict[str, Any]] = field(default_factory=list)

    # Error information if evaluation failed
    error: Optional[str] = None

    def meets_threshold(
        self,
        min_accuracy: float = 0.0,
        min_syntax_rate: float = 0.0,
        require_delimiters: bool = True,
        max_seconds_per_example: Optional[float] = None,
    ) -> bool:
        """Check if aggregate metrics meet the specified thresholds."""
        if not self.sample_outputs:
            return False
        runtime_ok = True
        if max_seconds_per_example is not None:
            runtime_ok = self.max_sample_time_seconds <= max_seconds_per_example
        delimiter_ok = self.contains_delimiters or not require_delimiters
        return (
            runtime_ok
            and delimiter_ok
            and self.accuracy >= min_accuracy
            and self.syntax_rate >= min_syntax_rate
        )

    def get_feedback_summary(self) -> str:
        """Generate a summary for feedback to the generator."""
        lines = [
            f"Evaluation Results ({self.num_examples} examples):",
            f"  Accuracy: {self.accuracy:.1%} ({self.num_correct}/{self.num_examples})",
            f"  Contains << >>: {'yes' if self.contains_delimiters else 'no'}",
            f"  Syntax Rate: {self.syntax_rate:.1%}",
            f"  Total Time: {self.total_time_seconds:.2f}s",
            f"  Slowest Example Time: {self.max_sample_time_seconds:.2f}s",
        ]

        failure_modes = self._summarize_failure_modes()
        if failure_modes:
            lines.append("\nPrimary Failure Modes:")
            for mode, count, detail in failure_modes:
                lines.append(f"  - {mode}: {count} example(s) {detail}")

        output_run_summary = self._summarize_output_run()
        if output_run_summary:
            lines.append("\nOutput Run Summary:")
            lines.extend(f"  {metric}" for metric in output_run_summary)

        # Anonymized failure summary (April 25): we used to dump up to 3 specific
        # failed examples (question + expected + actual) into the feedback to
        # gpt-5.4. That content leaked synthesis-sample specifics into the
        # generator's context, biasing strategies to fit those exact examples
        # and producing 12-22 pp held-out drops. We now report only aggregate
        # statistics — failure_mode counts (already populated above), runtime
        # budget exceedances, and any unexpected exception types — without any
        # question text, expected SQL, or actual SQL strings.
        if self.sample_outputs:
            n_runtime_exceeded = sum(
                1 for s in self.sample_outputs if s.get("runtime_budget_exceeded")
            )
            extras = []
            if n_runtime_exceeded:
                extras.append(
                    f"  Time budget exceeded on {n_runtime_exceeded} example(s)."
                )
            if extras:
                lines.append("\nAggregate Failure Stats:")
                lines.extend(extras)

        diagnostic_metrics = self._summarize_diagnostic_metrics()
        if diagnostic_metrics:
            lines.append("\nDiagnostic Error Decomposition:")
            lines.extend(f"  {metric}" for metric in diagnostic_metrics)

        structural_metrics = self._summarize_structural_metrics()
        if structural_metrics:
            lines.append("\nStructural Generation Metrics:")
            lines.extend(f"  {metric}" for metric in structural_metrics)

        return "\n".join(lines)

    @staticmethod
    def _format_trace_event(event: Dict[str, Any]) -> str:
        helper = event.get("helper", "unknown")
        detail = event.get("detail") or ""
        before = event.get("cost_before")
        after = event.get("cost_after")
        cost_part = ""
        if before is not None or after is not None:
            cost_part = f" [cost {before}->{after}]"
        return f"{helper}: {detail}{cost_part}".strip()

    def get_behavioral_context_summary(self, max_examples: int = 1, max_trace_events: int = 12) -> str:
        traced_examples = [s for s in self.sample_outputs if s.get("helper_trace")]
        if not traced_examples:
            return ""

        lines = ["Recent evaluated behavior from the most recent compiled/evaluated attempt:"]
        for idx, sample in enumerate(traced_examples[:max_examples]):
            trace = sample.get("helper_trace") or []
            counts = Counter(event.get("helper", "unknown") for event in trace)
            lines.append(f"Example {idx + 1}:")
            lines.append(
                f"  Token count: {sample.get('token_count', 'N/A')} | "
                f"Contains << >>: {'yes' if sample.get('contains_delimiters') else 'no'} | "
                f"Syntax rate: {sample.get('syntax_rate', 0.0):.1%}"
            )
            if counts:
                counts_summary = ", ".join(
                    f"{name}={count}" for name, count in counts.most_common(8)
                )
                lines.append(f"  Helper call counts: {counts_summary}")
            tail = trace[-max_trace_events:]
            if tail:
                lines.append("  Helper trace tail:")
                for event in tail:
                    lines.append(f"    - {self._format_trace_event(event)}")

        return "\n".join(lines)

    def _summarize_failure_modes(self) -> List[Tuple[str, int, str]]:
        """Classify the most common observable evaluation failure patterns."""
        counters: Dict[str, int] = {}
        details: Dict[str, str] = {}

        for sample in self.sample_outputs:
            error = sample.get("error")
            full_output = sample.get("full_output") or ""
            actual = sample.get("actual") or ""
            contains_delimiters = sample.get("contains_delimiters", False)
            uses_hidden_chunks = sample.get("uses_hidden_chunks", False)
            visible_delimiters = sample.get("visible_delimiters", contains_delimiters)
            used_constrained_chunk = sample.get("used_constrained_chunk", contains_delimiters)
            syntax_rate = float(sample.get("syntax_rate", 0.0))
            matched = False

            if sample.get("runtime_budget_exceeded"):
                key = "too_slow"
                counters[key] = counters.get(key, 0) + 1
                if key not in details:
                    details[key] = "(generation exceeded the per-example runtime budget)"
                matched = True
                if error:
                    continue

            if error:
                key = "runtime_or_generation_error"
                counters[key] = counters.get(key, 0) + 1
                if key not in details:
                    details[key] = f"(first error: {str(error)[:120]})"
                continue

            if not uses_hidden_chunks and "<<" in full_output and ">>" not in full_output:
                key = "unterminated_constrained_segment"
                counters[key] = counters.get(key, 0) + 1
                if key not in details:
                    details[key] = "(opened `<<` but did not close `>>`)"
                matched = True

            if uses_hidden_chunks:
                if not used_constrained_chunk:
                    key = "missing_constrained_chunk"
                    counters[key] = counters.get(key, 0) + 1
                    if key not in details:
                        details[key] = "(no internal parser-governed chunk was used)"
                    matched = True
            elif not contains_delimiters and "<<" not in full_output:
                key = "missing_constrained_segment"
                counters[key] = counters.get(key, 0) + 1
                if key not in details:
                    details[key] = "(no `<< >>` segment detected)"
                matched = True

            if not uses_hidden_chunks and ">>" in full_output and "<<" not in full_output:
                key = "premature_or_unmatched_closure"
                counters[key] = counters.get(key, 0) + 1
                if key not in details:
                    details[key] = "(generated `>>` without a matching opening `<<`)"
                matched = True

            if not uses_hidden_chunks and "<<" in full_output and ">>" in full_output and syntax_rate == 0.0:
                key = "malformed_constrained_content"
                counters[key] = counters.get(key, 0) + 1
                if key not in details:
                    details[key] = "(delimiters present, but constrained content failed syntax checks)"
                matched = True

            if not uses_hidden_chunks and self._looks_like_early_constrained_entry(full_output):
                key = "entered_constrained_mode_too_early"
                counters[key] = counters.get(key, 0) + 1
                if key not in details:
                    details[key] = "(output entered `<<` almost immediately after the prompt continuation began)"
                matched = True

            if self._has_repetition_loop(full_output):
                key = "repetition_loop"
                counters[key] = counters.get(key, 0) + 1
                if key not in details:
                    details[key] = "(local token pattern repeated in output)"
                matched = True

            if not actual:
                key = "answer_extraction_failed"
                counters[key] = counters.get(key, 0) + 1
                if key not in details:
                    details[key] = "(no extractable final answer)"
                matched = True

            if not matched and not sample.get("is_correct", False):
                key = "other_observed_failure"
                counters[key] = counters.get(key, 0) + 1
                if key not in details:
                    raw_preview = (full_output or "").replace("\n", " ")[:120] or str(actual)[:120]
                    anon = _anonymize_sql_preview(raw_preview) if raw_preview else "no preview"
                    details[key] = f"(uncategorized failure preview, anonymized: {anon})"

        ranked = sorted(counters.items(), key=lambda item: (-item[1], item[0]))
        return [(mode, count, details.get(mode, "")) for mode, count in ranked[:4]]

    @staticmethod
    def _mean(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _median(values: List[float]) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2

    @staticmethod
    def _visible_span_shape(output: str) -> Dict[str, Any]:
        """Return delimiter/span shape statistics without preserving span text."""
        opens = output.count("<<")
        closes = output.count(">>")
        spans = re.findall(r"<<\s*(.*?)\s*>>", output, flags=re.DOTALL)

        active_opens = 0
        unmatched_closes = 0
        i = 0
        while i < len(output):
            if output.startswith("<<", i):
                active_opens += 1
                i += 2
            elif output.startswith(">>", i):
                if active_opens > 0:
                    active_opens -= 1
                else:
                    unmatched_closes += 1
                i += 2
            else:
                i += 1

        first_open_tokens: Optional[int] = None
        first_open = output.find("<<")
        if first_open >= 0:
            first_open_tokens = len(output[:first_open].split())

        span_lengths = [len(span.strip().split()) for span in spans if span.strip()]
        complete_spans = len(spans)
        return {
            "opens": opens,
            "closes": closes,
            "complete_spans": complete_spans,
            "span_lengths": span_lengths,
            "first_open_tokens": first_open_tokens,
            "unterminated": active_opens > 0 or opens > complete_spans,
            "unmatched_close": unmatched_closes > 0 or closes > complete_spans,
            "balanced_with_span": complete_spans > 0 and opens == closes == complete_spans,
        }

    _CONSTRAINED_HELPERS = {
        "OpenConstrainedSpan",
        "EnterObservedConstrainedSpan",
        "CloseConstrainedSpan",
        "ConstrainedStep",
        "AdaptiveConstrainedStep",
        "GroupBoostedConstrainedStep",
        "AppendConstrainedToken",
        "ConstrainedSymbol",
        "ConstrainedSymbolInGenerated",
        "RollbackConstrainedSuffix",
    }

    _UNCONSTRAINED_HELPERS = {"UnconstrainedStep", "UnconstrainedChunk"}

    @classmethod
    def _sample_has_valid_span_or_chunk(cls, sample: Dict[str, Any]) -> bool:
        return (
            int(sample.get("num_valid_visible_spans", 0) or 0) > 0
            or bool(sample.get("used_constrained_chunk") and sample.get("uses_hidden_chunks"))
        )

    @classmethod
    def _sample_has_constrained_activity(cls, sample: Dict[str, Any]) -> bool:
        return any(
            event.get("helper", "unknown") in cls._CONSTRAINED_HELPERS
            for event in sample.get("helper_trace") or []
        )

    def get_diagnostic_counts(self) -> Dict[str, int]:
        """Return neutral per-example diagnostic buckets for prompt deltas."""
        samples = self.sample_outputs
        n = len(samples)
        syntax_valid = [s for s in samples if s.get("is_syntax_valid")]
        syntax_invalid = [s for s in samples if not s.get("is_syntax_valid")]
        valid_span = [s for s in samples if self._sample_has_valid_span_or_chunk(s)]
        constrained_activity = [s for s in samples if self._sample_has_constrained_activity(s)]
        visible_span_no_activity = [
            s
            for s in samples
            if int(s.get("num_visible_spans", 0) or 0) > 0
            and not self._sample_has_constrained_activity(s)
        ]

        return {
            "examples": n,
            "syntax_valid_correct": sum(1 for s in syntax_valid if s.get("is_correct")),
            "syntax_valid_wrong": sum(1 for s in syntax_valid if not s.get("is_correct")),
            "syntax_invalid_correct": sum(1 for s in syntax_invalid if s.get("is_correct")),
            "syntax_invalid_wrong": sum(1 for s in syntax_invalid if not s.get("is_correct")),
            "no_complete_span_wrong": sum(
                1
                for s in samples
                if int(s.get("num_visible_spans", 0) or 0) == 0
                and not s.get("is_correct")
            ),
            "answer_from_last_visible_span": sum(
                1 for s in samples if s.get("answer_source") == "last_visible_span"
            ),
            "answer_from_text_fallback": sum(
                1 for s in samples if s.get("answer_source") == "text_fallback"
            ),
            "answer_from_hidden_or_task_extractor": sum(
                1 for s in samples if s.get("answer_source") == "hidden_or_task_extractor"
            ),
            "no_extracted_answer": sum(
                1 for s in samples if not s.get("has_extracted_answer", False)
            ),
            "examples_with_final_answer_span": sum(
                1 for s in samples if s.get("answer_source") == "last_visible_span"
            ),
            "examples_with_valid_nonfinal_spans_only": sum(
                1
                for s in valid_span
                if s.get("answer_source") != "last_visible_span"
            ),
            "examples_with_no_valid_span": n - len(valid_span),
            "examples_with_constrained_activity": len(constrained_activity),
            "examples_without_constrained_activity": n - len(constrained_activity),
            "correct_with_constrained_activity": sum(
                1 for s in constrained_activity if s.get("is_correct")
            ),
            "wrong_with_constrained_activity": sum(
                1 for s in constrained_activity if not s.get("is_correct")
            ),
            "correct_without_constrained_activity": sum(
                1
                for s in samples
                if not self._sample_has_constrained_activity(s) and s.get("is_correct")
            ),
            "wrong_without_constrained_activity": sum(
                1
                for s in samples
                if not self._sample_has_constrained_activity(s) and not s.get("is_correct")
            ),
            "visible_span_without_constrained_activity": len(visible_span_no_activity),
        }

    @staticmethod
    def _is_helper_api_error(error: str) -> bool:
        """Heuristic bucket for actual helper/API exceptions, not behavior metrics."""
        lowered = error.lower()
        helper_terms = (
            "helper",
            "csdhelpers",
            "constrainedstep",
            "adaptiveconstrainedstep",
            "groupboostedconstrainedstep",
            "constrainedsymbol",
            "constrainedsymbolingenerated",
            "appendconstrainedtoken",
            "openconstrainedspan",
            "enterobservedconstrainedspan",
            "closeconstrainedspan",
            "rollbackconstrainedsuffix",
        )
        api_error_terms = (
            "attributeerror",
            "nameerror",
            "typeerror",
            "missing",
            "undefined",
            "not defined",
            "has no attribute",
            "takes",
            "argument",
        )
        return any(term in lowered for term in helper_terms) and any(
            term in lowered for term in api_error_terms
        )

    def _summarize_output_run(self) -> List[str]:
        """Summarize factual output behavior without causal labels."""
        if not self.sample_outputs:
            return []

        n = len(self.sample_outputs)
        errors = [str(sample.get("error") or "") for sample in self.sample_outputs if sample.get("error")]
        timeouts = sum(1 for sample in self.sample_outputs if sample.get("runtime_budget_exceeded"))
        nonempty_outputs = sum(
            1
            for sample in self.sample_outputs
            if int(sample.get("token_count", 0) or 0) > 0
            or bool((sample.get("scored_output") or sample.get("full_output") or "").strip())
        )

        lines = [
            "verification: passed",
            "compilation: passed",
            f"generated-token outputs: {nonempty_outputs}/{n} nonempty",
        ]
        if not errors and timeouts == 0:
            lines.append("All evaluated examples completed and returned generated output records.")
            return lines

        if errors:
            lines.append(f"examples without completed output records: {len(errors)}/{n}")
        if timeouts:
            lines.append(f"per-example time budget exceeded: {timeouts}/{n}")
        return lines

    def _summarize_diagnostic_metrics(self) -> List[str]:
        """Summarize where failures enter without exposing example content."""
        if not self.sample_outputs:
            return []

        counts = self.get_diagnostic_counts()
        n = counts["examples"]
        return [
            (
                "Correctness by syntax bucket: "
                f"syntax_valid_correct {counts['syntax_valid_correct']}/{n}, "
                f"syntax_valid_wrong {counts['syntax_valid_wrong']}/{n}, "
                f"syntax_invalid_correct {counts['syntax_invalid_correct']}/{n}, "
                f"syntax_invalid_wrong {counts['syntax_invalid_wrong']}/{n}"
            ),
            f"No-complete-span wrong answers: {counts['no_complete_span_wrong']}/{n}",
            (
                "Answer extraction source: "
                f"last_visible_span {counts['answer_from_last_visible_span']}/{n}, "
                f"text_fallback {counts['answer_from_text_fallback']}/{n}, "
                f"hidden_or_task_extractor {counts['answer_from_hidden_or_task_extractor']}/{n}, "
                f"none {counts['no_extracted_answer']}/{n}"
            ),
            (
                "Span usefulness: "
                f"final_answer_span {counts['examples_with_final_answer_span']}/{n}, "
                f"valid_nonfinal_spans_only {counts['examples_with_valid_nonfinal_spans_only']}/{n}, "
                f"no_valid_span {counts['examples_with_no_valid_span']}/{n}"
            ),
            (
                "Constrained intervention activity: "
                f"examples_with_activity {counts['examples_with_constrained_activity']}/{n}, "
                f"examples_without_activity {counts['examples_without_constrained_activity']}/{n}, "
                f"visible_span_without_activity {counts['visible_span_without_constrained_activity']}/{n}"
            ),
            (
                "Correctness conditioned on constrained activity: "
                f"correct_with_activity {counts['correct_with_constrained_activity']}/{n}, "
                f"wrong_with_activity {counts['wrong_with_constrained_activity']}/{n}, "
                f"correct_without_activity {counts['correct_without_constrained_activity']}/{n}, "
                f"wrong_without_activity {counts['wrong_without_constrained_activity']}/{n}"
            ),
        ]

    def _summarize_structural_metrics(self) -> List[str]:
        """Summarize neutral span/search behavior for refinement feedback."""
        if not self.sample_outputs:
            return []

        n = len(self.sample_outputs)
        shapes = [
            self._visible_span_shape(s.get("scored_output") or s.get("full_output") or "")
            for s in self.sample_outputs
        ]
        helper_counts: Counter[str] = Counter()

        constrained_calls = 0
        unconstrained_calls = 0
        for sample in self.sample_outputs:
            for event in sample.get("helper_trace") or []:
                helper = event.get("helper", "unknown")
                helper_counts[helper] += 1
                constrained_calls += int(helper in EvaluationResult._CONSTRAINED_HELPERS)
                unconstrained_calls += int(helper in EvaluationResult._UNCONSTRAINED_HELPERS)

        total_opens = sum(int(shape["opens"]) for shape in shapes)
        total_complete_spans = sum(int(shape["complete_spans"]) for shape in shapes)
        span_completion_rate = total_complete_spans / total_opens if total_opens else None

        first_open_positions = [
            float(shape["first_open_tokens"])
            for shape in shapes
            if shape["first_open_tokens"] is not None
        ]
        visible_span_lengths = [
            float(length)
            for shape in shapes
            for length in shape["span_lengths"]
        ]
        valid_span_lengths = [
            float(length)
            for sample in self.sample_outputs
            for length in sample.get("valid_visible_span_token_lengths", [])
        ]
        output_tokens = [
            float(sample.get("token_count", 0) or 0)
            for sample in self.sample_outputs
        ]
        runtimes = [
            float(sample.get("time_seconds", 0.0) or 0.0)
            for sample in self.sample_outputs
        ]

        examples_with_visible_open = sum(1 for shape in shapes if shape["opens"] > 0)
        examples_with_complete_span = sum(1 for shape in shapes if shape["complete_spans"] > 0)
        examples_with_balanced_span = sum(1 for shape in shapes if shape["balanced_with_span"])
        examples_with_unterminated = sum(1 for shape in shapes if shape["unterminated"])
        examples_with_unmatched_close = sum(1 for shape in shapes if shape["unmatched_close"])
        examples_without_complete_span = n - examples_with_complete_span
        examples_with_valid_span = sum(
            1 for sample in self.sample_outputs
            if self._sample_has_valid_span_or_chunk(sample)
        )
        examples_with_parser_span_failure = sum(
            1 for sample in self.sample_outputs
            if int(sample.get("num_visible_spans", 0) or 0)
            > int(sample.get("num_valid_visible_spans", 0) or 0)
        )
        examples_with_valid_span_wrong = sum(
            1 for sample in self.sample_outputs
            if (
                int(sample.get("num_valid_visible_spans", 0) or 0) > 0
                or bool(sample.get("used_constrained_chunk") and sample.get("uses_hidden_chunks"))
            )
            and not sample.get("is_correct")
        )
        examples_format_valid_wrong = sum(
            1 for sample in self.sample_outputs
            if sample.get("is_syntax_valid") and not sample.get("is_correct")
        )
        examples_without_extracted_answer = sum(
            1 for sample in self.sample_outputs
            if not sample.get("has_extracted_answer", False)
        )
        examples_with_tiny_valid_spans = sum(
            1 for sample in self.sample_outputs
            if sample.get("valid_visible_span_token_lengths")
            and max(sample.get("valid_visible_span_token_lengths")) <= 2
        )
        examples_with_long_visible_span = sum(
            1 for shape in shapes if any(length > 64 for length in shape["span_lengths"])
        )
        examples_hitting_max_steps = sum(
            1 for sample in self.sample_outputs if sample.get("hit_max_steps")
        )

        mean_spans = self._mean([float(shape["complete_spans"]) for shape in shapes])
        mean_first_open = self._mean(first_open_positions)
        median_first_open = self._median(first_open_positions)
        mean_visible_span_len = self._mean(visible_span_lengths)
        median_visible_span_len = self._median(visible_span_lengths)
        mean_valid_span_len = self._mean(valid_span_lengths)
        mean_output_tokens = self._mean(output_tokens)
        median_output_tokens = self._median(output_tokens)
        max_output_tokens = max(output_tokens, default=0.0)
        total_runtime = sum(runtimes)
        total_tokens = sum(output_tokens)
        time_per_token = total_runtime / total_tokens if total_tokens else None
        constrained_fraction = (
            constrained_calls / (constrained_calls + unconstrained_calls)
            if constrained_calls + unconstrained_calls
            else None
        )

        def fmt_optional(value: Optional[float], suffix: str = "") -> str:
            return "n/a" if value is None else f"{value:.2f}{suffix}"

        lines = [
            f"Examples with visible `<<`: {examples_with_visible_open}/{n}",
            f"Examples with complete visible spans: {examples_with_complete_span}/{n}",
            f"Examples without complete visible spans: {examples_without_complete_span}/{n}",
            f"Examples with balanced visible spans: {examples_with_balanced_span}/{n}",
            f"Examples with unterminated visible spans: {examples_with_unterminated}/{n}",
            f"Examples with unmatched visible close: {examples_with_unmatched_close}/{n}",
            f"Visible span completion rate: {fmt_optional(span_completion_rate)}",
            f"Avg complete visible spans/example: {fmt_optional(mean_spans)}",
            (
                "Tokens before first visible open: "
                f"avg {fmt_optional(mean_first_open)}, median {fmt_optional(median_first_open)}"
            ),
            (
                "Visible span token length: "
                f"avg {fmt_optional(mean_visible_span_len)}, median {fmt_optional(median_visible_span_len)}"
            ),
            f"Valid visible span token length avg: {fmt_optional(mean_valid_span_len)}",
            f"Examples with at least one valid span/chunk: {examples_with_valid_span}/{n}",
            f"Examples with visible parser span failure: {examples_with_parser_span_failure}/{n}",
            f"Examples with valid span/chunk but wrong answer: {examples_with_valid_span_wrong}/{n}",
            f"Examples syntax-valid but wrong answer: {examples_format_valid_wrong}/{n}",
            f"Examples without extracted answer: {examples_without_extracted_answer}/{n}",
            f"Examples with only tiny valid visible spans: {examples_with_tiny_valid_spans}/{n}",
            f"Examples with long visible span (>64 tokens): {examples_with_long_visible_span}/{n}",
            f"Examples hitting max steps: {examples_hitting_max_steps}/{n}",
            (
                "Generated tokens/example: "
                f"avg {fmt_optional(mean_output_tokens)}, median {fmt_optional(median_output_tokens)}, "
                f"max {max_output_tokens:.0f}"
            ),
            f"Runtime per generated token: {fmt_optional(time_per_token, 's')}",
        ]
        if helper_counts:
            top_helpers = ", ".join(
                f"{name}={count}" for name, count in helper_counts.most_common(8)
            )
            lines.append(f"Top helper calls: {top_helpers}")
            lines.append(
                "Constrained helper call fraction: "
                f"{fmt_optional(constrained_fraction)}"
            )

        return lines

    def _looks_like_early_constrained_entry(self, output: str) -> bool:
        """Heuristic: flags outputs that open a constrained segment almost immediately."""
        if "<<" not in output:
            return False
        prefix = output.split("<<", 1)[0].strip()
        if not prefix:
            return True
        return len(prefix.split()) <= 4

    def _has_repetition_loop(self, output: str) -> bool:
        """Detect short repeated local patterns that often indicate degenerate decoding."""
        tokens = output.split()
        if len(tokens) < 6:
            return False
        for width in (1, 2, 3):
            for start in range(0, len(tokens) - 3 * width + 1):
                chunk = tokens[start:start + width]
                if (
                    tokens[start + width:start + 2 * width] == chunk
                    and tokens[start + 2 * width:start + 3 * width] == chunk
                ):
                    return True
        return False

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "accuracy": self.accuracy,
            "contains_delimiters": self.contains_delimiters,
            "syntax_rate": self.syntax_rate,
            "num_examples": self.num_examples,
            "num_correct": self.num_correct,
            "total_time_seconds": self.total_time_seconds,
            "max_sample_time_seconds": self.max_sample_time_seconds,
            "error": self.error,
            "sample_outputs": self.sample_outputs,
        }


class Evaluator:
    """
    Evaluates synthesized CSD strategies on dataset samples.

    Supports both GSM-Symbolic and FOLIO datasets with their respective
    evaluation metrics and syntax validation.
    """

    def __init__(
        self,
        dataset_name: str = "gsm_symbolic",
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        backend: str = "huggingface",
        device: str = "cuda",
        sample_size: int = 10,
        max_steps: int = 600,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        vllm_tensor_parallel_size: Optional[int] = None,
        vllm_pipeline_parallel_size: int = 1,
        vllm_gpu_memory_utilization: float = 0.8,
        vllm_max_model_len: int = 4096,
        vllm_enforce_eager: bool = True,
        sample_seed: Optional[int] = None,
        max_seconds_per_example: Optional[float] = None,
        step_token_budget: int = 1,
        gsm_source_dir: str | Path | None = None,
        gsm_examples_file: str | Path | None = None,
        gsm_split_file: str | Path | None = None,
        gsm_split_name: str = "train",
        smiles_classes: Optional[List[str]] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            dataset_name: Dataset to evaluate on ("gsm_symbolic" or "folio")
            model_name: HuggingFace model for generation
            backend: Runtime LM backend ("huggingface" or "vllm")
            device: Device to run on ("cuda", "mps", "cpu")
            sample_size: Number of examples to evaluate on
            max_steps: Maximum generation steps per example
            load_in_4bit: Whether to load model in 4-bit quantization
            load_in_8bit: Whether to load model in 8-bit quantization
            vllm_tensor_parallel_size: Explicit tensor parallel size for vLLM
            vllm_pipeline_parallel_size: Explicit pipeline parallel size for vLLM
            vllm_gpu_memory_utilization: GPU memory fraction reserved by vLLM
            vllm_max_model_len: Max context length passed to vLLM
            vllm_enforce_eager: Disable cudagraph/compile in vLLM for stability
            sample_seed: Optional RNG seed for reproducible dataset sampling
            max_seconds_per_example: Optional runtime budget per example in seconds
            gsm_examples_file: Optional JSONL file of exact GSM examples to use.
            gsm_split_file: Optional JSON manifest with train_indices/eval_indices for GSM.
            gsm_split_name: Which split from gsm_split_file to use ("train" or "eval").
        """
        if backend not in {"huggingface", "vllm"}:
            raise NotImplementedError(
                "Evaluation backend must be 'huggingface' or 'vllm'. "
                "Hosted API backends are not supported by the current CSD runtime because "
                "the generated Dafny strategy needs direct token logits, masking, and tokenizer access."
            )

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.backend = backend
        self.device = device
        self.sample_size = sample_size
        self.max_steps = max_steps
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.vllm_pipeline_parallel_size = vllm_pipeline_parallel_size
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.vllm_max_model_len = vllm_max_model_len
        self.vllm_enforce_eager = vllm_enforce_eager
        self.sample_seed = sample_seed
        self.max_seconds_per_example = max_seconds_per_example
        self.step_token_budget = step_token_budget
        self.gsm_source_dir = gsm_source_dir
        self.gsm_examples_file = Path(gsm_examples_file) if gsm_examples_file is not None else None
        self.gsm_split_file = Path(gsm_split_file) if gsm_split_file is not None else None
        self.gsm_split_name = gsm_split_name
        self.smiles_classes = smiles_classes

        # Lazy-loaded components
        self._dataset = None
        self._env = None
        self._env_cache_key: Optional[tuple[Any, ...]] = None
        self._grammar_file = None
        self._base_grammar_text: Optional[str] = None
        self._dynamic_parser_factory_cache: Dict[Tuple[str, ...], Any] = {}
        self._syntax_parser_cache: Dict[Tuple[str, ...], Any] = {}

    def _load_gsm_split_indices(self) -> Optional[List[int]]:
        """Load explicit GSM example indices from a train/eval split manifest."""
        if self.gsm_split_file is None:
            return None

        manifest = json.loads(self.gsm_split_file.read_text())
        key = f"{self.gsm_split_name}_indices"
        if key not in manifest:
            available = sorted(k for k in manifest.keys() if k.endswith("_indices"))
            raise ValueError(
                f"Split file {self.gsm_split_file} does not contain {key}. "
                f"Available index fields: {available}"
            )

        indices = manifest[key]
        if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
            raise ValueError(f"{key} in {self.gsm_split_file} must be a list of integers")
        return indices

    def _load_gsm_examples_file(self) -> List[dict]:
        """Load an exact GSM example list from JSONL for reproducible ablations."""
        if self.gsm_examples_file is None:
            return []
        if not self.gsm_examples_file.exists():
            raise FileNotFoundError(f"GSM examples file does not exist: {self.gsm_examples_file}")

        rows: List[dict] = []
        with self.gsm_examples_file.open() as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise ValueError(
                        f"GSM examples file {self.gsm_examples_file} line {line_no} "
                        "must contain a JSON object"
                    )
                rows.append(row)

        if self.sample_size is not None and self.sample_size > 0:
            rows = rows[: self.sample_size]
        return rows

    def _get_grammar_file(self) -> Path:
        """Get the grammar file path for the dataset."""
        if self._grammar_file is None:
            grammars_dir = Path(__file__).parent.parent / "grammars"
            if self.dataset_name == "gsm_symbolic":
                self._grammar_file = grammars_dir / "gsm.lark"
            elif self.dataset_name == "folio":
                self._grammar_file = grammars_dir / "folio.lark"
            elif self.dataset_name == "spider":
                self._grammar_file = grammars_dir / "sql.lark"
            else:
                raise ValueError(f"Unknown dataset: {self.dataset_name}")
        return self._grammar_file

    def _get_grammar_text(self) -> str:
        """Load and cache the active grammar text."""
        if self._base_grammar_text is None:
            self._base_grammar_text = self._get_grammar_file().read_text()
        return self._base_grammar_text

    def _get_gsm_allowed_variables(self, example: dict) -> List[str]:
        """Return the numeric variable names allowed for one GSM example."""
        if self.dataset_name != "gsm_symbolic":
            return []

        from evaluations.gsm_symbolic.grammar import extract_variables_from_mapping

        variable_types = example.get("variable_types") or {}
        if not isinstance(variable_types, dict):
            return []
        return extract_variables_from_mapping(variable_types)

    def _build_gsm_dynamic_parser(self, env: Dict[str, Any], example: dict):
        """Create a per-example GSM parser restricted to CRANE's allowed variables."""
        if self.dataset_name != "gsm_symbolic":
            return None

        allowed_variables = self._get_gsm_allowed_variables(example)
        if not allowed_variables:
            return None

        from evaluations.common.parser_utils import create_lark_dafny_parser
        from evaluations.gsm_symbolic.grammar import build_dynamic_grammar

        cache_key = tuple(sorted(allowed_variables))
        parser_factory = self._dynamic_parser_factory_cache.get(cache_key)
        if parser_factory is None:
            grammar_text = build_dynamic_grammar(self._get_grammar_text(), list(cache_key))
            parser_factory = create_lark_dafny_parser(
                grammar_text,
                env["VerifiedDecoderAgent"],
                env["_dafny"],
                start="csd_start",
                tokenizer=env["tokenizer"],
            )
            self._dynamic_parser_factory_cache[cache_key] = parser_factory

        return parser_factory(env["lm"]._Tokens)

    def _build_spider_dynamic_parser(self, env: Dict[str, Any], example: dict):
        """
        Return a SQL parser for CRANE/CSD generation on Spider.

        Historically this narrowed the grammar per-example to the schema's
        exact tables/columns. That was correct but catastrophically slow:
        every unseen schema triggers a fresh syncode DFA mask store build
        (O(vocab_size × grammar_states), 15-30 min per schema). Across 50
        Spider examples × many iterations that's hours of rebuild time.

        itergen's faster design uses a single base SQL grammar (cached
        mask store) and enforces schema-awareness by runtime backtracking
        in the iterator. We do the same here: return None so run_crane_csd
        falls back to building a single parser from the base grammar_file,
        whose DFA mask store is built once and cached forever.

        Schema correctness is still measured: _exec_match_spider executes
        the predicted query against the SQLite database and compares to
        the gold query's rows, so invalid column/table references are
        caught at scoring time regardless of the grammar used for decode.
        """
        return None

    def _extract_answer_spider(self, output: str) -> Optional[str]:
        """Extract a SQL query: take the raw output up to the first blank line.

        We no longer require <<...>> delimiters for Spider — the entire output
        is the SQL query. If <<...>> happens to be present (legacy from the
        old contract or strategy-imposed), we still strip them so the SQL
        executes cleanly against SQLite.
        """
        if not output:
            return None
        raw = output.split("\n\n")[0]
        cleaned = raw.replace("\n", " ").replace("\r", " ").strip()
        # Strip stray legacy delimiters anywhere they appear.
        cleaned = cleaned.replace("<<", " ").replace(">>", " ")
        cleaned = " ".join(cleaned.split()).rstrip(";").strip()
        return cleaned or None

    def _exec_match_spider(
        self,
        pred_sql: Optional[str],
        gold_sql: Optional[str],
        example: dict,
    ) -> bool:
        """
        Cheap per-example execution-match for the synthesis feedback loop.

        Executes pred and gold on the example's SQLite database and compares
        result sets (order-insensitive). Not as thorough as Spider's official
        evaluator, but fast enough for per-iteration scoring. The CLI still
        calls the full Spider evaluator at batch time.
        """
        if not pred_sql or not gold_sql:
            return False

        import sqlite3
        from evaluations.sql_spider.dataset import default_db_dir

        db_id = example.get("db_id", "")
        if not db_id:
            return False
        db_path = default_db_dir() / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            return False

        def _run(sql: str):
            try:
                con = sqlite3.connect(str(db_path))
                con.text_factory = lambda b: b.decode("utf-8", errors="ignore")
                cur = con.cursor()
                cur.execute(sql)
                rows = cur.fetchall()
                con.close()
                return rows
            except Exception:
                return None

        pred_rows = _run(pred_sql)
        gold_rows = _run(gold_sql)
        if pred_rows is None or gold_rows is None:
            return False

        # Order-insensitive bag comparison (matches Spider's default behavior
        # unless the gold query has ORDER BY; we keep it simple here).
        try:
            return sorted(map(tuple, pred_rows)) == sorted(map(tuple, gold_rows))
        except TypeError:
            return list(map(tuple, pred_rows)) == list(map(tuple, gold_rows))

    def _get_syntax_parser(self, example: Optional[dict] = None):
        """Create or reuse a syntax parser for one example's allowed variables."""
        from lark import Lark

        if self.dataset_name != "gsm_symbolic" or example is None:
            return Lark(self._get_grammar_text(), start="start", parser="lalr")

        allowed_variables = self._get_gsm_allowed_variables(example)
        if not allowed_variables:
            return Lark(self._get_grammar_text(), start="start", parser="lalr")

        from evaluations.gsm_symbolic.grammar import build_dynamic_grammar

        cache_key = tuple(sorted(allowed_variables))
        parser = self._syntax_parser_cache.get(cache_key)
        if parser is None:
            grammar_text = build_dynamic_grammar(self._get_grammar_text(), list(cache_key))
            parser = Lark(grammar_text, start="start", parser="lalr")
            self._syntax_parser_cache[cache_key] = parser
        return parser

    def _load_dataset_sample(self) -> list:
        """Load a sample of the dataset for evaluation."""
        if self._dataset is not None:
            return self._dataset

        if self.dataset_name == "gsm_symbolic":
            if self.gsm_examples_file is not None:
                self._dataset = self._load_gsm_examples_file()
            else:
                from evaluations.gsm_symbolic.dataset import load_gsm_from_crane_folder

                ds = load_gsm_from_crane_folder(
                    crane_dir=self.gsm_source_dir,
                    limit=self.sample_size,
                    indices=self._load_gsm_split_indices(),
                )
                self._dataset = list(ds)
        elif self.dataset_name == "folio":
            from evaluations.folio.dataset import load_folio
            ds = load_folio(
                split="validation",
                num_samples=self.sample_size,
                seed=42,
            )
            self._dataset = list(ds)
        elif self.dataset_name == "spider":
            from evaluations.sql_spider.dataset import load_spider
            ds = load_spider(
                source="auto",
                limit=self.sample_size,
                random_sample=True,
                seed=self.sample_seed,
            )
            self._dataset = list(ds)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        return self._dataset

    def _setup_environment(self, compiled_module_path: Path) -> Dict[str, Any]:
        """
        Set up the Dafny environment for evaluation.

        Args:
            compiled_module_path: Path to the compiled CSD module

        Returns:
            Environment dict with loaded modules
        """
        run_dir = compiled_module_path.parent
        if run_dir.name == "generated_csd":
            run_dir = run_dir.parent

        env_cache_key = (
            str(run_dir.resolve()),
            self.dataset_name,
            self.model_name,
            self.backend,
            self.device,
            self.load_in_4bit,
            self.load_in_8bit,
            self.vllm_tensor_parallel_size,
            self.vllm_pipeline_parallel_size,
            self.vllm_gpu_memory_utilization,
            self.vllm_max_model_len,
            self.vllm_enforce_eager,
        )
        if self._env is not None and self._env_cache_key == env_cache_key:
            return self._env

        if self.dataset_name == "gsm_symbolic":
            from evaluations.gsm_symbolic.environment import setup_dafny_environment
        elif self.dataset_name == "spider":
            from evaluations.sql_spider.environment import setup_dafny_environment
        else:
            from evaluations.folio.environment import setup_dafny_environment

        def _make_env(gpu_memory_utilization: float) -> Dict[str, Any]:
            return setup_dafny_environment(
                run_dir=run_dir,
                model_name=self.model_name,
                backend=self.backend,
                device=self.device,
                grammar_file=self._get_grammar_file(),
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                vllm_tensor_parallel_size=self.vllm_tensor_parallel_size,
                vllm_pipeline_parallel_size=self.vllm_pipeline_parallel_size,
                vllm_gpu_memory_utilization=gpu_memory_utilization,
                vllm_max_model_len=self.vllm_max_model_len,
                vllm_enforce_eager=self.vllm_enforce_eager,
            )

        if self.backend != "vllm":
            return _make_env(self.vllm_gpu_memory_utilization)

        util_candidates: List[float] = []
        for candidate in [
            self.vllm_gpu_memory_utilization,
            0.55,
            0.5,
            0.45,
            0.4,
        ]:
            if candidate <= self.vllm_gpu_memory_utilization and candidate not in util_candidates:
                util_candidates.append(candidate)

        last_error: Exception | None = None
        for util in util_candidates:
            try:
                if util != self.vllm_gpu_memory_utilization:
                    print(
                        f"Retrying vLLM evaluator startup with lower "
                        f"gpu_memory_utilization={util:.2f}"
                    )
                env = _make_env(util)
                self._env = env
                self._env_cache_key = env_cache_key
                return env
            except Exception as exc:
                last_error = exc
                message = str(exc)
                startup_memory_error = (
                    "desired GPU memory utilization" in message
                    or "Free memory on device" in message
                    or "Engine core initialization failed" in message
                )
                if not startup_memory_error or util == util_candidates[-1]:
                    raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to initialize evaluation environment.")

    def _extract_constrained_content(self, output: str) -> List[str]:
        """Extract content within << >> delimiters.

        Non-greedy `.*?` with DOTALL so spans containing `<`/`>` operators
        (e.g. Spider SQL `WHERE a > b`) match. SQL has no `>>` operator,
        so the next `>>` always closes the span.
        """
        return re.findall(r"<<\s*(.*?)\s*>>", output, flags=re.DOTALL)

    def _truncate_gsm_output(self, output: str) -> str:
        """Trim obvious prompt restarts so scoring focuses on the first answer block."""
        cut_points: List[int] = []
        for marker in [
            "\nAssistant:",
            "\n\nAssistant:",
            "\nQ:",
            "\n\nQ:",
            "\nSolve the question above.",
            "\n\nSolve the question above.",
        ]:
            idx = output.find(marker)
            if idx > 0:
                cut_points.append(idx)

        if not cut_points:
            return output

        return output[:min(cut_points)].rstrip()

    def _parse_variable_assignments(self, text: str) -> dict:
        """Parse variable assignments from text like 'a = 5', 'n1 = 72.5', etc."""
        assignments = {}
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+]?\d+(?:\.\d+)?)\b'
        for match in re.finditer(pattern, text):
            var_name = match.group(1)
            try:
                assignments[var_name] = float(match.group(2))
            except ValueError:
                pass
        return assignments

    def _parse_symbolic_assignments(self, text: str) -> dict[str, str]:
        """Parse simple symbolic assignments like 'x = 2 * y' or 'y = 1/10'."""
        assignments: dict[str, str] = {}
        pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^,\n;]+)"
        for match in re.finditer(pattern, text):
            var_name = match.group(1)
            expr = match.group(2).strip()
            if expr.endswith(".") and len(expr) >= 2 and expr[-2].isdigit():
                expr = expr
            else:
                expr = expr.rstrip(".").strip()
            if expr:
                assignments[var_name] = expr
        return assignments

    def _safe_eval_arithmetic(self, expr: str) -> Optional[float]:
        """Safely evaluate a numeric arithmetic expression using AST (no eval())."""
        import ast
        import operator as op

        ops = {
            ast.Add: op.add, ast.Sub: op.sub,
            ast.Mult: op.mul, ast.Div: op.truediv,
            ast.FloorDiv: op.floordiv, ast.Mod: op.mod,
            ast.USub: op.neg, ast.UAdd: op.pos,
        }

        def _eval(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            elif isinstance(node, ast.Num):  # Python 3.7 compat
                return float(node.n)
            elif isinstance(node, ast.BinOp) and type(node.op) in ops:
                return ops[type(node.op)](_eval(node.left), _eval(node.right))
            elif isinstance(node, ast.UnaryOp) and type(node.op) in ops:
                return ops[type(node.op)](_eval(node.operand))
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == 'int' and len(node.args) == 1):
                return float(int(_eval(node.args[0])))
            else:
                raise ValueError(f"Unsupported node: {type(node)}")

        try:
            tree = ast.parse(expr.strip(), mode='eval')
            return _eval(tree.body)
        except Exception:
            return None

    def _evaluate_symbolic_expression(self, expr: str, var_values: dict) -> Optional[float]:
        """Substitute variable values into a symbolic expression and evaluate."""
        substituted = expr
        # Substitute longest names first to avoid partial replacement (n10 before n1)
        for var in sorted(var_values.keys(), key=len, reverse=True):
            substituted = re.sub(r'\b' + re.escape(var) + r'\b',
                                 str(var_values[var]), substituted)
        # If alphabetic chars remain, some variables were unresolved
        if re.search(r'[a-zA-Z_]', substituted):
            return None
        return self._safe_eval_arithmetic(substituted)

    def _resolve_symbolic_assignments(self, text: str) -> dict[str, float]:
        """Resolve assignment chains like 'x = 2 * y, y = 14' into numeric values."""
        raw_assignments = self._parse_symbolic_assignments(text)
        resolved: dict[str, float] = {}

        def resolve(name: str, stack: set[str]) -> Optional[float]:
            if name in resolved:
                return resolved[name]
            if name in stack:
                return None

            expr = raw_assignments.get(name)
            if expr is None:
                return None

            stack = set(stack)
            stack.add(name)
            substituted = expr
            var_names = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expr))
            for dep in sorted(var_names, key=len, reverse=True):
                if dep == name:
                    continue
                dep_value = resolve(dep, stack)
                if dep_value is None:
                    continue
                substituted = re.sub(r"\b" + re.escape(dep) + r"\b", str(dep_value), substituted)

            if re.search(r"[a-zA-Z_]", substituted):
                return None

            value = self._safe_eval_arithmetic(substituted)
            if value is None:
                return None
            resolved[name] = value
            return value

        for var_name in list(raw_assignments.keys()):
            resolve(var_name, set())

        return resolved

    def _extract_answer_expression_gsm(self, output: str) -> Optional[str]:
        """Extract the expression after 'The answer is ...' if present."""
        match = re.search(
            r"The answer is\s+(.+?)(?:\.\s*$|\n|$)",
            output,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None
        expr = match.group(1).strip()
        return expr.rstrip(".").strip() or None

    def _evaluate_gsm_expression(self, expr: str, output: str) -> Optional[str]:
        """Evaluate a GSM expression using resolved variable assignments when needed."""
        if not expr:
            return None

        if not re.search(r"[a-zA-Z_]", expr):
            result = self._safe_eval_arithmetic(expr)
            if result is not None:
                val = int(result) if result == int(result) else result
                return str(val)

        var_values = self._resolve_symbolic_assignments(output)
        if not var_values:
            var_values = self._parse_variable_assignments(output)

        if var_values:
            result = self._evaluate_symbolic_expression(expr, var_values)
            if result is not None:
                val = int(result) if result == int(result) else result
                return str(val)

        return None

    def _extract_answer_gsm(self, output: str) -> Optional[str]:
        """Extract numeric answer from GSM-Symbolic output within << >> delimiters."""
        truncated_output = self._truncate_gsm_output(output)

        answer_expr = self._extract_answer_expression_gsm(truncated_output)
        if answer_expr is not None:
            answer = self._evaluate_gsm_expression(answer_expr, truncated_output)
            if answer is not None:
                return answer

        matches = self._extract_constrained_content(truncated_output)
        if not matches:
            return None

        last_match = matches[-1].strip()

        # Case 1: expression contains "=" — take the part after "=" (e.g. "a + b = 8")
        if "=" in last_match:
            answer_part = last_match.split("=")[-1].strip()
            num_match = re.search(r"[-+]?\d*\.?\d+", answer_part)
            if num_match:
                return num_match.group()

        # Case 2: purely numeric expression — evaluate directly (e.g. "5 + 3")
        answer = self._evaluate_gsm_expression(last_match, truncated_output)
        if answer is not None:
            return answer

        # Case 3: symbolic expression — parse variable assignments from surrounding text
        # and substitute in (e.g. "a + b" with "a = 5, b = 3" defined earlier)
        var_values = self._resolve_symbolic_assignments(truncated_output)
        if not var_values:
            var_values = self._parse_variable_assignments(truncated_output)
        if var_values:
            result = self._evaluate_symbolic_expression(last_match, var_values)
            if result is not None:
                val = int(result) if result == int(result) else result
                return str(val)

        return None

    def _fol_keyword_to_unicode(self, text: str) -> str:
        """Convert {keyword} FOL syntax from grammar to Unicode symbols for Prover9."""
        replacements = {
            "{forall}": "∀",
            "{exists}": "∃",
            "{and}": "∧",
            "{or}": "∨",
            "{xor}": "⊕",
            "{not}": "¬",
            "{implies}": "→",
            "{iff}": "↔",
        }
        for keyword, symbol in replacements.items():
            text = text.replace(keyword, symbol)
        return text

    def _extract_answer_folio(self, output: str, example: Optional[Any] = None) -> Optional[str]:
        """
        Extract FOL answer from FOLIO output using Prover9 solver.

        Extracts FOL formulas from constrained << >> segments, converts
        {keyword} syntax to Unicode, builds a Prover9 logic program, and
        runs the solver to determine True/False/Unknown.

        Falls back to keyword matching if the solver fails.
        """
        segments = self._extract_constrained_content(output)
        if not segments:
            return self._extract_answer_folio_fallback(output)

        # Convert {keyword} syntax to Unicode FOL symbols
        fol_segments = [self._fol_keyword_to_unicode(s.strip()) for s in segments]

        # Heuristic: all segments except the last are premises, last is conclusion
        if len(fol_segments) >= 2:
            premises = fol_segments[:-1]
            conclusion = fol_segments[-1]
        elif len(fol_segments) == 1:
            # Single segment — treat as conclusion, no FOL premises
            premises = []
            conclusion = fol_segments[0]
        else:
            return self._extract_answer_folio_fallback(output)

        # Build Prover9 logic program
        logic_lines = ["Premises:"]
        for p in premises:
            logic_lines.append(f"{p} ::: premise")
        logic_lines.append("Conclusion:")
        logic_lines.append(f"{conclusion} ::: conclusion")
        logic_program = "\n".join(logic_lines)

        try:
            from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program

            program = FOL_Prover9_Program(logic_program, dataset_name="FOLIO")
            if not program.flag:
                return self._extract_answer_folio_fallback(output)

            answer, error = program.execute_program()
            if answer in ("True", "False", "Unknown"):
                return answer
        except Exception:
            pass

        return self._extract_answer_folio_fallback(output)

    def _extract_answer_folio_fallback(self, output: str) -> Optional[str]:
        """Fallback: extract FOL answer via keyword matching."""
        output_lower = output.lower()
        if "true" in output_lower:
            return "True"
        elif "false" in output_lower:
            return "False"
        elif "unknown" in output_lower or "uncertain" in output_lower:
            return "Unknown"
        return None

    def _answers_match(self, actual: Optional[str], expected: str) -> bool:
        """Check if actual and expected answers match, normalizing Uncertain/Unknown."""
        if actual is None:
            return False
        a = str(actual).strip().lower()
        e = str(expected).strip().lower()
        if a in ("uncertain", "unknown"):
            a = "unknown"
        if e in ("uncertain", "unknown"):
            e = "unknown"
        return a == e

    def _gsm_symbolic_equivalence(
        self, model_expr: Optional[str], expected_expr: str, variable_types: dict
    ) -> bool:
        """Check symbolic equivalence via random value substitution (matches CRANE's method)."""
        if model_expr is None:
            return False
        import random as _rng

        var_names = set(re.findall(r'\b[a-zA-Z_]\w*\b', model_expr + ' ' + expected_expr))
        var_names -= {'int'}

        for name in var_names:
            if name not in variable_types:
                return False

        for _ in range(200):
            env = {}
            for var in var_names:
                vtype = variable_types.get(var, 'int')
                if vtype == 'float between 0 and 1':
                    env[var] = _rng.uniform(0.001, 1)
                elif vtype == 'float':
                    env[var] = _rng.uniform(0.001, 100)
                else:
                    env[var] = _rng.randint(1, 100)
            try:
                val_model = eval(model_expr, {"__builtins__": {}}, {**env, 'int': int})
                val_expected = eval(expected_expr, {"__builtins__": {}}, {**env, 'int': int})
            except Exception:
                return False
            if abs(val_model - val_expected) > 1e-6 * max(1, abs(val_expected)):
                return False
        return True

    def _get_expected_answer(self, example: dict) -> str:
        """Get the expected answer from a dataset example."""
        if self.dataset_name == "gsm_symbolic":
            symbolic = example.get("answer_parsed", "")
            if symbolic:
                return symbolic
            answer_str = example.get("answer", "")
            match = re.search(r"####\s*([-+]?\d*\.?\d+)", answer_str)
            if match:
                return match.group(1)
            return answer_str
        elif self.dataset_name == "spider":
            return (example.get("query") or "").strip()
        else:
            if hasattr(example, "label"):
                return example.label
            return example.get("label", "Unknown")

    def _format_prompt(self, example: dict) -> Union[str, List[dict]]:
        """Format a dataset example as a prompt."""
        if self.dataset_name == "gsm_symbolic":
            from evaluations.gsm_symbolic.cli import _build_gsm_prompt
            question = example.get("question_parsed") or example.get("question", "")
            return _build_gsm_prompt(question, do_cot=True, modify_system_prompt=True)
        elif self.dataset_name == "spider":
            db_id = example.get("db_id", "")
            db_info = example.get("db_info", "")
            question = example.get("question", "")
            return (
                "You are given a database schema and a question. "
                "Write a SINGLE SQL query answering the question, using ONLY the tables and columns in the schema. "
                "Emit the SQL on a single line. Stop after the query — no explanation, no code fences.\n\n"
                "Example:\n"
                "db_id: concert_singer\n"
                "db_info: # singer ( singer_id , name , country , age )\n"
                "question: How many singers do we have?\n"
                "SQL: SELECT count(*) FROM singer\n\n"
                f"db_id: {db_id}\n"
                f"db_info: {db_info}\n"
                f"question: {question}\n"
                "SQL: "
            )
        else:
            # FOLIOExample is a dataclass — access via attribute or .get() for dict fallback
            if hasattr(example, "premises"):
                premises_raw = example.premises
                conclusion = example.conclusion
            else:
                premises_raw = example.get("premises", "")
                conclusion = example.get("conclusion", "")
            premises_str = premises_raw if isinstance(premises_raw, str) else "\n".join(f"- {p}" for p in premises_raw)
            return (
                "Translate each premise and the conclusion into first-order logic (FOL), "
                "then determine if the conclusion is True, False, or Unknown.\n\n"
                "FOL syntax rules:\n"
                "  - Predicates: UppercaseName(arg), e.g. Dog(x), Mammal(fido)\n"
                "  - Variables: single lowercase letter (x, y, z)\n"
                "  - Constants: lowercase 2+ chars (fido, alice, bob)\n"
                "  - Quantifiers: {forall} x, {exists} x\n"
                "  - Connectives: {and}, {or}, {not}, {implies}, {iff}\n\n"
                "Wrap EACH FOL formula in << >> delimiters. "
                "Write one << >> per premise, then one << >> for the conclusion. "
                "Then state True, False, or Unknown.\n\n"
                "Example:\n"
                "Premises:\n"
                "- All dogs are mammals.\n"
                "- Fido is a dog.\n\n"
                "Conclusion: Fido is a mammal.\n\n"
                "Answer:\n"
                "<<{forall} x (Dog(x) {implies} Mammal(x))>>\n"
                "<<Dog(fido)>>\n"
                "<<Mammal(fido)>>\n"
                "True\n\n"
                f"Premises:\n{premises_str}\n\n"
                f"Conclusion: {conclusion}\n\n"
                "Answer:"
            )

    def _contains_delimiters(self, output: str) -> bool:
        """Check if the output contains at least one non-empty << >> segment."""
        return "<<" in output and ">>" in output

    def _check_syntax_validity(
        self,
        output: str,
        example: Optional[dict] = None,
    ) -> Tuple[bool, List[Tuple[str, bool]]]:
        """
        Check if constrained segments have valid syntax.

        Returns:
            Tuple of (all_valid, list of (segment, is_valid) tuples)
        """
        from lark.exceptions import LarkError

        segments: List[Tuple[str, bool]] = []
        matches = self._extract_constrained_content(output)

        if not matches:
            return True, []

        try:
            parser = self._get_syntax_parser(example)
            for match in matches:
                try:
                    parser.parse(match.strip())
                    segments.append((match, True))
                except LarkError:
                    segments.append((match, False))
        except Exception:
            return True, [(m, True) for m in matches]

        all_valid = all(is_valid for _, is_valid in segments) if segments else True
        return all_valid, segments

    def evaluate_sample(
        self,
        compiled_module_path: Path,
        sample_size: Optional[int] = None,
    ) -> EvaluationResult:
        """
        Evaluate the compiled CSD on a sample of the dataset.

        Args:
            compiled_module_path: Path to the compiled GeneratedCSD.py module
            sample_size: Number of examples to evaluate (overrides init value)

        Returns:
            EvaluationResult with metrics and sample outputs
        """
        if sample_size is not None:
            self.sample_size = sample_size

        # Always re-sample so each iteration gets a fresh random example
        self._dataset = None

        start_time = time.time()
        sample_outputs: List[Dict[str, Any]] = []

        try:
            dataset = self._load_dataset_sample()
            env = self._setup_environment(compiled_module_path)

            if self.dataset_name == "gsm_symbolic":
                from evaluations.gsm_symbolic.generation import run_crane_csd
            elif self.dataset_name == "spider":
                from evaluations.sql_spider.generation import run_crane_csd
            else:
                from evaluations.folio.generation import run_crane_csd

            num_correct = 0
            all_examples_contain_delimiters = True
            num_examples_syntax_pass = 0

            for i, example in enumerate(dataset):
                print(f"  [EVAL] Processing example {i+1}/{len(dataset)}...", flush=True)
                example_start = time.time()
                prompt = self._format_prompt(example)
                expected = self._get_expected_answer(example)

                try:
                    print(f"  [EVAL]   Running CSD strategy (max_steps={self.max_steps})...", flush=True)
                    with _PerExampleTimer(self.max_seconds_per_example):
                        output_text, token_count, gen_time, constrained_segments, helper_trace = run_crane_csd(
                            env=env,
                            prompt_text=prompt,
                            max_steps=self.max_steps,
                            step_token_budget=self.step_token_budget,
                            grammar_file=self._get_grammar_file(),
                            dynamic_parser=(
                                self._build_gsm_dynamic_parser(env, example)
                                if self.dataset_name == "gsm_symbolic"
                                else self._build_spider_dynamic_parser(env, example)
                                if self.dataset_name == "spider"
                                else None
                            ),
                        )
                    example_time = time.time() - example_start
                    print(f"  [EVAL]   Generated {token_count} tokens in {example_time:.2f}s", flush=True)

                    scored_output = (
                        self._truncate_gsm_output(output_text)
                        if self.dataset_name == "gsm_symbolic"
                        else output_text
                    )

                    if self.dataset_name == "gsm_symbolic":
                        expr_matches = re.findall(r"<<\s*(.*?)\s*>>", scored_output, flags=re.DOTALL)
                        actual = expr_matches[-1].strip() if expr_matches else None
                        answer_source = "last_visible_span" if expr_matches else "none"
                    elif self.dataset_name == "spider":
                        actual = self._extract_answer_spider(scored_output)
                        answer_source = "hidden_or_task_extractor" if actual is not None else "none"
                    else:
                        actual = self._extract_answer_folio(scored_output, example=example)
                        answer_source = "hidden_or_task_extractor" if actual is not None else "none"

                    if self.dataset_name == "gsm_symbolic":
                        vt = example.get("variable_types", {})
                        if isinstance(vt, str):
                            try:
                                vt = eval(vt)
                            except Exception:
                                vt = {}
                        if vt and example.get("answer_parsed"):
                            is_correct = self._gsm_symbolic_equivalence(actual, expected, vt)
                        else:
                            numeric_actual = self._extract_answer_gsm(scored_output)
                            if actual is None and numeric_actual is not None:
                                answer_source = "text_fallback"
                            numeric_expected = re.search(r"####\s*([-+]?\d*\.?\d+)", example.get("answer", ""))
                            if numeric_expected:
                                is_correct = self._answers_match(numeric_actual, numeric_expected.group(1))
                            else:
                                is_correct = self._answers_match(numeric_actual, expected)
                    elif self.dataset_name == "spider":
                        is_correct = self._exec_match_spider(actual, expected, example)
                    else:
                        is_correct = self._answers_match(actual, expected)
                    if is_correct:
                        num_correct += 1

                    visible_delimiters = self._contains_delimiters(scored_output)
                    used_hidden_chunk = bool(constrained_segments) or any(
                        event.get("helper") in EvaluationResult._CONSTRAINED_HELPERS
                        for event in (helper_trace or [])
                    )
                    contains_delimiters = (
                        used_hidden_chunk if self.dataset_name == "spider" else visible_delimiters
                    )
                    all_examples_contain_delimiters = (
                        all_examples_contain_delimiters and contains_delimiters
                    )

                    all_valid_syntax, segments = self._check_syntax_validity(scored_output, example=example)
                    # Per-example syntax pass:
                    # - GSM/FOLIO: visible <<...>> chunks must exist and parse.
                    # - Spider: chunks are internal/hidden; visible delimiter tokens are not
                    #   part of the answer contract, so count parser-governed chunk usage.
                    if self.dataset_name == "spider":
                        example_syntax_pass = used_hidden_chunk
                    else:
                        example_syntax_pass = bool(segments) and all_valid_syntax
                    num_examples_syntax_pass += int(example_syntax_pass)
                    example_syntax_rate = 1.0 if example_syntax_pass else 0.0

                    if self.dataset_name == "gsm_symbolic" and example_syntax_rate < 1.0:
                        if is_correct:
                            num_correct -= 1
                        actual = ""
                        answer_source = "syntax_gate"
                        is_correct = False

                    visible_span_lengths = [
                        len(segment.strip().split()) for segment, _ in segments
                    ]
                    valid_visible_span_lengths = [
                        len(segment.strip().split())
                        for segment, is_valid in segments
                        if is_valid
                    ]
                    num_valid_visible_spans = sum(1 for _, is_valid in segments if is_valid)

                    if hasattr(example, "conclusion"):
                        q_str = (example.premises + " | " + example.conclusion)[:200]
                    else:
                        q_str = example.get("question", str(example.get("premises", "")))[:200]
                    sample_outputs.append({
                        "question": q_str,
                        "expected": expected,
                        "actual": actual if actual == "" else actual or output_text[:100],
                        "full_output": output_text,
                        "scored_output": scored_output,
                        "answer_source": answer_source,
                        "has_extracted_answer": bool(actual) or answer_source == "text_fallback",
                        "is_correct": is_correct,
                        "contains_delimiters": contains_delimiters,
                        "visible_delimiters": visible_delimiters,
                        "used_constrained_chunk": used_hidden_chunk,
                        "uses_hidden_chunks": self.dataset_name == "spider",
                        "is_syntax_valid": example_syntax_pass,
                        "syntax_rate": example_syntax_rate,
                        "num_visible_spans": len(segments),
                        "num_valid_visible_spans": num_valid_visible_spans,
                        "visible_span_token_lengths": visible_span_lengths,
                        "valid_visible_span_token_lengths": valid_visible_span_lengths,
                        "token_count": token_count,
                        "hit_max_steps": token_count >= self.max_steps,
                        "time_seconds": gen_time,
                        "runtime_budget_exceeded": (
                            self.max_seconds_per_example is not None
                            and gen_time > self.max_seconds_per_example
                        ),
                        "helper_trace": helper_trace,
                    })

                except Exception as e:
                    if hasattr(example, "conclusion"):
                        q_str = (example.premises + " | " + example.conclusion)[:200]
                    else:
                        q_str = example.get("question", str(example.get("premises", "")))[:200]
                    elapsed = time.time() - example_start
                    timed_out = isinstance(e, PerExampleTimeout)
                    if timed_out:
                        print(f"  [EVAL]   Timed out after {elapsed:.2f}s", flush=True)
                    sample_outputs.append({
                        "question": q_str,
                        "expected": expected,
                        "actual": None,
                        "full_output": "",
                        "scored_output": "",
                        "answer_source": "none",
                        "has_extracted_answer": False,
                        "is_correct": False,
                        "contains_delimiters": False,
                        "visible_delimiters": False,
                        "used_constrained_chunk": False,
                        "uses_hidden_chunks": self.dataset_name == "spider",
                        "is_syntax_valid": False,
                        "syntax_rate": 0.0,
                        "num_visible_spans": 0,
                        "num_valid_visible_spans": 0,
                        "visible_span_token_lengths": [],
                        "valid_visible_span_token_lengths": [],
                        "token_count": 0,
                        "hit_max_steps": False,
                        "time_seconds": elapsed,
                        "runtime_budget_exceeded": timed_out or (
                            self.max_seconds_per_example is not None
                            and elapsed > self.max_seconds_per_example
                        ),
                        "error": str(e),
                    })
                    if timed_out:
                        total_time = time.time() - start_time
                        evaluated_count = len(sample_outputs)
                        max_sample_time = max(
                            (float(sample.get("time_seconds", 0.0)) for sample in sample_outputs),
                            default=0.0,
                        )
                        return EvaluationResult(
                            success=True,
                            accuracy=num_correct / max(1, evaluated_count),
                            contains_delimiters=False,
                            syntax_rate=num_examples_syntax_pass / max(1, evaluated_count),
                            num_examples=evaluated_count,
                            num_correct=num_correct,
                            total_time_seconds=total_time,
                            max_sample_time_seconds=max_sample_time,
                            error=(
                                "Evaluation stopped early because one example exceeded "
                                f"the {self.max_seconds_per_example:.2f}s runtime budget."
                            ),
                            sample_outputs=sample_outputs,
                        )

            total_time = time.time() - start_time
            num_examples = len(dataset)
            max_sample_time = max((float(sample.get("time_seconds", 0.0)) for sample in sample_outputs), default=0.0)

            return EvaluationResult(
                success=True,
                accuracy=num_correct / max(1, num_examples),
                contains_delimiters=all_examples_contain_delimiters,
                syntax_rate=num_examples_syntax_pass / max(1, num_examples),
                num_examples=num_examples,
                num_correct=num_correct,
                total_time_seconds=total_time,
                max_sample_time_seconds=max_sample_time,
                sample_outputs=sample_outputs,
            )

        except Exception as e:
            return EvaluationResult(
                success=False,
                accuracy=0.0,
                contains_delimiters=False,
                syntax_rate=0.0,
                num_examples=0,
                num_correct=0,
                total_time_seconds=time.time() - start_time,
                error=str(e),
                sample_outputs=sample_outputs,
            )
