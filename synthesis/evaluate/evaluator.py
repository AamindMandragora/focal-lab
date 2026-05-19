"""
Evaluation module for synthesis feedback loop.

Provides quick evaluation of synthesized CSD strategies on dataset samples
to enable feedback-driven refinement based on actual performance metrics.
"""

from __future__ import annotations

import json
import os
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
    accuracy_denominator: Optional[int] = None
    accuracy_definition: str = "correct_examples_over_all_examples"
    invalid_outputs_excluded_from_accuracy: int = 0
    max_sample_time_seconds: float = 0.0
    early_stopped: bool = False
    early_stop_reason: Optional[str] = None
    planned_num_examples: Optional[int] = None
    task_guidance: List[str] = field(default_factory=list)

    # Sample outputs for feedback (question, expected, actual, is_correct)
    sample_outputs: List[Dict[str, Any]] = field(default_factory=list)

    # Error information if evaluation failed
    error: Optional[str] = None
    aux_metrics: Dict[str, Any] = field(default_factory=dict)

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
        if self.early_stopped:
            return False
        runtime_ok = True
        if max_seconds_per_example is not None:
            runtime_ok = self.max_sample_time_seconds <= max_seconds_per_example
        return (
            runtime_ok
            and self.accuracy >= min_accuracy
            and self.syntax_rate >= min_syntax_rate
        )

    def get_feedback_summary(self) -> str:
        """Generate a summary for feedback to the generator."""
        eval_count_label = (
            f"{self.num_examples}/{self.planned_num_examples}"
            if self.early_stopped and self.planned_num_examples
            else str(self.num_examples)
        )
        lines = [
            f"Evaluation Results ({eval_count_label} examples):",
            (
                "  Accuracy: "
                f"{self.accuracy:.1%} "
                f"({self.num_correct}/{self.accuracy_denominator or self.num_examples})"
            ),
            f"  Contains << >>: {'yes' if self.contains_delimiters else 'no'}",
            f"  Syntax Rate: {self.syntax_rate:.1%}",
            f"  Total Time: {self.total_time_seconds:.2f}s",
            f"  Slowest Example Time: {self.max_sample_time_seconds:.2f}s",
        ]
        if self.early_stopped:
            lines.append("  Early Stop: yes")
            if self.early_stop_reason:
                lines.append(f"  Early Stop Reason: {self.early_stop_reason}")
        if self.accuracy_definition != "correct_examples_over_all_examples":
            lines.append(f"  Accuracy Definition: {self.accuracy_definition}")
        if self.invalid_outputs_excluded_from_accuracy:
            lines.append(
                "  Invalid outputs excluded from accuracy denominator: "
                f"{self.invalid_outputs_excluded_from_accuracy}"
            )
        if self.task_guidance:
            lines.extend(["", "Prompt guidance used by this attempt:"])
            for guidance in self.task_guidance:
                lines.append(f"  - {guidance}")

        smiles_trial = self.aux_metrics.get("smiles_paper_trial")
        if isinstance(smiles_trial, dict):
            lines.extend(
                [
                    "",
                    "SMILES Quality Metrics (paper-aligned, single trial):",
                    f"  RDKit Validity: {smiles_trial.get('validity_rdkit', 0.0):.1%}",
                    f"  Membership: {smiles_trial.get('membership', 0.0):.1%}",
                    (
                        "  Diversity (avg pairwise Tanimoto distance): "
                        f"{smiles_trial.get('diversity_tanimoto') if smiles_trial.get('diversity_tanimoto') is not None else 'n/a'}"
                    ),
                    (
                        "  RetroStar score: "
                        f"{smiles_trial.get('retro_score') if smiles_trial.get('retro_score') is not None else 'n/a'}"
                    ),
                    (
                        "  Samples to 100 unique valid (cap 1000): "
                        f"{smiles_trial.get('samples_to_target_unique_valid', 'n/a')}"
                    ),
                    (
                        "  Unique valid molecules: "
                        f"{smiles_trial.get('unique_valid_count', 0)}/{smiles_trial.get('sample_count', 0)}"
                    ),
                ]
            )

        anti = self.aux_metrics.get("anti_degeneracy")
        if isinstance(anti, dict):
            lines.extend(
                [
                    "",
                    "Anti-Degeneracy Diagnostics:",
                    f"  Delimiter churn ratio: {anti.get('delimiter_churn_ratio', 0.0):.3f}",
                    f"  Tiny-span rate: {anti.get('tiny_span_rate', 0.0):.1%}",
                    f"  Max-steps hit rate: {anti.get('max_steps_hit_rate', 0.0):.1%}",
                    f"  Applied penalty: {anti.get('penalty', 0.0):.1%}",
                    (
                        "  Membership adjusted by penalty: "
                        f"{anti.get('adjusted_membership_score', self.accuracy):.1%}"
                    ),
                ]
            )

        early_stop = self.aux_metrics.get("early_stop")
        if isinstance(early_stop, dict):
            lines.extend(
                [
                    "",
                    "Early Stop:",
                    f"  Reason: {early_stop.get('reason', 'unknown')}",
                    (
                        "  Max possible accuracy: "
                        f"{float(early_stop.get('max_possible_accuracy', 0.0)):.1%}"
                    ),
                    (
                        "  Target accuracy: "
                        f"{float(early_stop.get('target_accuracy', 0.0)):.1%}"
                    ),
                    (
                        "  Evaluated examples: "
                        f"{early_stop.get('evaluated_examples', 0)}/"
                        f"{early_stop.get('total_examples', self.num_examples)}"
                    ),
                ]
            )

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

        provenance_metrics = self._summarize_provenance_metrics()
        if provenance_metrics:
            lines.append("\nOutput Provenance and Failure Localization:")
            lines.extend(f"  {metric}" for metric in provenance_metrics)

        contrast_metrics = self._summarize_correct_wrong_contrast()
        if contrast_metrics:
            lines.append("\nCorrect-vs-Wrong Behavioral Contrast:")
            lines.extend(f"  {metric}" for metric in contrast_metrics)

        structural_metrics = self._summarize_structural_metrics()
        if structural_metrics:
            lines.append("\nStructural Generation Metrics:")
            lines.extend(f"  {metric}" for metric in structural_metrics)

        snapshots = self._summarize_representative_snapshots()
        if snapshots:
            lines.append("\nRepresentative Factual Snapshots:")
            lines.extend(snapshots)

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

    @staticmethod
    def _redact_artifact_preview(value: Any, max_chars: int = 96) -> str:
        """Return a compact structural preview without preserving dataset-specific text."""
        if value is None:
            return "none"
        text = str(value).replace("\\n", " ").replace("\\r", " ")
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return "empty"
        text = _anonymize_sql_preview(text)
        keep_words = {"true", "false", "yes", "no", "none", "null", "and", "or", "not"}
        text = re.sub(r"[-+]?\d+(?:\.\d+)?", "<num>", text)
        text = re.sub(
            r"(?<!<)\b[A-Za-z_][A-Za-z0-9_]*\b(?!>)",
            lambda m: m.group(0) if m.group(0).lower() in keep_words or m.group(0).upper() in _SQL_KEYWORDS else "<id>",
            text,
        )
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        return text

    @staticmethod
    def _format_counter(counter: Counter[str], denominator: int, max_items: int = 5) -> str:
        from synthesis.evaluate.benchmarks.common.formatting import format_named_counter

        return format_named_counter(
            counter,
            denominator,
            max_items=max_items,
            min_denominator=1,
        )

    @classmethod
    def _helper_counts_for_sample(cls, sample: Dict[str, Any]) -> Counter[str]:
        return Counter(
            event.get("helper", "unknown")
            for event in sample.get("helper_trace") or []
        )

    @classmethod
    def _control_tags_for_sample(cls, sample: Dict[str, Any]) -> List[str]:
        helpers = set(cls._helper_counts_for_sample(sample))
        tags: List[str] = []
        if helpers & cls._UNCONSTRAINED_HELPERS:
            tags.append("free_lm_generation")
        if "UnconstrainedChunk" in helpers:
            tags.append("free_lm_chunking")
        if "EnterObservedConstrainedSpan" in helpers:
            tags.append("observed_span_entry")
        if "OpenConstrainedSpan" in helpers:
            tags.append("explicit_span_entry")
        if helpers & cls._HARD_TOKEN_HELPERS:
            tags.append("hard_token_constrained")
        if helpers & cls._CONFIDENCE_HELPERS:
            tags.append("confidence_gated")
        if helpers & cls._GROUP_OR_ADAPTIVE_HELPERS:
            tags.append("group_or_adaptive_bias")
        if helpers & cls._SAFE_LOGIT_STEP_HELPERS:
            tags.append("safe_logit_step")
        if helpers & cls._SOFT_CONSTRAINED_HELPERS:
            tags.append("soft_then_hard_fallback")
        if helpers & cls._SYMBOL_HELPERS:
            tags.append("symbol_or_chunk_acceptance")
        if helpers & cls._REPAIR_HELPERS:
            tags.append("parser_repair_or_rollback")
        if not any(event.get("helper", "unknown") in cls._CONSTRAINED_HELPERS for event in sample.get("helper_trace") or []):
            tags.append("no_constrained_activity")
        return tags or ["no_trace"]

    @classmethod
    def _derive_answer_provenance(cls, sample: Dict[str, Any]) -> str:
        source = sample.get("answer_source") or "none"
        tags = set(sample.get("provenance_tags") or cls._control_tags_for_sample(sample))
        if source == "last_visible_span":
            if "no_constrained_activity" in tags:
                return "last_visible_span_without_constrained_activity"
            if "explicit_span_entry" in tags:
                return "last_visible_span_after_explicit_entry"
            if "observed_span_entry" in tags:
                return "last_visible_span_after_observed_entry"
            return "last_visible_span_with_constrained_activity"
        if source == "text_fallback":
            return "free_text_fallback"
        if source == "hidden_or_task_extractor":
            if "no_constrained_activity" in tags:
                return "task_extractor_without_constrained_activity"
            return "task_extractor_with_constrained_activity"
        return "no_scored_answer"

    @classmethod
    def _derive_failure_location(cls, sample: Dict[str, Any]) -> str:
        if sample.get("runtime_budget_exceeded"):
            return "time_budget_exceeded"
        if sample.get("error"):
            return "output_record_error"
        if sample.get("is_correct"):
            return "correct"
        if not sample.get("has_extracted_answer", False):
            return "answer_extraction_or_completion"
        if sample.get("hit_max_steps"):
            return "token_budget_exhausted"
        if not sample.get("uses_hidden_chunks"):
            if int(sample.get("num_visible_spans", 0) or 0) == 0:
                return "span_absent"
            if int(sample.get("num_valid_visible_spans", 0) or 0) == 0:
                return "no_valid_visible_span"
            if not sample.get("is_syntax_valid"):
                return "visible_span_syntax"
        if sample.get("is_syntax_valid") and not sample.get("is_correct"):
            return "syntax_valid_semantic_mismatch"
        if cls._sample_has_constrained_activity(sample):
            return "wrong_after_constrained_activity"
        return "wrong_without_constrained_activity"

    @classmethod
    def _annotate_sample_observability(cls, sample: Dict[str, Any]) -> Dict[str, Any]:
        tags = cls._control_tags_for_sample(sample)
        sample["provenance_tags"] = tags
        sample["answer_provenance"] = cls._derive_answer_provenance(sample)
        sample["failure_location"] = cls._derive_failure_location(sample)
        return sample

    def get_provenance_counts(self) -> Dict[str, Counter[str]]:
        """Return neutral provenance/localization buckets for prompt deltas."""
        answer_provenance: Counter[str] = Counter()
        control_tags: Counter[str] = Counter()
        failure_location: Counter[str] = Counter()
        for sample in self.sample_outputs:
            if "provenance_tags" not in sample:
                self._annotate_sample_observability(sample)
            answer_provenance[sample.get("answer_provenance", "unknown")] += 1
            failure_location[sample.get("failure_location", "unknown")] += 1
            for tag in sample.get("provenance_tags") or []:
                control_tags[tag] += 1
        return {
            "answer_provenance": answer_provenance,
            "control_tags": control_tags,
            "failure_location": failure_location,
        }

    def _summarize_provenance_metrics(self) -> List[str]:
        if not self.sample_outputs:
            return []
        n = len(self.sample_outputs)
        counts = self.get_provenance_counts()
        return [
            "Answer provenance: " + self._format_counter(counts["answer_provenance"], n),
            "Control path tags: " + self._format_counter(counts["control_tags"], n),
            "Failure localization: " + self._format_counter(counts["failure_location"], n),
        ]

    def _summarize_correct_wrong_contrast(self) -> List[str]:
        if not self.sample_outputs:
            return []

        def summarize(label: str, samples: List[Dict[str, Any]]) -> str:
            n = len(samples)
            if n == 0:
                return f"{label}: 0 examples"
            provenance = Counter(sample.get("answer_provenance", "unknown") for sample in samples)
            tags: Counter[str] = Counter()
            locations = Counter(sample.get("failure_location", "unknown") for sample in samples)
            for sample in samples:
                for tag in sample.get("provenance_tags") or []:
                    tags[tag] += 1
            avg_tokens = self._mean([float(sample.get("token_count", 0) or 0) for sample in samples]) or 0.0
            avg_valid_spans = self._mean([float(sample.get("num_valid_visible_spans", 0) or 0) for sample in samples]) or 0.0
            syntax_valid = sum(1 for sample in samples if sample.get("is_syntax_valid"))
            return (
                f"{label}: {n} examples; syntax_valid {syntax_valid}/{n}; "
                f"avg_tokens {avg_tokens:.2f}; avg_valid_spans {avg_valid_spans:.2f}; "
                f"provenance {self._format_counter(provenance, n, max_items=3)}; "
                f"control_tags {self._format_counter(tags, n, max_items=4)}; "
                f"locations {self._format_counter(locations, n, max_items=4)}"
            )

        correct = [sample for sample in self.sample_outputs if sample.get("is_correct")]
        wrong = [sample for sample in self.sample_outputs if not sample.get("is_correct")]
        return [summarize("Correct examples", correct), summarize("Wrong examples", wrong)]

    def _summarize_representative_snapshots(self, max_snapshots: int = 4) -> List[str]:
        """Show small redacted factual records for distinct observed failure locations."""
        if not self.sample_outputs:
            return []
        selected: List[tuple[str, Dict[str, Any]]] = []
        seen_locations: set[str] = set()
        priority = [
            "syntax_valid_semantic_mismatch",
            "visible_span_syntax",
            "answer_extraction_or_completion",
            "span_absent",
            "no_valid_visible_span",
            "time_budget_exceeded",
            "wrong_after_constrained_activity",
            "wrong_without_constrained_activity",
        ]
        wrong_samples = [sample for sample in self.sample_outputs if not sample.get("is_correct")]
        for location in priority:
            for sample in wrong_samples:
                if sample.get("failure_location") == location and location not in seen_locations:
                    selected.append((location, sample))
                    seen_locations.add(location)
                    break
            if len(selected) >= max_snapshots:
                break
        if not selected:
            return []

        lines = []
        for location, sample in selected:
            tags = ",".join((sample.get("provenance_tags") or [])[:4]) or "none"
            lines.append(
                "  - "
                f"location={location}; "
                f"syntax={'valid' if sample.get('is_syntax_valid') else 'invalid'}; "
                f"answer_source={sample.get('answer_source', 'none')}; "
                f"provenance={sample.get('answer_provenance', 'unknown')}; "
                f"expected={self._redact_artifact_preview(sample.get('expected'))}; "
                f"actual={self._redact_artifact_preview(sample.get('actual'))}; "
                f"valid_spans={sample.get('num_valid_visible_spans', 0)}/{sample.get('num_visible_spans', 0)}; "
                f"control_tags={tags}"
            )
        return lines

    def get_behavioral_context_summary(self, max_examples: int = 1, max_trace_events: int = 12) -> str:
        traced_examples = [s for s in self.sample_outputs if s.get("helper_trace")]
        if not traced_examples:
            return ""

        lines = ["Recent evaluated behavior from the most recent compiled/evaluated attempt:"]
        for idx, sample in enumerate(traced_examples[:max_examples]):
            trace = sample.get("helper_trace") or []
            counts = Counter(event.get("helper", "unknown") for event in trace)
            lines.append(f"Example {idx + 1}:")
            if "provenance_tags" not in sample:
                self._annotate_sample_observability(sample)
            lines.append(
                f"  Token count: {sample.get('token_count', 'N/A')} | "
                f"Contains << >>: {'yes' if sample.get('contains_delimiters') else 'no'} | "
                f"Syntax rate: {sample.get('syntax_rate', 0.0):.1%} | "
                f"Provenance: {sample.get('answer_provenance', 'unknown')} | "
                f"Location: {sample.get('failure_location', 'unknown')}"
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

    _HARD_TOKEN_HELPERS = {"ConstrainedStep"}
    _CONFIDENCE_HELPERS = {"ConfidenceGatedStep"}
    _GROUP_OR_ADAPTIVE_HELPERS = {"AdaptiveConstrainedStep", "GroupBoostedConstrainedStep"}
    _SAFE_LOGIT_STEP_HELPERS = {
        "SafeBoostedConstrainedStep",
        "SafePenalizedConstrainedStep",
        "SafeRepetitionPenaltyStep",
        "SafeTemperatureConstrainedStep",
    }
    _SOFT_CONSTRAINED_HELPERS = {"SoftConstrainedStep", "SafeSoftConstrainedStep"}
    _SYMBOL_HELPERS = {"ConstrainedSymbol", "ConstrainedSymbolInGenerated"}
    _REPAIR_HELPERS = {
        "RollbackConstrainedSpan",
        "RollbackConstrainedSuffix",
        "RollbackToValidPrefix",
    }
    _CONSTRAINED_HELPERS = {
        "OpenConstrainedSpan",
        "EnterObservedConstrainedSpan",
        "CloseConstrainedSpan",
        "AppendConstrainedToken",
        *_HARD_TOKEN_HELPERS,
        *_CONFIDENCE_HELPERS,
        *_GROUP_OR_ADAPTIVE_HELPERS,
        *_SAFE_LOGIT_STEP_HELPERS,
        *_SOFT_CONSTRAINED_HELPERS,
        *_SYMBOL_HELPERS,
        *_REPAIR_HELPERS,
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
            "accuracy_denominator": self.accuracy_denominator or self.num_examples,
            "accuracy_definition": self.accuracy_definition,
            "invalid_outputs_excluded_from_accuracy": self.invalid_outputs_excluded_from_accuracy,
            "total_time_seconds": self.total_time_seconds,
            "max_sample_time_seconds": self.max_sample_time_seconds,
            "early_stopped": self.early_stopped,
            "early_stop_reason": self.early_stop_reason,
            "planned_num_examples": self.planned_num_examples,
            "error": self.error,
            "sample_outputs": self.sample_outputs,
            "aux_metrics": self.aux_metrics,
        }


class Evaluator:
    """
    Evaluates synthesized CSD strategies on dataset samples.
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
        vllm_max_model_len: int = 16384,
        vllm_enforce_eager: bool = True,
        sample_seed: Optional[int] = None,
        max_seconds_per_example: Optional[float] = None,
        step_token_budget: int = 1,
        gsm_source_dir: str | Path | None = None,
        gsm_split_file: str | Path | None = None,
        gsm_split_name: str = "train",
        spider_split_file: str | Path | None = None,
        spider_split_name: str = "train",
        smiles_classes: Optional[List[str]] = None,
        grammars_dir: str | Path | None = None,
        prompt_tier: int = 2,
    ):
        """
        Initialize the evaluator.

        Args:
            dataset_name: Dataset to evaluate on ("gsm_symbolic", "spider", or "smiles")
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
            gsm_split_file: Optional JSON manifest with train_indices/eval_indices for GSM.
            gsm_split_name: Which split from gsm_split_file to use ("train" or "eval").
            spider_split_file: Optional JSON manifest with train_indices/test_indices for Spider.
            spider_split_name: Which split from spider_split_file to use ("train" or "test").
            prompt_tier: 1 for answer-only (GCD / IterGen / CARS-style) or 2 for chain-of-thought.
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
        from synthesis.evaluate.benchmarks.common.model_utils import resolve_vllm_tensor_parallel_size

        self.vllm_tensor_parallel_size = resolve_vllm_tensor_parallel_size(vllm_tensor_parallel_size)
        self.vllm_pipeline_parallel_size = vllm_pipeline_parallel_size
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.vllm_max_model_len = vllm_max_model_len
        self.vllm_enforce_eager = vllm_enforce_eager
        self.sample_seed = sample_seed
        self.max_seconds_per_example = max_seconds_per_example
        self.step_token_budget = step_token_budget
        self.gsm_source_dir = gsm_source_dir
        self.gsm_split_file = Path(gsm_split_file) if gsm_split_file is not None else None
        self.gsm_split_name = gsm_split_name
        self.spider_split_file = Path(spider_split_file) if spider_split_file is not None else None
        self.spider_split_name = spider_split_name
        self.smiles_classes = smiles_classes
        self.grammars_dir = Path(grammars_dir).expanduser() if grammars_dir is not None else None
        if prompt_tier not in (1, 2):
            raise ValueError(f"prompt_tier must be 1 or 2, got {prompt_tier!r}")
        self.prompt_tier = int(prompt_tier)

        # Lazy-loaded components
        self._dataset = None
        self._env = None
        self._env_cache_key: Optional[tuple[Any, ...]] = None
        self._grammar_file = None
        self._base_grammar_text: Optional[str] = None
        self._dynamic_parser_factory_cache: Dict[Tuple[Any, ...], Any] = {}
        self._syntax_parser_cache: Dict[Tuple[str, ...], Any] = {}

    def unload_runtime(self) -> None:
        """Release cached runtime model state so the generator can reclaim GPU memory."""
        self._env = None
        self._env_cache_key = None
        if self.backend == "vllm":
            try:
                from synthesis.evaluate.benchmarks.common.model_utils import clear_vllm_engine_cache

                clear_vllm_engine_cache()
            except Exception:
                pass
        else:
            import gc

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _read_split_manifest(self, split_file: str | Path) -> dict:
        """Load a benchmark split manifest from a filesystem path."""
        return json.loads(Path(split_file).read_text())

    def _load_gsm_split_indices(self) -> Optional[List[int]]:
        """Load explicit GSM example indices from a train/eval split manifest."""
        if self.gsm_split_file is None:
            return None

        manifest = self._read_split_manifest(self.gsm_split_file)
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

    def _normalize_spider_split_name(self) -> str:
        """Map legacy ``eval`` alias to Spider's ``test`` split field."""
        if self.spider_split_name == "eval":
            return "test"
        return self.spider_split_name

    def _load_spider_split_indices(self) -> Optional[List[int]]:
        """Load explicit Spider example indices from a train/test split manifest."""
        if self.spider_split_file is None:
            return None

        manifest = self._read_split_manifest(self.spider_split_file)
        split_name = self._normalize_spider_split_name()
        key = f"{split_name}_indices"
        if key not in manifest and split_name == "test":
            key = "eval_indices"
        if key not in manifest:
            available = sorted(k for k in manifest.keys() if k.endswith("_indices"))
            raise ValueError(
                f"Split file {self.spider_split_file} does not contain {key}. "
                f"Available index fields: {available}"
            )

        indices = manifest[key]
        if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
            raise ValueError(f"{key} in {self.spider_split_file} must be a list of integers")
        return indices

    def _get_grammar_file(self) -> Path:
        """Get the grammar file path for the dataset."""
        if self._grammar_file is None:
            grammars_dir = self.grammars_dir or Path(
                os.environ.get(
                    "CSD_GRAMMARS_DIR",
                    str(Path(__file__).parent / "grammars"),
                )
            ).expanduser()
            from synthesis.evaluate.benchmarks.registry import get_logic

            logic = get_logic(self.dataset_name)
            self._grammar_file = logic.get_grammar_file(self, grammars_dir)
        return self._grammar_file

    def _normalize_smiles_classes(self) -> List[str]:
        """Return the selected SMILES classes as a normalized list."""
        from synthesis.evaluate.benchmarks.smiles.eval_logic import normalize_classes

        return normalize_classes(self)

    def _get_grammar_text(self) -> str:
        """Load and cache the active grammar text."""
        if self._base_grammar_text is None:
            self._base_grammar_text = self._get_grammar_file().read_text()
        return self._base_grammar_text

    def _get_syntax_parser(self, example: Optional[dict] = None):
        """Create or reuse a syntax parser for one example's allowed variables."""
        logic = self._benchmark_logic()
        return logic.get_syntax_parser(self, example)

    def _load_dataset_sample(self) -> list:
        """Load a sample of the dataset for evaluation."""
        if self._dataset is not None:
            return self._dataset
        from synthesis.evaluate.benchmarks.registry import get_logic

        logic = get_logic(self.dataset_name)
        self._dataset = logic.load_dataset_sample(self)

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
        if run_dir.name in {"generated_csd", "python"}:
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
            from synthesis.evaluate.benchmarks.gsm_symbolic.environment import setup_dafny_environment
        elif self.dataset_name == "spider":
            from synthesis.evaluate.benchmarks.sql_spider.environment import setup_dafny_environment
        elif self.dataset_name == "smiles":
            from synthesis.evaluate.benchmarks.smiles.environment import setup_dafny_environment
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        def _make_env(
            gpu_memory_utilization: float,
            tensor_parallel_size: int | None,
        ) -> Dict[str, Any]:
            return setup_dafny_environment(
                run_dir=run_dir,
                model_name=self.model_name,
                backend=self.backend,
                device=self.device,
                grammar_file=self._get_grammar_file(),
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                vllm_tensor_parallel_size=tensor_parallel_size,
                vllm_pipeline_parallel_size=self.vllm_pipeline_parallel_size,
                vllm_gpu_memory_utilization=gpu_memory_utilization,
                vllm_max_model_len=self.vllm_max_model_len,
                vllm_enforce_eager=self.vllm_enforce_eager,
            )

        if self.backend != "vllm":
            env = _make_env(self.vllm_gpu_memory_utilization, self.vllm_tensor_parallel_size)
            self._env = env
            self._env_cache_key = env_cache_key
            return env

        from synthesis.evaluate.benchmarks.common.model_utils import (
            clear_vllm_engine_cache,
            narrow_cuda_visible_devices_to_index,
            pick_cuda_device_index_with_most_free_memory,
        )

        def _is_vllm_startup_memory_error(exc: Exception) -> bool:
            message = str(exc)
            return (
                "desired GPU memory utilization" in message
                or "Free memory on device" in message
                or "Engine core initialization failed" in message
            )

        try:
            import torch
        except Exception:
            torch = None  # type: ignore[assignment]

        requested_tp = self.vllm_tensor_parallel_size or 1

        tp_candidates: List[int] = []
        for candidate in (requested_tp, 1):
            if candidate >= 1 and candidate not in tp_candidates:
                tp_candidates.append(candidate)

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
        narrowed_visible_devices = False
        for tp in tp_candidates:
            if tp == 1 and tp != requested_tp and not narrowed_visible_devices:
                best_idx = pick_cuda_device_index_with_most_free_memory()
                chosen = narrow_cuda_visible_devices_to_index(best_idx)
                narrowed_visible_devices = True
                print(
                    "Retrying vLLM evaluator startup on a single GPU "
                    f"(CUDA_VISIBLE_DEVICES={chosen})"
                )
                clear_vllm_engine_cache()
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            for util in util_candidates:
                try:
                    if tp != requested_tp and util == util_candidates[0]:
                        print(
                            f"Retrying vLLM evaluator startup with "
                            f"tensor_parallel_size={tp}"
                        )
                    elif util != self.vllm_gpu_memory_utilization:
                        print(
                            f"Retrying vLLM evaluator startup with lower "
                            f"gpu_memory_utilization={util:.2f}"
                        )
                    env = _make_env(util, tp)
                    self._env = env
                    self._env_cache_key = env_cache_key
                    return env
                except Exception as exc:
                    last_error = exc
                    if not _is_vllm_startup_memory_error(exc):
                        raise
                    clear_vllm_engine_cache()
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    is_last_attempt = tp == tp_candidates[-1] and util == util_candidates[-1]
                    if is_last_attempt:
                        raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to initialize evaluation environment.")

    def _extract_constrained_content(self, output: str) -> List[str]:
        """Extract content within << >> delimiters."""
        return re.findall(r"<<\s*([^<>]+?)\s*>>", output)

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
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"int", "ToInt"} and len(node.args) == 1:
                    value = _eval(node.args[0])
                    return float(int(value))
                if node.func.id == "z3_floor_div" and len(node.args) == 2:
                    divisor = _eval(node.args[1])
                    if divisor == 0:
                        return 0.0
                    return float(int(_eval(node.args[0]) / divisor))
            else:
                raise ValueError(f"Unsupported node: {type(node)}")

        try:
            tree = ast.parse(expr.strip(), mode='eval')
            return _eval(tree.body)
        except Exception:
            return None

    def _evaluate_symbolic_expression(self, expr: str, var_values: dict) -> Optional[float]:
        """Substitute variable values into a symbolic expression and evaluate."""
        from synthesis.evaluate.benchmarks.gsm_symbolic.expression_normalize import (
            has_unbound_problem_variables,
            normalize_gsm_symbolic_for_equivalence,
        )

        expr = normalize_gsm_symbolic_for_equivalence(expr)
        substituted = expr
        # Substitute longest names first to avoid partial replacement (n10 before n1)
        for var in sorted(var_values.keys(), key=len, reverse=True):
            substituted = re.sub(r'\b' + re.escape(var) + r'\b',
                                 str(var_values[var]), substituted)
        if has_unbound_problem_variables(substituted):
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

    def _extract_answer_smiles(self, output: str, example: Optional[dict] = None) -> Optional[str]:
        """Extract the generated SMILES string from raw output."""
        from synthesis.evaluate.benchmarks.smiles.metrics import clean_smiles_output

        smiles = clean_smiles_output(output)
        return smiles or None

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

        from synthesis.evaluate.benchmarks.gsm_symbolic.expression_normalize import (
            normalize_gsm_symbolic_for_equivalence,
            reserved_equivalence_names,
        )

        model_expr = normalize_gsm_symbolic_for_equivalence(model_expr)
        expected_expr = normalize_gsm_symbolic_for_equivalence(expected_expr)

        var_names = set(re.findall(r'\b[a-zA-Z_]\w*\b', model_expr + ' ' + expected_expr))
        var_names -= reserved_equivalence_names()

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
            val_model = self._evaluate_symbolic_expression(model_expr, env)
            val_expected = self._evaluate_symbolic_expression(expected_expr, env)
            if val_model is None or val_expected is None:
                return False
            if abs(val_model - val_expected) > 1e-6 * max(1, abs(val_expected)):
                return False
        return True

    def _get_expected_answer(self, example: dict) -> str:
        """Get the expected answer from a dataset example."""
        from synthesis.evaluate.benchmarks.registry import get_logic

        logic = get_logic(self.dataset_name)
        return logic.expected_answer(self, example)

    def _format_prompt(self, example: dict) -> Union[str, List[dict]]:
        """Format a dataset example as a prompt."""
        from synthesis.evaluate.benchmarks.registry import get_logic

        logic = get_logic(self.dataset_name)
        if self.prompt_tier == 1:
            return logic.format_prompt_expression_only(self, example)
        return logic.format_prompt(self, example)

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

        if self.dataset_name == "smiles":
            from synthesis.evaluate.benchmarks.smiles.metrics import evaluate_smiles_output

            class_name = (example or {}).get("class_name", "smiles")
            prompt_exemplars = (example or {}).get("prompt_exemplars", [])
            grammar_text = (example or {}).get("grammar_text", "")
            eval_row = evaluate_smiles_output(
                class_name,
                output,
                grammar_text,
                prompt_exemplars,
                require_rdkit=True,
            )
            smiles = eval_row["smiles"]
            if not smiles:
                return False, []
            return bool(eval_row["syntax_valid"]), [(smiles, bool(eval_row["syntax_valid"]))]

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

    def _ensure_smiles_rdkit_available(self) -> None:
        logic = self._benchmark_logic()
        logic.ensure_runtime_prereqs(self)

    def _compute_smiles_aux_metrics(
        self,
        sample_outputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        logic = self._benchmark_logic()
        return logic.compute_aux_metrics(self, sample_outputs)

    def _benchmark_logic(self):
        from synthesis.evaluate.benchmarks.registry import get_logic

        return get_logic(self.dataset_name)

    def _extract_actual_for_example(
        self,
        scored_output: str,
        example: dict[str, Any],
    ) -> tuple[Optional[str], str, Optional[dict[str, Any]]]:
        logic = self._benchmark_logic()
        actual, answer_source, aux = logic.extract_actual(self, scored_output, example)
        return actual, answer_source, aux

    def _is_correct_for_example(
        self,
        actual: Optional[str],
        expected: str,
        example: dict[str, Any],
        aux: Optional[dict[str, Any]],
        scored_output: str,
    ) -> bool:
        logic = self._benchmark_logic()
        return bool(logic.is_correct(self, actual, expected, example, aux, scored_output))

    def _uses_hidden_chunks(self) -> bool:
        logic = self._benchmark_logic()
        return bool(logic.uses_hidden_chunks())

    def _example_syntax_pass(
        self,
        all_valid_syntax: bool,
        segments: list[tuple[str, bool]],
        used_hidden_chunk: bool,
        aux: Optional[dict[str, Any]],
    ) -> bool:
        logic = self._benchmark_logic()
        return bool(logic.example_syntax_pass(all_valid_syntax, segments, used_hidden_chunk, aux))

    def _accuracy_applicable_for_example(self, aux: Optional[dict[str, Any]]) -> bool:
        logic = self._benchmark_logic()
        return bool(logic.accuracy_applicable(aux))

    def evaluate_sample(
        self,
        compiled_module_path: Path,
        sample_size: Optional[int] = None,
        min_accuracy: Optional[float] = None,
        early_stop_min_accuracy: Optional[float] = None,
        early_stop_min_syntax_rate: Optional[float] = None,
        early_stop_runtime_failures: Optional[int] = None,
        min_examples_before_threshold_stop: Optional[int] = None,
    ) -> EvaluationResult:
        """
        Evaluate the compiled CSD on a sample of the dataset.

        Args:
            compiled_module_path: Path to the compiled GeneratedCSD.py module
            sample_size: Number of examples to evaluate (overrides init value)
            min_accuracy: Backward-compatible target accuracy for early stop.
            early_stop_min_accuracy: Optional target accuracy for early stop.
            early_stop_min_syntax_rate: Optional target syntax rate for early stop.
            early_stop_runtime_failures: Optional runtime-failure count for early stop.
            min_examples_before_threshold_stop: If set, the threshold-impossible
                accuracy and syntax-rate early stops are suppressed until at
                least this many examples have been evaluated. The runtime-budget
                early stop is unaffected (it is a different signal). Lets the
                synthesis feedback loop see usable data even when the strategy
                cannot possibly clear the acceptance threshold.

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
            self._ensure_smiles_rdkit_available()
            dataset = self._load_dataset_sample()
            env = self._setup_environment(compiled_module_path)

            logic = self._benchmark_logic()
            run_crane_csd = logic.get_generation_runner()

            num_correct = 0
            all_examples_contain_delimiters = True
            num_examples_syntax_pass = 0
            num_accuracy_examples = 0
            planned_num_examples = len(dataset)
            target_min_accuracy = (
                early_stop_min_accuracy
                if early_stop_min_accuracy is not None
                else min_accuracy
            )
            early_stop_enabled = (
                target_min_accuracy is not None
                or early_stop_min_syntax_rate is not None
                or early_stop_runtime_failures is not None
            )

            def _accuracy_upper_bound() -> float:
                remaining = max(0, planned_num_examples - len(sample_outputs))
                return logic.accuracy_upper_bound(
                    num_correct,
                    remaining,
                    num_accuracy_examples,
                    planned_num_examples,
                )

            def build_result(early_stop_reason: Optional[str] = None) -> EvaluationResult:
                total_time = time.time() - start_time
                evaluated_count = len(sample_outputs)
                max_sample_time = max(
                    (float(sample.get("time_seconds", 0.0)) for sample in sample_outputs),
                    default=0.0,
                )
                denominator_basis = (
                    planned_num_examples
                    if early_stop_reason and "target accuracy" in early_stop_reason
                    else evaluated_count
                )
                accuracy_denominator = logic.final_accuracy_denominator(
                    denominator_basis,
                    num_accuracy_examples,
                )
                accuracy_definition = logic.accuracy_definition()
                invalid_excluded = logic.invalid_outputs_excluded(
                    evaluated_count,
                    num_accuracy_examples,
                )
                aux_metrics = self._compute_smiles_aux_metrics(sample_outputs)
                if early_stop_reason is not None:
                    aux_metrics["early_stop"] = {
                        "reason": early_stop_reason,
                        "target_accuracy": target_min_accuracy,
                        "target_syntax_rate": early_stop_min_syntax_rate,
                        "max_possible_accuracy": _accuracy_upper_bound(),
                        "evaluated_examples": evaluated_count,
                        "total_examples": planned_num_examples,
                        "remaining_examples": max(0, planned_num_examples - evaluated_count),
                    }
                task_guidance = sorted({
                    sample.get("task_guidance")
                    for sample in sample_outputs
                    if sample.get("task_guidance")
                })
                return EvaluationResult(
                    success=True,
                    accuracy=num_correct / max(1, accuracy_denominator),
                    contains_delimiters=all_examples_contain_delimiters,
                    syntax_rate=num_examples_syntax_pass / max(1, evaluated_count),
                    num_examples=evaluated_count,
                    num_correct=num_correct,
                    accuracy_denominator=accuracy_denominator,
                    accuracy_definition=accuracy_definition,
                    invalid_outputs_excluded_from_accuracy=invalid_excluded,
                    total_time_seconds=total_time,
                    max_sample_time_seconds=max_sample_time,
                    early_stopped=early_stop_reason is not None,
                    early_stop_reason=early_stop_reason,
                    planned_num_examples=planned_num_examples,
                    error=early_stop_reason,
                    sample_outputs=sample_outputs,
                    task_guidance=task_guidance,
                    aux_metrics=aux_metrics,
                )

            def early_stop_reason_if_any() -> Optional[str]:
                if not early_stop_enabled or not sample_outputs:
                    return None

                evaluated_count = len(sample_outputs)
                remaining = planned_num_examples - evaluated_count
                runtime_failures = sum(
                    1 for sample in sample_outputs if sample.get("runtime_budget_exceeded")
                )
                if (
                    early_stop_runtime_failures is not None
                    and runtime_failures >= early_stop_runtime_failures
                ):
                    return (
                        "threshold-impossible early stop: "
                        f"{runtime_failures} example(s) exceeded the per-example runtime budget."
                    )

                # SMILES excludes invalid molecules from the accuracy denominator, so
                # an accuracy upper bound is not comparable until all syntax outcomes
                # are known. Keep this synthesis gate to fixed-denominator tasks.
                if self.dataset_name == "smiles":
                    return None

                # Guard threshold-impossible early stops so the synthesis feedback
                # loop always sees a usable amount of evaluation data. The
                # runtime-failures gate above is intentionally not affected: it
                # signals actual budget exhaustion, not an unreachable target.
                if (
                    min_examples_before_threshold_stop is not None
                    and evaluated_count < min_examples_before_threshold_stop
                ):
                    return None

                if target_min_accuracy is not None:
                    best_possible_accuracy = _accuracy_upper_bound()
                    if best_possible_accuracy < target_min_accuracy:
                        return (
                            "target accuracy unreachable: "
                            f"best possible accuracy is {best_possible_accuracy:.1%} "
                            f"after {evaluated_count}/{planned_num_examples} examples, "
                            f"below required {target_min_accuracy:.1%}."
                        )

                if early_stop_min_syntax_rate is not None:
                    best_possible_syntax = (
                        num_examples_syntax_pass + remaining
                    ) / max(1, planned_num_examples)
                    if best_possible_syntax < early_stop_min_syntax_rate:
                        return (
                            "threshold-impossible early stop: "
                            f"best possible syntax is {best_possible_syntax:.1%} "
                            f"after {evaluated_count}/{planned_num_examples} examples, "
                            f"below required {early_stop_min_syntax_rate:.1%}."
                        )

                return None

            smiles_prompt_states: dict[str, Any] | None = None
            if self.dataset_name == "smiles":
                smiles_prompt_states = logic.init_prompt_states(dataset)

            for i, example in enumerate(dataset):
                print(f"  [EVAL] Processing example {i+1}/{len(dataset)}...", flush=True)
                example_start = time.time()
                if smiles_prompt_states is not None:
                    logic.apply_prompt_state(example, smiles_prompt_states)
                prompt = self._format_prompt(example)
                expected = self._get_expected_answer(example)
                benchmark_aux: Optional[dict[str, Any]] = None

                try:
                    print(f"  [EVAL]   Running CSD strategy (max_steps={self.max_steps})...", flush=True)
                    with _PerExampleTimer(self.max_seconds_per_example):
                        output_text, token_count, gen_time, constrained_segments, helper_trace = run_crane_csd(
                            env=env,
                            prompt_text=prompt,
                            max_steps=self.max_steps,
                            step_token_budget=self.step_token_budget,
                            grammar_file=self._get_grammar_file(),
                            dynamic_parser=logic.build_dynamic_parser(self, env, example),
                        )
                    example_time = time.time() - example_start
                    print(f"  [EVAL]   Generated {token_count} tokens in {example_time:.2f}s", flush=True)

                    from synthesis.evaluate.completion_text import completion_for_scoring

                    completion = completion_for_scoring(prompt, output_text)
                    scored_output = (
                        self._truncate_gsm_output(completion)
                        if self.dataset_name == "gsm_symbolic"
                        else completion
                    )

                    actual, answer_source, benchmark_aux = self._extract_actual_for_example(scored_output, example)
                    if smiles_prompt_states is not None:
                        benchmark_aux = logic.record_prompt_result(
                            example,
                            smiles_prompt_states,
                            actual or "",
                            benchmark_aux,
                        )
                    is_correct = self._is_correct_for_example(
                        actual,
                        expected,
                        example,
                        benchmark_aux,
                        scored_output,
                    )

                    visible_delimiters = self._contains_delimiters(scored_output)
                    used_hidden_chunk = bool(constrained_segments) or any(
                        event.get("helper") in EvaluationResult._CONSTRAINED_HELPERS
                        for event in (helper_trace or [])
                    )
                    contains_delimiters = used_hidden_chunk if self._uses_hidden_chunks() else visible_delimiters
                    all_examples_contain_delimiters = (
                        all_examples_contain_delimiters and contains_delimiters
                    )

                    all_valid_syntax, segments = self._check_syntax_validity(scored_output, example=example)
                    # Per-example syntax pass:
                    # - GSM: visible <<...>> chunks must exist and parse.
                    # - SMILES: the full output is the generated molecule string.
                    # - Spider: chunks are internal/hidden; visible delimiter tokens are not
                    #   part of the answer contract, so count parser-governed chunk usage.
                    example_syntax_pass = self._example_syntax_pass(
                        all_valid_syntax,
                        segments,
                        used_hidden_chunk,
                        benchmark_aux,
                    )
                    accuracy_applicable = self._accuracy_applicable_for_example(benchmark_aux)
                    if accuracy_applicable:
                        num_accuracy_examples += 1
                    if is_correct:
                        num_correct += 1
                    num_examples_syntax_pass += int(example_syntax_pass)
                    example_syntax_rate = 1.0 if example_syntax_pass else 0.0
                    if self._uses_hidden_chunks() and self.dataset_name == "smiles":
                        visible_span_lengths = [len((benchmark_aux or {}).get("smiles", "").split())] if actual else []
                        valid_visible_span_lengths = visible_span_lengths if example_syntax_pass else []
                        num_valid_visible_spans = 1 if example_syntax_pass and actual else 0
                        segments = [(actual or "", example_syntax_pass)] if actual else []
                    else:
                        visible_span_lengths = [
                            len(segment.strip().split()) for segment, _ in segments
                        ]
                        valid_visible_span_lengths = [
                            len(segment.strip().split())
                            for segment, is_valid in segments
                            if is_valid
                        ]
                        num_valid_visible_spans = sum(1 for _, is_valid in segments if is_valid)

                    from synthesis.evaluate.baseline_store import normalize_baseline_question

                    full_question = normalize_baseline_question(
                        self.dataset_name, example=example
                    )
                    prompt_text = (
                        prompt if isinstance(prompt, str) else str(prompt)
                    )
                    sample = {
                        "question": full_question,
                        "prompt": prompt_text,
                        "generated": completion,
                        "expected": expected,
                        "actual": actual or completion[:100],
                        "full_output": completion,
                        "scored_output": scored_output,
                        "answer_source": answer_source,
                        "has_extracted_answer": actual is not None or answer_source == "text_fallback",
                        "is_correct": is_correct,
                        "accuracy_applicable": accuracy_applicable,
                        "contains_delimiters": contains_delimiters,
                        "visible_delimiters": visible_delimiters,
                        "used_constrained_chunk": used_hidden_chunk,
                        "uses_hidden_chunks": self._uses_hidden_chunks(),
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
                        "task_guidance": getattr(env.get("lm"), "task_guidance", None),
                        "smiles_eval": benchmark_aux if self.dataset_name == "smiles" else None,
                    }
                    if self.dataset_name == "smiles":
                        sample["smiles_eval"] = benchmark_aux
                    sample_outputs.append(EvaluationResult._annotate_sample_observability(sample))
                    early_reason = early_stop_reason_if_any()
                    if early_reason:
                        print(f"  [EVAL] Early stopping synthesis eval: {early_reason}", flush=True)
                        return build_result(early_reason)

                except Exception as e:
                    from synthesis.evaluate.baseline_store import normalize_baseline_question

                    full_question = normalize_baseline_question(
                        self.dataset_name, example=example
                    )
                    prompt_text = (
                        prompt if isinstance(prompt, str) else str(prompt)
                    )
                    elapsed = time.time() - example_start
                    timed_out = isinstance(e, PerExampleTimeout)
                    if timed_out:
                        print(f"  [EVAL]   Timed out after {elapsed:.2f}s", flush=True)
                    sample = {
                        "question": full_question,
                        "prompt": prompt_text,
                        "generated": "",
                        "expected": expected,
                        "actual": None,
                        "full_output": "",
                        "scored_output": "",
                        "answer_source": "none",
                        "has_extracted_answer": False,
                        "is_correct": False,
                        "accuracy_applicable": self._accuracy_applicable_for_example(None),
                        "contains_delimiters": False,
                        "visible_delimiters": False,
                        "used_constrained_chunk": False,
                        "uses_hidden_chunks": self._uses_hidden_chunks(),
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
                        "helper_trace": [],
                        "task_guidance": getattr(env.get("lm"), "task_guidance", None),
                    }
                    sample_outputs.append(EvaluationResult._annotate_sample_observability(sample))
                    all_examples_contain_delimiters = False
                    early_reason = early_stop_reason_if_any()
                    if early_reason:
                        print(f"  [EVAL] Early stopping synthesis eval: {early_reason}", flush=True)
                        return build_result(early_reason)
                    if timed_out and early_stop_enabled:
                        reason = (
                            "Evaluation stopped early because one example exceeded "
                            f"the {self.max_seconds_per_example:.2f}s runtime budget."
                        )
                        print(f"  [EVAL] Early stopping eval: {reason}", flush=True)
                        return build_result(reason)

            return build_result()

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
                task_guidance=sorted({
                    sample.get("task_guidance")
                    for sample in sample_outputs
                    if sample.get("task_guidance")
                }),
            )
