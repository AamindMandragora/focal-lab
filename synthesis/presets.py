"""
Shared synthesis presets for dataset-specific helper scripts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SynthesisPreset:
    """Dataset-specific defaults for ``run_synthesis.py`` wrappers."""

    dataset: str
    output_name: str
    task_description: str
    min_accuracy: float
    min_format_rate: float
    min_syntax_rate: float
    eval_sample_size: int
    eval_max_steps: int

    def to_cli_args(
        self,
        *,
        model_name: str,
        eval_model_name: str,
        max_iterations: int,
        temperature: float,
        device: str,
        output_name: str | None = None,
        max_tokens: int | None = None,
        generation_timeout: int | None = None,
        min_accuracy: float | None = None,
        min_format_rate: float | None = None,
        min_syntax_rate: float | None = None,
        eval_sample_size: int | None = None,
        eval_max_steps: int | None = None,
    ) -> list[str]:
        """Build ``run_synthesis.py`` CLI arguments for this preset."""
        return [
            "--task",
            self.task_description,
            "--dataset",
            self.dataset,
            "--output-name",
            output_name or self.output_name,
            "--model",
            model_name,
            "--eval-model",
            eval_model_name,
            "--temperature",
            str(temperature),
            "--device",
            device,
            "--max-tokens",
            str(1200 if max_tokens is None else max_tokens),
            "--generation-timeout",
            str(0 if generation_timeout is None else generation_timeout),
            "--max-iterations",
            str(max_iterations),
            "--min-accuracy",
            str(self.min_accuracy if min_accuracy is None else min_accuracy),
            "--min-format-rate",
            str(self.min_format_rate if min_format_rate is None else min_format_rate),
            "--min-syntax-rate",
            str(self.min_syntax_rate if min_syntax_rate is None else min_syntax_rate),
            "--eval-sample-size",
            str(self.eval_sample_size if eval_sample_size is None else eval_sample_size),
            "--eval-max-steps",
            str(self.eval_max_steps if eval_max_steps is None else eval_max_steps),
        ]


MODEL_PRESETS = {
    "gpt54": "gpt-5.4",
    "qwen3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
}

DEFAULT_GENERATION_MODEL_PRESET = "gpt54"
DEFAULT_EVAL_MODEL_PRESET = "qwen7b"


DATASET_PRESETS = {
    "gsm_symbolic": SynthesisPreset(
        dataset="gsm_symbolic",
        output_name="gsm_crane_csd",
        task_description=(
            "Generate short symbolic mathematical expressions for GSM-Symbolic "
            "reasoning. The parser enforces a strict arithmetic expression grammar "
            "with numeric constants and optional variables. CRITICAL RULES: "
            "1. Any content intended for parser/evaluator handling must appear "
            "inside << >> delimiters, and every delimited span must be "
            "grammar-valid. Pure arithmetic expressions are valid; the final "
            "<< >> segment may be either a single expression or a single equation. "
            "2. Prefer natural delimiter strategies with a small helper surface. "
            "Use delimiter-masked AppendUnconstrainedStep for ordinary reasoning. "
            "After an explicit final-answer cue, scratch-to-final transition, or "
            "real budget pressure, use AppendUnconstrainedNudgeLeftDelimiterStep "
            "until helpers.EndsWithLeftDelimiter(generated) is true. Inside the "
            "natural opening phase, start nudging with enough remaining budget "
            "for several attempts; do not wait for tiny thresholds like "
            "`stepsLeft <= 4` or `not helpers.HasBudget(stepsLeft, 6)`. "
            "Inside the "
            "span, use AppendConstrainedOrRightDelimiterStep until "
            "helpers.EndsWithRightDelimiter(generated) is true. Do not use plain "
            "AppendConstrainedStep in natural delimiter mode; it cannot emit "
            "the closing delimiter. Do not break on `not helpers.CanConstrain` "
            "before checking helpers.IsComplete, because complete suffixes are "
            "exactly when `>>` may be emitted. Prefer the positive span guard "
            "`helpers.IsComplete(generated) or helpers.CanConstrain(generated)`, "
            "then call AppendConstrainedOrRightDelimiterStep. Track durable open-span state "
            "such as `phase` or `inside_span`; helpers.EndsWithLeftDelimiter "
            "is an opening event, not the whole inside-span condition. Do not "
            "open immediately after a "
            "generic first word like To or The. "
            "Multiple verified spans are allowed, but for GSM prefer one delayed "
            "final expression unless earlier spans are named scratch assignments "
            "that the final span reuses. A strong observed GSM policy is a single "
            "late final span: keep delimiters masked for substantial free-form "
            "reasoning, run a short wrap-up / answer-cue phase, then repeatedly "
            "nudge for a natural left delimiter and stop after that first closed "
            "final span. This is preferred over forced scratch spans unless the "
            "strategy deliberately reuses named scratch assignments. If relying "
            "mostly on counters instead of a clear final-answer cue, keep the "
            "answer-ready threshold around forty-plus reasoning/setup steps and "
            "include a short wrap-up phase before nudging. Do not use "
            "parser-distance or valid-continuation predicates as finality signals "
            "after only a short setup, and do not switch to an open/span phase "
            "then immediately break before a helper step emits the final span. "
            "3. Prefer a compact complete arithmetic expression in the final "
            "span; the evaluator computes it, and expressions are less brittle "
            "than forcing a fully simplified numeral. The GSM CSD grammar rejects "
            "lone numerals such as 1 or 8 and first-operation fragments such as "
            "16 * 8; emit an expression with at least one top-level "
            "plus/minus clause, e.g. 8 + 0, when the answer is directly known. "
            "Completion means the strategy may close the span, not that it must close "
            "immediately if continuing the expression is still semantically useful. "
            "4. Use variables only when they genuinely help. Dataset variables "
            "must have prompt bindings; optional scratch variables such as x_1 "
            "or total_1 may be introduced by earlier complete delimited "
            "assignments like <<x_1 = 48 / 2>> and then reused later. Pure "
            "numeric expressions like 16 * 8.5 + 4 * 10.5 + 13 are allowed. "
            "5. Preserve numeric values exactly from the problem statement. Do "
            "not round or truncate decimals like 8.5 into 8, do not replace "
            "values like 13 with 1, and prefer copying all relevant numbers "
            "from the question into the expression before simplifying. "
            "6. The constrained answer segment should stay short, compact, and "
            "mathematically meaningful. "
            "7. The final constrained segment must be complete before the closing "
            "delimiter. Prefer helpers.IsComplete(generated), "
            "helpers.ValidContinuationCount(generated), and "
            "helpers.ParserDistanceToComplete(generated) over direct parser calls. "
            "After emitting the right delimiter, "
            "either stop if this was the final answer span, or continue delimiter-masked "
            "free-form reasoning before opening another verified span. In the "
            "answer phase, test helpers.IsComplete(generated) before open-ended "
            "CanConstrain branches, or add an explicit not-complete guard, so "
            "complete expressions cannot extend forever. "
            "8. Multiple independent delimited verified spans are encouraged when "
            "they emerge naturally from the reasoning. The final << >> span is the "
            "graded answer. Prefer natural interleaving like reasoning text, then "
            "a complete scratch assignment span, more reasoning, another scratch "
            "span, and finally an aggregated answer span. Do not force a fixed "
            "template or fixed span count. Strongly prefer final spans that "
            "compose earlier scratch values and remaining constants rather than "
            "stopping after the first local complete span."
        ),
        min_accuracy=0.5,
        min_format_rate=1.0,
        min_syntax_rate=1.0,
        eval_sample_size=10,
        eval_max_steps=300,
    ),
    "folio": SynthesisPreset(
        dataset="folio",
        output_name="folio_csd",
        task_description=(
            "Generate first-order logic formulas for FOLIO reasoning. The parser "
            "enforces a strict FOL grammar with quantifiers, predicates, constants, "
            "and logical connectives. CRITICAL RULES: "
            "1. Quantifiers use {forall} and {exists} with single lowercase "
            "variables. "
            "2. Predicates are uppercase/camel-case and constants are lowercase. "
            "3. Use simple well-typed formulas over overly deep nesting. "
            "4. Parentheses must stay balanced and formulas must be complete. "
            "5. The final constrained segment should contain the answer-bearing "
            "formula only."
        ),
        min_accuracy=0.5,
        min_format_rate=0.8,
        min_syntax_rate=0.8,
        eval_sample_size=10,
        eval_max_steps=1500,
    ),
    "pddl": SynthesisPreset(
        dataset="pddl",
        output_name="pddl_csd",
        task_description=(
            "Generate a PDDL planning strategy for Blocks World problems. The "
            "grammar enforces a strict sequence of actions: (pick-up X), "
            "(put-down X), (stack X Y), (unstack X Y). CRITICAL RULES: "
            "1. The constrained answer contains only actions. "
            "2. Actions must satisfy their preconditions. "
            "3. Plans are evaluated by simulation and should achieve the stated "
            "goal. "
            "4. Prefer short correct plans over verbose ones."
        ),
        min_accuracy=0.3,
        min_format_rate=0.5,
        min_syntax_rate=0.5,
        eval_sample_size=5,
        eval_max_steps=128,
    ),
    "sygus_slia": SynthesisPreset(
        dataset="sygus_slia",
        output_name="sygus_slia_csd",
        task_description=(
            "Generate a string-manipulation strategy for SyGuS SLIA problems. "
            "The grammar enforces a strict S-expression format using SLIA string "
            "operations and integer arithmetic. CRITICAL RULES: "
            "1. The constrained answer is a single complete S-expression. "
            "2. Variables are bare identifiers and string literals are quoted. "
            "3. Integer arguments must be valid integer expressions. "
            "4. Prefer compact correct expressions over unnecessary nesting."
        ),
        min_accuracy=0.3,
        min_format_rate=0.5,
        min_syntax_rate=0.5,
        eval_sample_size=5,
        eval_max_steps=256,
    ),
    "spider": SynthesisPreset(
        dataset="spider",
        output_name="spider_sql_csd",
        task_description=(
            "Generate SQL queries for the Spider text-to-SQL benchmark. The parser "
            "enforces a SQL SELECT-query grammar. CRITICAL RULES: "
            "1. The final constrained segment is a single SQL query inside << >>. "
            "2. Prefer compact SELECT statements over verbose explanations in the "
            "constrained answer span. "
            "3. Use schema/table/column names from the prompt exactly when available. "
            "4. Use standard SQL clauses such as SELECT, FROM, WHERE, GROUP BY, "
            "HAVING, ORDER BY, LIMIT, JOIN, and nested SELECT only when needed. "
            "5. Do not put prose, markdown, or semicolon-only filler inside the final "
            "constrained segment. "
            "6. Prefer one deliberate SQL answer span: short free-form lead-in (if any), "
            "then emit << SQL >> and close it; avoid long unconstrained rambles that "
            "never reach a closed span."
        ),
        min_accuracy=0.2,
        min_format_rate=0.8,
        min_syntax_rate=0.8,
        eval_sample_size=5,
        eval_max_steps=512,
    ),
}


def get_synthesis_preset(dataset: str) -> SynthesisPreset:
    """Return the preset for a dataset or raise a helpful error."""
    try:
        return DATASET_PRESETS[dataset]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_PRESETS))
        raise ValueError(f"Unknown synthesis preset '{dataset}'. Expected one of: {supported}") from exc


def resolve_model_name(model: str | None = None, model_preset: str | None = None) -> str:
    """Resolve a model alias, preferring an explicit model name."""
    if model:
        return model
    preset_name = model_preset or DEFAULT_GENERATION_MODEL_PRESET
    try:
        return MODEL_PRESETS[preset_name]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"Unknown model preset '{preset_name}'. Expected one of: {supported}") from exc
