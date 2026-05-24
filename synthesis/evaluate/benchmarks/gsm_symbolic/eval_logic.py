"""GSM-Symbolic evaluation logic delegated from the global evaluator."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from synthesis.evaluate.benchmarks.common import benchmark_defaults as defaults
from synthesis.evaluate.benchmarks.common.delimited_output import extract_last_delimited_span

uses_hidden_chunks = defaults.uses_hidden_chunks
example_syntax_pass = defaults.example_syntax_pass_from_segments
accuracy_applicable = defaults.accuracy_applicable_always
accuracy_upper_bound = defaults.accuracy_upper_bound_with_remaining
final_accuracy_denominator = defaults.final_accuracy_denominator_all_examples
invalid_outputs_excluded = defaults.invalid_outputs_excluded_none
accuracy_definition = defaults.accuracy_definition_standard


def get_grammar_file(evaluator: Any, grammars_dir: Path) -> Path:
    return grammars_dir / "gsm.lark"


def load_dataset_sample(evaluator: Any) -> list[dict[str, Any]]:
    from synthesis.evaluate.benchmarks.gsm_symbolic.dataset import (
        load_gsm_from_crane_folder,
        load_gsm_symbolic,
    )
    from synthesis.project_defaults import default_gsm_source_dir

    indices = evaluator._load_gsm_split_indices()

    crane_dir = evaluator.gsm_source_dir
    if crane_dir is None and indices is not None:
        crane_dir = default_gsm_source_dir()

    # Local CRANE JSONs are required when a fixed split manifest is in use.
    if crane_dir is not None:
        ds = load_gsm_from_crane_folder(
            crane_dir=crane_dir,
            limit=evaluator.sample_size,
            indices=indices,
        )
        return list(ds)

    # Default path: HuggingFace GSM-Symbolic.
    split = "train" if evaluator.gsm_split_name == "train" else "test"

    if indices is not None:
        needed = max(indices) + 1 if indices else 0
        ds = load_gsm_symbolic(
            config="main",
            split=split,
            limit=needed if needed > 0 else None,
            random_sample=False,
        )
        selected: list[dict[str, Any]] = []
        for idx in indices:
            if idx < 0 or idx >= len(ds):
                raise IndexError(
                    f"GSM split index {idx} is out of range for HF split "
                    f"{split} (size={len(ds)})."
                )
            selected.append(ds[idx])
        if evaluator.sample_size is not None and evaluator.sample_size > 0:
            selected = selected[: evaluator.sample_size]
        print(
            f"Loaded {len(selected)} examples from HuggingFace GSM-Symbolic "
            f"(split={split}) using split manifest indices"
        )
        return selected

    ds = load_gsm_symbolic(
        config="main",
        split=split,
        limit=evaluator.sample_size,
        random_sample=evaluator.sample_seed is not None,
        seed=evaluator.sample_seed,
    )
    return list(ds)


def _gsm_question_text(example: dict[str, Any]) -> str:
    # Prefer symbolic `{placeholder}` text over instantiated HF `question` when both exist.
    return (
        example.get("question_parsed")
        or example.get("original_question")
        or example.get("question", "")
    )


def format_prompt(evaluator: Any, example: dict[str, Any]) -> str:
    """Tier-2 chain-of-thought prompt (Unconstrained, CRANE, MetaDecode)."""
    from synthesis.evaluate.prompt_tiers import render_benchmark_prompt

    return render_benchmark_prompt("gsm_symbolic", tier=2, example=example)


def format_prompt_expression_only(evaluator: Any, example: dict[str, Any]) -> str:
    """Tier-1 answer-only prompt (GCD, IterGen, CARS)."""
    from synthesis.evaluate.prompt_tiers import render_benchmark_prompt

    return render_benchmark_prompt("gsm_symbolic", tier=1, example=example)


def format_prompt_chain_of_thought(evaluator: Any, example: dict[str, Any]) -> str:
    """Tier-2 chain-of-thought prompt."""
    return format_prompt(evaluator, example)


def expected_answer(evaluator: Any, example: dict[str, Any]) -> str:
    symbolic = example.get("answer_parsed", "")
    if symbolic:
        return symbolic
    answer_str = example.get("answer", "")
    match = re.search(r"####\s*([-+]?\d*\.?\d+)", answer_str)
    if match:
        return match.group(1)
    return answer_str


def build_dynamic_parser(evaluator: Any, env: dict[str, Any], example: dict[str, Any]):
    from synthesis.evaluate.benchmarks.common.parser_utils import create_lark_dafny_parser
    from synthesis.evaluate.benchmarks.gsm_symbolic.grammar import (
        build_dynamic_grammar,
        extract_variables_from_mapping,
    )

    variable_types = example.get("variable_types") or {}
    if not isinstance(variable_types, dict):
        return None
    allowed_variables = extract_variables_from_mapping(variable_types)
    if not allowed_variables:
        return None

    cache_key = tuple(sorted(allowed_variables))
    parser_factory = evaluator._dynamic_parser_factory_cache.get(cache_key)
    if parser_factory is None:
        grammar_text = build_dynamic_grammar(evaluator._get_grammar_text(), list(cache_key))
        parser_factory = create_lark_dafny_parser(
            grammar_text,
            env["VerifiedDecoderAgent"],
            env["_dafny"],
            start="csd_start",
            tokenizer=env["tokenizer"],
        )
        evaluator._dynamic_parser_factory_cache[cache_key] = parser_factory

    return parser_factory(env["lm"]._Tokens)


def extract_actual(evaluator: Any, scored_output: str, example: dict[str, Any]) -> tuple[str | None, str, dict[str, Any] | None]:
    actual, found = extract_last_delimited_span(scored_output)
    if found:
        return actual, "last_visible_span", None
    lines = (scored_output or "").strip().splitlines()
    body = lines[0].strip() if lines else ""
    if body:
        return body, "constrained_body", None
    return None, "none", None


def is_correct(
    evaluator: Any,
    actual: str | None,
    expected: str,
    example: dict[str, Any],
    aux: dict[str, Any] | None,
    scored_output: str,
) -> bool:
    vt = example.get("variable_types", {})
    if isinstance(vt, str):
        try:
            vt = ast.literal_eval(vt)
        except (ValueError, SyntaxError):
            vt = {}
    if not isinstance(vt, dict):
        vt = {}
    if vt and example.get("answer_parsed"):
        return evaluator._gsm_symbolic_equivalence(actual, expected, vt)
    numeric_actual = evaluator._extract_answer_gsm(scored_output)
    numeric_expected = re.search(r"####\s*([-+]?\d*\.?\d+)", example.get("answer", ""))
    if numeric_expected:
        return evaluator._answers_match(numeric_actual, numeric_expected.group(1))
    return evaluator._answers_match(numeric_actual, expected)


def get_generation_runner():
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import run_crane_csd

    return run_crane_csd


def get_syntax_parser(evaluator: Any, example: dict[str, Any] | None):
    from lark import Lark
    from synthesis.evaluate.benchmarks.gsm_symbolic.grammar import (
        build_dynamic_grammar,
        build_numeric_only_grammar,
        extract_variables_from_mapping,
    )

    if example is None:
        return Lark(evaluator._get_grammar_text(), start="start", parser="lalr")

    variable_types = example.get("variable_types") or {}
    if not isinstance(variable_types, dict):
        grammar_text = build_numeric_only_grammar(evaluator._get_grammar_text())
        return Lark(grammar_text, start="start", parser="lalr")
    allowed_variables = extract_variables_from_mapping(variable_types)
    if not allowed_variables:
        grammar_text = build_numeric_only_grammar(evaluator._get_grammar_text())
        return Lark(grammar_text, start="start", parser="lalr")

    cache_key = tuple(sorted(allowed_variables))
    parser = evaluator._syntax_parser_cache.get(cache_key)
    if parser is None:
        grammar_text = build_dynamic_grammar(evaluator._get_grammar_text(), list(cache_key))
        parser = Lark(grammar_text, start="start", parser="lalr")
        evaluator._syntax_parser_cache[cache_key] = parser
    return parser


def ensure_runtime_prereqs(evaluator: Any) -> None:
    return None


def compute_aux_metrics(evaluator: Any, sample_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {}


