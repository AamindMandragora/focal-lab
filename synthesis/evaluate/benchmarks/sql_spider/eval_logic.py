"""Spider evaluation logic delegated from the global evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from synthesis.evaluate.benchmarks.common import benchmark_defaults as defaults
from synthesis.evaluate.benchmarks.common.delimited_output import extract_sql_scored_output


uses_hidden_chunks = defaults.uses_hidden_chunks
example_syntax_pass = defaults.example_syntax_pass_from_segments
accuracy_applicable = defaults.accuracy_applicable_always
accuracy_upper_bound = defaults.accuracy_upper_bound_with_remaining
final_accuracy_denominator = defaults.final_accuracy_denominator_all_examples
invalid_outputs_excluded = defaults.invalid_outputs_excluded_none
accuracy_definition = defaults.accuracy_definition_standard


def get_grammar_file(evaluator: Any, grammars_dir: Path) -> Path:
    return grammars_dir / "sql.lark"


def load_dataset_sample(evaluator: Any) -> list[dict[str, Any]]:
    from synthesis.evaluate.benchmarks.sql_spider.dataset import load_spider

    split_indices = evaluator._load_spider_split_indices()
    ds = load_spider(
        source="auto",
        limit=evaluator.sample_size,
        random_sample=split_indices is None,
        seed=evaluator.sample_seed,
        indices=split_indices,
    )
    return list(ds)


def format_prompt(evaluator: Any, example: dict[str, Any]) -> str:
    """Tier-2 chain-of-thought Spider prompt."""
    from synthesis.evaluate.prompt_tiers import render_benchmark_prompt

    return render_benchmark_prompt("spider", tier=2, example=example)


def format_prompt_expression_only(evaluator: Any, example: dict[str, Any]) -> str:
    """Tier-1 answer-only Spider prompt."""
    from synthesis.evaluate.prompt_tiers import render_benchmark_prompt

    return render_benchmark_prompt("spider", tier=1, example=example)


def format_prompt_chain_of_thought(evaluator: Any, example: dict[str, Any]) -> str:
    """Tier-2 chain-of-thought Spider prompt."""
    return format_prompt(evaluator, example)


def expected_answer(evaluator: Any, example: dict[str, Any]) -> str:
    return (example.get("query") or "").strip()


def build_dynamic_parser(evaluator: Any, env: dict[str, Any], example: dict[str, Any]):
    return None


def extract_actual(evaluator: Any, scored_output: str, example: dict[str, Any]) -> tuple[str | None, str, dict[str, Any] | None]:
    actual, source = extract_sql_scored_output(scored_output)
    return actual, source, None


def is_correct(
    evaluator: Any,
    actual: str | None,
    expected: str,
    example: dict[str, Any],
    aux: dict[str, Any] | None,
    scored_output: str,
) -> bool:
    if not actual or not expected:
        return False
    from synthesis.evaluate.benchmarks.sql_spider.executor import prediction_matches_gold

    return prediction_matches_gold(actual, example)


def get_generation_runner():
    from synthesis.evaluate.benchmarks.sql_spider.generation import run_crane_csd

    return run_crane_csd


def get_syntax_parser(evaluator: Any, example: dict[str, Any] | None):
    from lark import Lark

    return Lark(evaluator._get_grammar_text(), start="start", parser="lalr")


def ensure_runtime_prereqs(evaluator: Any) -> None:
    return None


def compute_aux_metrics(evaluator: Any, sample_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {}
