"""Spider evaluation logic delegated from the global evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from synthesis.evaluate.benchmarks.common import benchmark_defaults as defaults
from synthesis.evaluate.benchmarks.common.delimited_output import extract_sql_scored_output
from synthesis.evaluate.benchmarks.sql_spider.prompts import format_spider_messages, format_spider_prompt


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
    # Flattened few-shot format for synthesis: concise output keeps step budget
    # well within limits so constrained <<SQL>> spans can complete.
    # The inline few-shot example is LOAD-BEARING: a zero-shot IterGen-aligned
    # prompt (no example) made both Qwen Instruct models stop emitting << >>
    # entirely -> syntax collapsed to 0.7%/3.3% and accuracy fell 57.3->43.7 (7B)
    # and 44->20.7 (1.5B), confirmed on seed334 held-out 300 on 2026-06-05.
    # (Multi-turn lifted unconstrained 38%->44% but exhausts max_steps in
    # constrained mode and produces 0%/0% — confirmed 2026-05-28.)
    return format_spider_prompt(
        example,
        instruction=(
            "Write ONE SQL query using ONLY tables and columns shown in the schema.\n\n"
            "Return exactly one line: `SQL: <<YOUR QUERY>>`."
        ),
        few_shot_answer_line="SQL: <<SELECT count(*) FROM singer>>",
    )


def format_prompt_expression_only(evaluator: Any, example: dict[str, Any]) -> str:
    """Hard-mask / constrained decoders: emit only ``SQL: <<query>>``."""
    return format_spider_prompt(
        example,
        instruction=(
            "Write ONE SQL query using ONLY tables and columns shown in the schema.\n\n"
            "Return exactly one line: `SQL: <<YOUR QUERY>>`."
        ),
        few_shot_answer_line="SQL: <<SELECT count(*) FROM singer>>",
    )


def format_prompt_chain_of_thought(evaluator: Any, example: dict[str, Any]) -> list[dict]:
    """Legacy CRANE-style runs: require explicit reasoning before the delimited query."""
    return format_spider_messages(
        example,
        instruction=(
            "Write a SINGLE SQL query answering the question, using ONLY the tables "
            "and columns in the schema.\n\n"
            "Reason step by step (tables, joins, filters). "
            "Then output SQL: followed by your query wrapped in << >>. "
            "Stop after the closing >>."
        ),
        few_shot_answer_line=(
            "Let's think step by step. We only need the singer table. "
            "SQL: <<SELECT count(*) FROM singer>>"
        ),
    )


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
