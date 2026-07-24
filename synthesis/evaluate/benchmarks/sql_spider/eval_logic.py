"""Spider evaluation logic delegated from the global evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from synthesis.evaluate.benchmarks.common import benchmark_defaults as defaults
from synthesis.evaluate.benchmarks.common.delimited_output import extract_sql_scored_output
from synthesis.evaluate.benchmarks.sql_spider.prompts import (
    format_spider_itergen_aligned_prompt,
    format_spider_messages,
    format_spider_prompt,
)


def _token0_enabled() -> bool:
    """Spider no-delimiter / token-0-constrained mode. DEFAULT ON (2026-06-22):
    the whole answer is grammar-constrained from the first token with NO visible
    << >> delimiters — the IterGen-aligned decoding surface. Set
    SPIDER_TOKEN0_CONSTRAINED=0 to opt back into the legacy visible-<<>>-span path
    (needed to reproduce the pre-2026-06-22 accepted-board strategies, which force
    << via OpenConstrainedSpan). GSM is a separate benchmark and is unaffected.
    """
    import os

    return os.environ.get("SPIDER_TOKEN0_CONSTRAINED", "1") != "0"


def uses_hidden_chunks() -> bool:
    # Token-0-constrained (no << >>) mode: the whole output is parser-governed,
    # so chunk usage is "hidden" (no visible delimiter tokens). Mirrors how the
    # SMILES benchmark treats its single constrained span.
    return _token0_enabled()


def example_syntax_pass(
    all_valid_syntax: bool,
    segments: list,
    used_hidden_chunk: bool,
    aux: dict | None,
) -> bool:
    if _token0_enabled():
        # No << >> spans exist to extract in token-0 mode, so `segments` is empty
        # and the default `bool(segments) and all_valid_syntax` would score 0%.
        # Credit the grammar-parse result computed over the EXTRACTED SQL in
        # extract_actual instead (does not touch the grammar or correctness grader).
        return bool(aux and aux.get("syntax_valid"))
    return bool(segments) and all_valid_syntax


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
    #
    # Token-0 mode (DEFAULT, see _token0_enabled) uses IterGen's EXACT bare prompt
    # (no few-shot, no << >> instruction) for a fair head-to-head: the model is
    # grammar-constrained from token 0 so it emits no delimiters and no SQL! echo,
    # making the bare prompt safe (the recorded zero-shot collapse above was a
    # REACTIVE <<>> strategy, which no longer applies). SPIDER_ALIGNED_PROMPT=1
    # forces the aligned prompt even under the legacy <<>> opt-out path.
    # SPIDER_PARITY_LEGACY_PROMPT=1: match run_itergen_legacy_adapter's
    # expression_only few-shot prompt (used to freeze spider_legacy_n5).
    import os

    if os.environ.get("SPIDER_PARITY_LEGACY_PROMPT") == "1":
        return format_prompt_expression_only(evaluator, example)

    if _token0_enabled() or os.environ.get("SPIDER_ALIGNED_PROMPT") == "1":
        return format_spider_itergen_aligned_prompt(example)

    # CRANE baseline (SPIDER_CRANE_COT=1, legacy visible-<<>> path): CRANE is
    # reasoning-based, so it gets the chain-of-thought prompt (reason step by
    # step, then wrap the query in << >>) paired with the CraneGeneration body.
    if os.environ.get("SPIDER_CRANE_COT") == "1":
        return format_prompt_chain_of_thought(evaluator, example)

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
    if not _token0_enabled():
        return actual, source, None
    # Token-0 mode: the syntax metric has no << >> spans to parse, so compute
    # grammar validity here over the EXTRACTED SQL and stash it in aux (the same
    # pattern SMILES uses). Reuses the evaluator's cached sql.lark parser; does
    # NOT modify the grammar file or the correctness grader (prediction_matches_gold).
    syntax_valid = False
    if actual:
        try:
            parser = evaluator._get_syntax_parser(example)
            parser.parse(actual.strip())
            syntax_valid = True
        except Exception:
            syntax_valid = False
    return actual, source, {"syntax_valid": syntax_valid}


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

    if _token0_enabled():
        # Begin inside a constrained chunk from token 0 (no leading << forced, no
        # visible delimiters) — the IterGen-style decoding surface.
        # completion_mode=True: IterGen `start(str)` tokenizes the raw prompt with
        # NO apply_chat_template (chat template only when prompt is a message list).
        # Our adapter always passes expression_only / aligned prompts as strings, so
        # ChatML here was an asymmetry vs frozen IterGen (bare SELECT vs leading WS).
        def _token0_runner(*args, **kwargs):
            kwargs.setdefault("start_inside_constrained", True)
            kwargs.setdefault("completion_mode", True)
            return run_crane_csd(*args, **kwargs)

        return _token0_runner
    return run_crane_csd


def get_syntax_parser(evaluator: Any, example: dict[str, Any] | None):
    from lark import Lark

    return Lark(evaluator._get_grammar_text(), start="start", parser="lalr")


def ensure_runtime_prereqs(evaluator: Any) -> None:
    return None


def compute_aux_metrics(evaluator: Any, sample_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {}
