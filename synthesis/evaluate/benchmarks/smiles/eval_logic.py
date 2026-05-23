"""SMILES evaluation logic delegated from the global evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def normalize_classes(evaluator: Any) -> list[str]:
    from synthesis.evaluate.benchmarks.smiles.dataset import normalize_smiles_classes

    return normalize_smiles_classes(evaluator.smiles_classes)


def get_grammar_file(evaluator: Any, grammars_dir: Path) -> Path:
    from synthesis.evaluate.benchmarks.smiles.dataset import get_smiles_task

    classes = normalize_classes(evaluator)
    if not classes:
        raise ValueError("At least one SMILES class is required via --smiles-classes.")
    # Bootstrap parser only: per-example decoding uses build_dynamic_parser().
    return Path(get_smiles_task(classes[0])["grammar_path"])


def load_dataset_sample(evaluator: Any) -> list[dict[str, Any]]:
    from synthesis.evaluate.benchmarks.smiles.dataset import load_smiles

    return load_smiles(classes=normalize_classes(evaluator), samples_per_class=evaluator.sample_size)


def format_prompt(evaluator: Any, example: dict[str, Any]) -> str:
    """SMILES prompt for tier 2 or synthesis-selected tier; ``prompt_state`` may append suffix rows."""
    from synthesis.evaluate.prompt_tiers import render_benchmark_prompt, render_smiles_eval_prompt

    if getattr(evaluator, "grammar_prompt_tier", None) is not None:
        return render_smiles_eval_prompt(evaluator, example)
    return render_benchmark_prompt("smiles", tier=2, example=example)


def format_prompt_expression_only(evaluator: Any, example: dict[str, Any]) -> str:
    """Tier-1 SMILES prompt (legacy body-only adapters) or synthesis answer-only delimited text."""
    from synthesis.evaluate.prompt_tiers import render_benchmark_prompt, render_smiles_eval_prompt

    if getattr(evaluator, "grammar_prompt_tier", None) is not None:
        return render_smiles_eval_prompt(evaluator, example)
    return render_benchmark_prompt("smiles", tier=1, example=example)


def format_prompt_chain_of_thought(evaluator: Any, example: dict[str, Any]) -> str:
    """Tier-2 SMILES prompt."""
    return format_prompt(evaluator, example)


def expected_answer(evaluator: Any, example: dict[str, Any]) -> str:
    return str(example.get("class_name", ""))


def grammar_text_for_prompt_tier(example: dict[str, Any], prompt_tier: int) -> str:
    """Return decoder grammar: tier 1 body-only; tier 2 delimited ``<<`` SMILES ``>>`` span."""
    from synthesis.evaluate.benchmarks.smiles.grammar_helpers import (
        build_smiles_tier1_body_grammar,
        build_smiles_tier2_delimited_grammar,
    )

    base = str(example.get("grammar_text", ""))
    if int(prompt_tier) == 1:
        return build_smiles_tier1_body_grammar(base)
    return build_smiles_tier2_delimited_grammar(base)


def grammar_tier_for_evaluator(evaluator: Any) -> int:
    """Grammar tier may differ from prompt tier during compiled CSD evaluation."""
    grammar_tier = getattr(evaluator, "grammar_prompt_tier", None)
    if grammar_tier is not None:
        return int(grammar_tier)
    prompt_tier = getattr(evaluator, "prompt_tier", None)
    if prompt_tier is not None:
        return int(prompt_tier)
    return 2


def build_dynamic_parser(evaluator: Any, env: dict[str, Any], example: dict[str, Any]):
    from synthesis.evaluate.benchmarks.common.parser_utils import create_lark_dafny_parser

    grammar_text = grammar_text_for_prompt_tier(example, grammar_tier_for_evaluator(evaluator))
    class_name = str(example.get("class_name", "smiles"))
    cache_key = ("smiles", class_name, grammar_text)
    parser_factory = evaluator._dynamic_parser_factory_cache.get(cache_key)
    if parser_factory is None:
        parser_factory = create_lark_dafny_parser(
            grammar_text,
            env["VerifiedDecoderAgent"],
            env["_dafny"],
            start="start",
            tokenizer=env["tokenizer"],
        )
        evaluator._dynamic_parser_factory_cache[cache_key] = parser_factory
    return parser_factory(env["lm"]._Tokens)


def extract_actual(
    evaluator: Any,
    scored_output: str,
    example: dict[str, Any],
) -> tuple[str | None, str, dict[str, Any] | None]:
    import re
    from synthesis.evaluate.benchmarks.smiles.metrics import evaluate_smiles_output

    class_name = example.get("class_name", "smiles")
    base_grammar_text = str(
        example.get("base_grammar_text") or example.get("grammar_text") or ""
    )
    tier_grammar_text = grammar_text_for_prompt_tier(example, grammar_tier_for_evaluator(evaluator))
    prompt_exemplars = example.get("prompt_exemplars", [])

    from synthesis.evaluate.benchmarks.common.delimited_output import extract_last_delimited_span

    span, found = extract_last_delimited_span(scored_output)
    candidate = span if found else scored_output

    smiles_eval = evaluate_smiles_output(
        class_name,
        candidate,
        tier_grammar_text,
        prompt_exemplars,
        require_rdkit=True,
        base_grammar_text=base_grammar_text,
    )
    return smiles_eval["smiles"] or None, "smiles_eval", smiles_eval


def is_correct(
    evaluator: Any,
    actual: str | None,
    expected: str,
    example: dict[str, Any],
    aux: dict[str, Any] | None,
    scored_output: str,
) -> bool:
    if aux and "novel_valid" in aux:
        return bool(aux.get("novel_valid"))
    return bool(aux and aux.get("unique_valid_candidate"))


def init_prompt_states(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    from synthesis.evaluate.benchmarks.smiles.prompt_state import init_prompt_states as _init

    return _init(dataset)


def apply_prompt_state(example: dict[str, Any], states: dict[str, Any]) -> None:
    from synthesis.evaluate.benchmarks.smiles.prompt_state import apply_prompt_state as _apply

    _apply(example, states)


def record_prompt_result(
    example: dict[str, Any],
    states: dict[str, Any],
    smiles: str,
    eval_row: dict[str, Any] | None,
) -> dict[str, Any] | None:
    from synthesis.evaluate.benchmarks.smiles.prompt_state import record_prompt_result as _record

    return _record(example, states, smiles, eval_row)


def uses_hidden_chunks() -> bool:
    return False


def example_syntax_pass(
    all_valid_syntax: bool,
    segments: list[tuple[str, bool]],
    used_hidden_chunk: bool,
    aux: dict[str, Any] | None,
) -> bool:
    if aux is not None and "syntax_valid" in aux:
        return bool(aux.get("syntax_valid"))
    if segments:
        return all(is_valid for _, is_valid in segments)
    return bool(all_valid_syntax)


def accuracy_applicable(aux: dict[str, Any] | None) -> bool:
    return bool(aux and aux.get("accuracy_applicable"))


def get_generation_runner():
    from synthesis.evaluate.benchmarks.smiles.generation import run_crane_csd

    return run_crane_csd


def get_syntax_parser(evaluator: Any, example: dict[str, Any] | None):
    from lark import Lark

    grammar_text = (
        example.get("grammar_text", "")
        if isinstance(example, dict)
        else ""
    )
    cache_key = ("smiles", grammar_text)
    parser = evaluator._syntax_parser_cache.get(cache_key)
    if parser is None:
        parser = Lark(grammar_text, start="start", parser="lalr")
        evaluator._syntax_parser_cache[cache_key] = parser
    return parser


def ensure_runtime_prereqs(evaluator: Any) -> None:
    from synthesis.evaluate.benchmarks.smiles.metrics import rdkit_available

    if not rdkit_available():
        raise RuntimeError(
            "SMILES evaluation requires RDKit but it is not available in this environment. "
            "Install RDKit before running smiles synthesis/evaluation."
        )


def compute_aux_metrics(evaluator: Any, sample_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    from synthesis.evaluate.benchmarks.smiles.metrics import smiles_trial_metrics

    paper_metrics = smiles_trial_metrics(
        sample_outputs,
        target_unique_valid=100,
        sample_cap=1000,
    )

    helper_events = [
        event
        for sample in sample_outputs
        for event in (sample.get("helper_trace") or [])
        if isinstance(event, dict)
    ]
    helper_count = len(helper_events)
    open_close_helpers = {
        "OpenConstrainedSpan",
        "CloseConstrainedSpan",
        "EnterObservedConstrainedSpan",
    }
    churn_calls = sum(
        1 for event in helper_events if event.get("helper") in open_close_helpers
    )
    delimiter_churn_ratio = churn_calls / max(1, helper_count)

    tiny_spans = 0
    for sample in sample_outputs:
        smiles_eval = sample.get("smiles_eval") or {}
        smiles = str(smiles_eval.get("smiles") or "")
        if smiles and len(smiles) <= 3:
            tiny_spans += 1
    tiny_span_rate = tiny_spans / max(1, len(sample_outputs))

    max_steps_hits = sum(1 for sample in sample_outputs if sample.get("hit_max_steps"))
    max_steps_hit_rate = max_steps_hits / max(1, len(sample_outputs))

    penalty = min(
        0.60,
        0.35 * delimiter_churn_ratio
        + 0.35 * tiny_span_rate
        + 0.30 * max_steps_hit_rate,
    )
    membership = float(paper_metrics.get("membership", 0.0) or 0.0)
    adjusted_membership = max(0.0, membership - penalty)

    anti = {
        "delimiter_churn_ratio": delimiter_churn_ratio,
        "tiny_span_rate": tiny_span_rate,
        "max_steps_hit_rate": max_steps_hit_rate,
        "penalty": penalty,
        "adjusted_membership_score": adjusted_membership,
    }
    return {
        "smiles_paper_trial": paper_metrics,
        "anti_degeneracy": anti,
    }


def accuracy_upper_bound(
    num_correct: int,
    remaining: int,
    num_accuracy_examples: int,
    total_planned_examples: int,
) -> float:
    return (num_correct + remaining) / max(1, num_accuracy_examples + remaining)


def final_accuracy_denominator(num_examples: int, num_accuracy_examples: int) -> int:
    return num_accuracy_examples


def invalid_outputs_excluded(num_examples: int, num_accuracy_examples: int) -> int:
    return num_examples - num_accuracy_examples


def accuracy_definition() -> str:
    return "class_membership_among_syntax_valid_molecules"
