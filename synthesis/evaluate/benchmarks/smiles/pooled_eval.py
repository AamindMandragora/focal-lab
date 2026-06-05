"""Pooled SMILES evaluation: one session per class, attempt-normalized scoring."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from synthesis.evaluate.prompt_tiers import PromptTier

DEFAULT_SMILES_POOLED_MAX_ATTEMPTS = 200
DEFAULT_SMILES_POOLED_SUCCESS_TARGET = 100
SMILES_POOLED_MAX_NEW_TOKENS = 512


class SmilesStopCriterion(str, Enum):
    GRAMMAR_SUCCESS = "grammar_success"
    NOVEL_VALID = "novel_valid"


class SmilesPromptFeedback(str, Enum):
    STATIC = "static"
    DYNAMIC_GOOD_BAD = "dynamic_good_bad"


@dataclass(frozen=True)
class SmilesPooledConfig:
    max_attempts: int = DEFAULT_SMILES_POOLED_MAX_ATTEMPTS
    success_target: int = DEFAULT_SMILES_POOLED_SUCCESS_TARGET
    max_new_tokens: int = SMILES_POOLED_MAX_NEW_TOKENS
    stop_criterion: SmilesStopCriterion = SmilesStopCriterion.NOVEL_VALID
    prompt_feedback: SmilesPromptFeedback = SmilesPromptFeedback.DYNAMIC_GOOD_BAD
    prompt_tier: PromptTier = 1


def smiles_pooled_config_from_args(
    args: Any,
    *,
    stop_criterion: SmilesStopCriterion,
    prompt_feedback: SmilesPromptFeedback | None = None,
    prompt_tier: PromptTier = 1,
) -> SmilesPooledConfig:
    max_attempts = max(
        1,
        int(
            getattr(args, "cars_search_steps", None)
            or getattr(args, "rs_search_steps", None)
            or DEFAULT_SMILES_POOLED_MAX_ATTEMPTS
        ),
    )
    success_target = max(
        1,
        int(getattr(args, "cars_success_target", DEFAULT_SMILES_POOLED_SUCCESS_TARGET)),
    )
    feedback = prompt_feedback
    if feedback is None:
        feedback = (
            SmilesPromptFeedback.STATIC
            if stop_criterion == SmilesStopCriterion.GRAMMAR_SUCCESS
            else SmilesPromptFeedback.DYNAMIC_GOOD_BAD
        )
    return SmilesPooledConfig(
        max_attempts=max_attempts,
        success_target=success_target,
        max_new_tokens=SMILES_POOLED_MAX_NEW_TOKENS,
        stop_criterion=stop_criterion,
        prompt_feedback=feedback,
        prompt_tier=prompt_tier,
    )


def smiles_rdkit_syntax_valid(eval_row: dict[str, Any]) -> bool:
    if eval_row.get("rdkit_available"):
        return bool(eval_row.get("rdkit_valid"))
    return bool(eval_row.get("syntax_valid"))


def score_smiles_attempt(
    class_name: str,
    output_text: str,
    *,
    prompt_exemplars: list[str],
    tier_grammar: str,
    base_grammar: str,
) -> dict[str, Any]:
    from synthesis.evaluate.benchmarks.smiles.metrics import evaluate_smiles_output

    return evaluate_smiles_output(
        class_name,
        output_text,
        tier_grammar,
        prompt_exemplars,
        require_rdkit=True,
        base_grammar_text=base_grammar,
    )


def append_novel_exemplar(exemplars: list[str], eval_row: dict[str, Any]) -> bool:
    if not eval_row.get("unique_valid_candidate"):
        return False
    smiles = str(eval_row.get("smiles") or "").strip()
    if not smiles or smiles in exemplars:
        return False
    exemplars.append(smiles)
    return True


def should_stop_pooled_session(
    *,
    attempt_index: int,
    config: SmilesPooledConfig,
    grammar_successes: int,
    novel_valid_count: int,
) -> bool:
    if attempt_index + 1 >= config.max_attempts:
        return True
    if (
        config.stop_criterion == SmilesStopCriterion.GRAMMAR_SUCCESS
        and grammar_successes >= config.success_target
    ):
        return True
    if (
        config.stop_criterion == SmilesStopCriterion.NOVEL_VALID
        and novel_valid_count >= config.success_target
    ):
        return True
    return False


AttemptGenerator = Callable[
    [str, dict[str, Any], int],
    tuple[str, int | None, bool, dict[str, Any]],
]


def run_smiles_pooled_class(
    *,
    class_name: str,
    config: SmilesPooledConfig,
    generate_attempt: AttemptGenerator,
    rows: list[dict[str, Any]],
    make_row: Callable[..., dict[str, Any]],
    checkpoint: Callable[[], None] | None = None,
    log_prefix: str = "SMILES",
) -> dict[str, int]:
    from synthesis.evaluate.benchmarks.smiles.dataset import get_smiles_task
    from synthesis.evaluate.benchmarks.smiles.native_prompt import (
        full_prompt_exemplars,
        render_native_smiles_prompt,
        render_native_smiles_prompt_with_feedback,
    )
    from synthesis.evaluate.benchmarks.smiles.prompt_state import SmilesPromptState

    task = get_smiles_task(class_name)
    example = dict(task)
    static_exemplars = list(full_prompt_exemplars(class_name))
    prompt_exemplars = list(static_exemplars)
    tier_grammar = str(example.get("grammar_text") or "")
    base_grammar = str(example.get("base_grammar_text") or tier_grammar)
    static_prompt = config.prompt_feedback == SmilesPromptFeedback.STATIC
    prompt_state = None if static_prompt else SmilesPromptState(static_exemplars)

    def _render_prompt() -> str:
        if static_prompt:
            return render_native_smiles_prompt(
                class_name,
                static_exemplars,
                tier=config.prompt_tier,
            )
        assert prompt_state is not None
        return render_native_smiles_prompt_with_feedback(
            class_name,
            good_results=prompt_state.good_results,
            bad_results=prompt_state.bad_results,
            tier=config.prompt_tier,
        )

    grammar_successes = 0
    novel_valid_count = 0
    prompt = _render_prompt()
    attempts_run = 0

    for attempt_idx in range(config.max_attempts):
        attempts_run += 1
        attempt_started = time.perf_counter()
        output_text, token_count, grammar_ok, extra = generate_attempt(
            prompt,
            example,
            attempt_idx,
        )
        if grammar_ok:
            grammar_successes += 1

        eval_row = score_smiles_attempt(
            class_name,
            output_text,
            prompt_exemplars=prompt_exemplars,
            tier_grammar=tier_grammar,
            base_grammar=base_grammar,
        )
        syntax_valid = smiles_rdkit_syntax_valid(eval_row)
        is_correct = bool(eval_row.get("unique_valid_candidate"))
        if is_correct:
            novel_valid_count += 1
        if not static_prompt and prompt_state is not None:
            attempt_value = str(eval_row.get("smiles") or output_text or "").strip()
            prompt_state.record_attempt(attempt_value, eval_row)
            prompt = _render_prompt()

        attempt_example = dict(example)
        attempt_example["attempt_index"] = attempt_idx
        row = make_row(
            example=attempt_example,
            prompt=prompt,
            output_text=output_text,
            eval_row=eval_row,
            syntax_valid=syntax_valid,
            is_correct=is_correct,
            generation_seconds=time.perf_counter() - attempt_started,
            token_count=token_count,
            grammar_ok=grammar_ok,
            class_name=class_name,
            extra=extra,
        )
        rows.append(row)
        if checkpoint is not None:
            checkpoint()
        print(
            f"{log_prefix} {class_name} attempt {attempt_idx + 1}/{config.max_attempts} "
            f"(novel valid {novel_valid_count}/{config.success_target}, "
            f"grammar {grammar_successes}): correct={is_correct} syntax={syntax_valid}",
            flush=True,
        )
        if should_stop_pooled_session(
            attempt_index=attempt_idx,
            config=config,
            grammar_successes=grammar_successes,
            novel_valid_count=novel_valid_count,
        ):
            break

    return {
        "attempts": attempts_run,
        "grammar_successes": grammar_successes,
        "novel_valid_count": novel_valid_count,
    }


def pooled_smiles_extra_metrics(rows: list[dict[str, Any]], *, adapter: str) -> dict[str, Any]:
    return {
        "adapter": adapter,
        "total_attempts": len(rows),
        "grammar_successes": sum(1 for row in rows if row.get("grammar_success")),
        "novel_valid_count": sum(1 for row in rows if row.get("correct")),
        "rdkit_valid_count": sum(1 for row in rows if row.get("syntax_valid")),
    }


def finalize_pooled_smiles_metadata(
    metadata: dict[str, Any],
    *,
    prompt_feedback: SmilesPromptFeedback,
) -> dict[str, Any]:
    final = dict(metadata)
    final["scoring"] = "attempt_normalized"
    if prompt_feedback == SmilesPromptFeedback.STATIC:
        final["prompt_style"] = "native_acrylates_txt_static"
    else:
        final["prompt_style"] = "native_acrylates_txt_dynamic_good_bad"
    final.pop("checkpoint", None)
    final["complete"] = True
    return final
