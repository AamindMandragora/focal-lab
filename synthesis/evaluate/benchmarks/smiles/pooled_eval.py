"""Pooled SMILES evaluation: one session per class, unique-molecule scoring."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from synthesis.evaluate.prompt_tiers import PromptTier

DEFAULT_SMILES_POOLED_MAX_ATTEMPTS = 200
# Target count of first-occurrence unique RDKit-valid molecules per class session.
DEFAULT_SMILES_POOLED_SUCCESS_TARGET = 100
SMILES_POOLED_MAX_NEW_TOKENS = 512


class SmilesStopCriterion(str, Enum):
    GRAMMAR_SUCCESS = "grammar_success"
    UNIQUE_SYNTAX_VALID = "unique_syntax_valid"
    # Backward-compatible alias for older call sites.
    NOVEL_VALID = "unique_syntax_valid"


class SmilesPromptFeedback(str, Enum):
    STATIC = "static"
    DYNAMIC_GOOD_BAD = "dynamic_good_bad"


@dataclass(frozen=True)
class SmilesPooledConfig:
    max_attempts: int = DEFAULT_SMILES_POOLED_MAX_ATTEMPTS
    success_target: int = DEFAULT_SMILES_POOLED_SUCCESS_TARGET
    max_new_tokens: int = SMILES_POOLED_MAX_NEW_TOKENS
    stop_criterion: SmilesStopCriterion = SmilesStopCriterion.UNIQUE_SYNTAX_VALID
    prompt_feedback: SmilesPromptFeedback = SmilesPromptFeedback.DYNAMIC_GOOD_BAD
    prompt_tier: PromptTier = 1


@dataclass(frozen=True)
class SmilesPooledScoreSummary:
    success_target: int
    total_attempts: int
    unique_syntax_valid_count: int
    unique_in_class_count: int
    syntax_rate: float
    accuracy: float
    class_name: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "success_target": self.success_target,
            "total_attempts": self.total_attempts,
            "unique_syntax_valid_count": self.unique_syntax_valid_count,
            "unique_in_class_count": self.unique_in_class_count,
            "syntax_rate": self.syntax_rate,
            "accuracy": self.accuracy,
            "class_name": self.class_name or None,
        }


def smiles_unique_syntax_valid_target_from_args(args: Any) -> int:
    """Resolve the per-class unique syntax-valid molecule target from CLI/runtime args."""
    for attr in ("smiles_unique_syntax_valid_target", "cars_success_target"):
        raw = getattr(args, attr, None)
        if raw is not None:
            return max(1, int(raw))
    return DEFAULT_SMILES_POOLED_SUCCESS_TARGET


def smiles_pooled_config_from_args(
    args: Any,
    *,
    stop_criterion: SmilesStopCriterion = SmilesStopCriterion.UNIQUE_SYNTAX_VALID,
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
    success_target = smiles_unique_syntax_valid_target_from_args(args)
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


def _record_eval_row(record: Mapping[str, Any]) -> dict[str, Any]:
    eval_row = record.get("smiles_eval")
    if isinstance(eval_row, dict):
        return eval_row
    return {
        "smiles": record.get("extracted") or record.get("actual") or "",
        "syntax_valid": record.get("syntax_valid"),
        "rdkit_available": record.get("rdkit_available"),
        "rdkit_valid": record.get("rdkit_valid"),
        "class_membership": record.get("class_membership"),
        "valid_class_membership": record.get("valid_class_membership"),
        "is_prompt_exemplar": record.get("is_prompt_exemplar"),
        "unique_valid_candidate": record.get("unique_valid_candidate"),
    }


def _prompt_exemplars_for_record(record: Mapping[str, Any]) -> list[str]:
    exemplars = record.get("prompt_exemplars")
    if isinstance(exemplars, list) and exemplars:
        return [str(value).strip() for value in exemplars if str(value).strip()]
    class_name = str(record.get("class_name") or "").strip()
    if class_name:
        from synthesis.evaluate.benchmarks.smiles.native_prompt import full_prompt_exemplars

        return list(full_prompt_exemplars(class_name))
    return []


def aggregate_unique_smiles_records(
    records: Sequence[Mapping[str, Any]],
    *,
    success_target: int = DEFAULT_SMILES_POOLED_SUCCESS_TARGET,
    prompt_exemplars: Sequence[str] | None = None,
    class_name: str = "",
) -> SmilesPooledScoreSummary:
    """
    Score first-occurrence unique molecules against ``success_target``.

    Syntax numerator: unique, non-exemplar, first-occurrence RDKit-valid SMILES.
    Accuracy numerator: subset that is also in the requested class.
    """
    exemplars = [str(value).strip() for value in (prompt_exemplars or ()) if str(value).strip()]
    if not exemplars and records:
        exemplars = _prompt_exemplars_for_record(records[0])

    seen = set(exemplars)
    unique_syntax_valid_count = 0
    unique_in_class_count = 0

    for record in records:
        eval_row = _record_eval_row(record)
        smiles = str(eval_row.get("smiles") or "").strip()
        if not smiles or smiles in seen:
            continue
        seen.add(smiles)

        syntax_valid = smiles_rdkit_syntax_valid(eval_row)
        if syntax_valid:
            unique_syntax_valid_count += 1
            in_class = bool(
                eval_row.get("valid_class_membership")
                if "valid_class_membership" in eval_row
                else eval_row.get("class_membership")
            )
            if in_class:
                unique_in_class_count += 1

    target = max(1, int(success_target))
    return SmilesPooledScoreSummary(
        success_target=target,
        total_attempts=len(records),
        unique_syntax_valid_count=unique_syntax_valid_count,
        unique_in_class_count=unique_in_class_count,
        syntax_rate=unique_syntax_valid_count / target,
        accuracy=unique_in_class_count / target,
        class_name=class_name,
    )


def aggregate_smiles_pooled_scores(
    records: Sequence[Mapping[str, Any]],
    *,
    success_target: int = DEFAULT_SMILES_POOLED_SUCCESS_TARGET,
) -> SmilesPooledScoreSummary:
    """Aggregate pooled SMILES scores, averaging per-class rates when needed."""
    if not records:
        target = max(1, int(success_target))
        return SmilesPooledScoreSummary(
            success_target=target,
            total_attempts=0,
            unique_syntax_valid_count=0,
            unique_in_class_count=0,
            syntax_rate=0.0,
            accuracy=0.0,
        )

    class_names = sorted(
        {
            str(record.get("class_name") or "").strip()
            for record in records
            if str(record.get("class_name") or "").strip()
        }
    )
    if not class_names:
        return aggregate_unique_smiles_records(records, success_target=success_target)

    per_class = [
        aggregate_unique_smiles_records(
            [record for record in records if str(record.get("class_name") or "") == class_name],
            success_target=success_target,
            class_name=class_name,
        )
        for class_name in class_names
    ]
    target = max(1, int(success_target))
    return SmilesPooledScoreSummary(
        success_target=target,
        total_attempts=sum(summary.total_attempts for summary in per_class),
        unique_syntax_valid_count=sum(summary.unique_syntax_valid_count for summary in per_class),
        unique_in_class_count=sum(summary.unique_in_class_count for summary in per_class),
        syntax_rate=sum(summary.syntax_rate for summary in per_class) / len(per_class),
        accuracy=sum(summary.accuracy for summary in per_class) / len(per_class),
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
    unique_syntax_valid_count: int,
    novel_valid_count: int | None = None,
) -> bool:
    del novel_valid_count
    if attempt_index + 1 >= config.max_attempts:
        return True
    if (
        config.stop_criterion == SmilesStopCriterion.GRAMMAR_SUCCESS
        and grammar_successes >= config.success_target
    ):
        return True
    if (
        config.stop_criterion in {
            SmilesStopCriterion.UNIQUE_SYNTAX_VALID,
            SmilesStopCriterion.NOVEL_VALID,
        }
        and unique_syntax_valid_count >= config.success_target
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
    scoring_seen = set(static_exemplars)

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
    unique_syntax_valid_count = 0
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
        smiles = str(eval_row.get("smiles") or "").strip()
        is_first_occurrence = bool(smiles and smiles not in scoring_seen)
        if is_first_occurrence:
            scoring_seen.add(smiles)
            if syntax_valid:
                unique_syntax_valid_count += 1
        is_correct = bool(is_first_occurrence and eval_row.get("unique_valid_candidate"))
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
            extra={
                **extra,
                "prompt_exemplars": prompt_exemplars,
                "is_first_occurrence": is_first_occurrence,
                "smiles_eval": eval_row,
            },
        )
        rows.append(row)
        if checkpoint is not None:
            checkpoint()
        print(
            f"{log_prefix} {class_name} attempt {attempt_idx + 1}/{config.max_attempts} "
            f"(unique syntax-valid {unique_syntax_valid_count}/{config.success_target}, "
            f"grammar {grammar_successes}): correct={is_correct} syntax={syntax_valid}",
            flush=True,
        )
        if should_stop_pooled_session(
            attempt_index=attempt_idx,
            config=config,
            grammar_successes=grammar_successes,
            unique_syntax_valid_count=unique_syntax_valid_count,
        ):
            break

    return {
        "attempts": attempts_run,
        "grammar_successes": grammar_successes,
        "unique_syntax_valid_count": unique_syntax_valid_count,
        "novel_valid_count": sum(1 for row in rows if row.get("correct")),
    }


def pooled_smiles_extra_metrics(
    rows: list[dict[str, Any]],
    *,
    adapter: str,
    success_target: int = DEFAULT_SMILES_POOLED_SUCCESS_TARGET,
) -> dict[str, Any]:
    summary = aggregate_smiles_pooled_scores(rows, success_target=success_target)
    return {
        "adapter": adapter,
        "total_attempts": summary.total_attempts,
        "grammar_successes": sum(1 for row in rows if row.get("grammar_success")),
        "unique_syntax_valid_count": summary.unique_syntax_valid_count,
        "unique_in_class_count": summary.unique_in_class_count,
        "novel_valid_count": summary.unique_in_class_count,
        "rdkit_valid_count": summary.unique_syntax_valid_count,
        "success_target": summary.success_target,
    }


def finalize_pooled_smiles_metadata(
    metadata: dict[str, Any],
    *,
    prompt_feedback: SmilesPromptFeedback,
    success_target: int = DEFAULT_SMILES_POOLED_SUCCESS_TARGET,
) -> dict[str, Any]:
    final = dict(metadata)
    final["scoring"] = "unique_over_success_target"
    final["success_target"] = max(1, int(success_target))
    if prompt_feedback == SmilesPromptFeedback.STATIC:
        final["prompt_style"] = "native_acrylates_txt_static"
    else:
        final["prompt_style"] = "native_acrylates_txt_dynamic_good_bad"
    final.pop("checkpoint", None)
    final["complete"] = True
    return final
