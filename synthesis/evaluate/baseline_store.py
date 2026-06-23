"""Baseline JSON storage helpers.

Each baseline file uses a common per-answer schema:

- ``question``: normalized benchmark question (plain text)
- ``prompt``: full prompt sent to the model
- ``generated``: raw model completion (generation suffix only)
- ``extracted``: parsed answer used for scoring
- ``correct``, ``syntax_valid``: per-example booleans
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluator import EvaluationResult

# Legacy field kept for readers that still look for a single answer string.
_LEGACY_ANSWER_FIELD = "generated_answer"


def normalize_baseline_question(
    dataset: str,
    *,
    example: dict[str, Any] | None = None,
    row: dict[str, Any] | None = None,
    fallback: str = "",
) -> str:
    """Return a plain question string for row alignment across strategies."""
    source = example if example is not None else row if row is not None else {}
    bench = "gsm_symbolic" if dataset in ("gsm", "gsm_symbolic") else dataset

    if bench == "gsm_symbolic":
        text = (
            source.get("question_parsed")
            or source.get("original_question")
            or source.get("question")
        )
        return str(text or fallback).strip()

    if bench == "spider":
        text = str(source.get("question") or fallback).strip()
        if "### Question:" in text:
            if not source.get("db_info") and "SQL tables information:" in text:
                mid = text.split("SQL tables information:", 1)[1]
                db_info = mid.split("### Question:", 1)[0].strip()
                source = {**source, "db_info": db_info}
            text = text.split("### Question:", 1)[-1].strip()
        if text.endswith("SQL:"):
            text = text[:-4].strip()
        return text

    if bench == "smiles":
        return str(source.get("class_name") or source.get("question") or fallback).strip()

    return str(source.get("question") or fallback).strip()


def _prompt_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    return str(value)


def _coalesce_bool(row: dict[str, Any], *keys: str) -> bool | None:
    for key in keys:
        if key in row and isinstance(row[key], bool):
            return bool(row[key])
    return None


def compose_baseline_answer(
    *,
    dataset: str,
    question: str | None = None,
    example: dict[str, Any] | None = None,
    row: dict[str, Any] | None = None,
    prompt: Any = None,
    generated: str | None = None,
    extracted: str | None = None,
    correct: bool | None = None,
    syntax_valid: bool | None = None,
    num_tokens: int | None = None,
    generation_seconds: float | None = None,
) -> dict[str, Any]:
    """Build one standardized ``answers[]`` entry."""
    src = row or example or {}
    if question is None:
        question = normalize_baseline_question(dataset, example=example, row=row)

    if prompt is None:
        prompt = src.get("prompt") or src.get("prompt_used")
    if generated is None:
        generated = (
            src.get("generated")
            or src.get("raw_generated")
            or src.get("llm_response")
            or src.get("response")
            or src.get("pred")
            or src.get("full_output")
            or ""
        )
    if extracted is None:
        extracted = (
            src.get("extracted")
            if src.get("extracted") is not None
            else src.get("parsed_completion")
            if src.get("parsed_completion") is not None
            else src.get("actual")
            if src.get("actual") is not None
            else ""
        )
    if correct is None:
        correct = _coalesce_bool(src, "correct", "is_correct")
    if syntax_valid is None:
        syntax_valid = _coalesce_bool(
            src, "syntax_valid", "is_syntax_valid", "grammar_valid", "out_parse_success"
        )
    if num_tokens is None and src.get("num_tokens") is not None:
        num_tokens = int(src["num_tokens"])
    elif num_tokens is None and src.get("token_count") is not None:
        num_tokens = int(src["token_count"])
    if generation_seconds is None and src.get("generation_seconds") is not None:
        generation_seconds = float(src["generation_seconds"])
    elif generation_seconds is None and src.get("time_seconds") is not None:
        generation_seconds = float(src["time_seconds"])

    extracted_s = str(extracted or "")
    entry: dict[str, Any] = {
        "question": str(question or ""),
        "prompt": _prompt_text(prompt),
        "generated": str(generated or ""),
        "extracted": extracted_s,
        "correct": bool(correct) if correct is not None else False,
        "syntax_valid": bool(syntax_valid) if syntax_valid is not None else False,
        _LEGACY_ANSWER_FIELD: extracted_s,
    }
    if num_tokens is not None:
        entry["num_tokens"] = int(num_tokens)
    if generation_seconds is not None:
        entry["generation_seconds"] = round(float(generation_seconds), 6)
    return entry


def baseline_answer_row_complete(row: dict[str, Any]) -> bool:
    """True when a row has the fields required by the baseline schema."""
    if not isinstance(row, dict):
        return False
    required = ("question", "prompt", "generated", "extracted", "correct", "syntax_valid")
    if not all(key in row for key in required):
        return False
    if _LEGACY_ANSWER_FIELD not in row:
        return False
    return True


def _finalize_runtime_metrics(
    metrics: dict[str, Any],
    *,
    rows_or_samples: list[dict[str, Any]],
    time_key: str = "generation_seconds",
    token_key: str = "num_tokens",
    run_wall_time_seconds: float | None = None,
    evaluator_total_time_seconds: float | None = None,
    evaluator_max_sample_time_seconds: float | None = None,
) -> dict[str, Any]:
    """Always emit runtime metric keys (null when unavailable)."""
    times = [
        float(item[time_key])
        for item in rows_or_samples
        if item.get(time_key) is not None
    ]
    toks = [
        int(item[token_key])
        for item in rows_or_samples
        if item.get(token_key) is not None
    ]
    metrics["examples_with_generation_timing"] = len(times)
    metrics["total_generation_seconds"] = round(sum(times), 4) if times else None
    metrics["mean_generation_seconds_per_example"] = (
        round(sum(times) / len(times), 6) if times else None
    )
    metrics["examples_with_token_counts"] = len(toks)
    metrics["total_output_tokens"] = int(sum(toks)) if toks else None
    metrics["mean_output_tokens_per_example"] = (
        round(sum(toks) / len(toks), 4) if toks else None
    )
    metrics["run_wall_time_seconds"] = (
        round(float(run_wall_time_seconds), 4) if run_wall_time_seconds is not None else None
    )
    metrics["evaluator_total_time_seconds"] = (
        round(float(evaluator_total_time_seconds), 4)
        if evaluator_total_time_seconds is not None
        else None
    )
    metrics["evaluator_max_sample_time_seconds"] = (
        round(float(evaluator_max_sample_time_seconds), 4)
        if evaluator_max_sample_time_seconds is not None
        else None
    )
    return metrics


def build_metrics_from_eval_samples(
    samples: list[dict[str, Any]],
    *,
    evaluator_total_time_seconds: float | None = None,
    evaluator_max_sample_time_seconds: float | None = None,
    run_wall_time_seconds: float | None = None,
) -> dict[str, Any]:
    """Aggregate optional timing / token fields from evaluator samples."""
    metrics: dict[str, Any] = {"num_examples": len(samples)}
    return _finalize_runtime_metrics(
        metrics,
        rows_or_samples=samples,
        time_key="time_seconds",
        token_key="token_count",
        run_wall_time_seconds=run_wall_time_seconds,
        evaluator_total_time_seconds=evaluator_total_time_seconds,
        evaluator_max_sample_time_seconds=evaluator_max_sample_time_seconds,
    )


def _answers_from_rows(rows: list[dict[str, Any]], *, dataset: str) -> list[dict[str, Any]]:
    return [compose_baseline_answer(dataset=dataset, row=row) for row in rows]


def build_minimal_baseline_record(
    result: EvaluationResult,
    *,
    dataset: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build baseline payload from an :class:`EvaluationResult`."""
    samples = result.sample_outputs or []
    answers = [
        compose_baseline_answer(
            dataset=dataset,
            row=sample,
            question=normalize_baseline_question(dataset, row=sample),
            prompt=sample.get("prompt"),
            generated=sample.get("generated") or sample.get("full_output") or "",
            extracted=sample.get("actual") if sample.get("actual") is not None else sample.get("scored_output") or "",
            correct=sample.get("is_correct"),
            syntax_valid=sample.get("is_syntax_valid"),
            num_tokens=sample.get("token_count"),
            generation_seconds=sample.get("time_seconds"),
        )
        for sample in samples
    ]

    metrics = build_metrics_from_eval_samples(
        samples,
        evaluator_total_time_seconds=result.total_time_seconds,
        evaluator_max_sample_time_seconds=result.max_sample_time_seconds,
    )

    payload: dict[str, Any] = {
        "accuracy": float(result.accuracy),
        "syntax_rate": float(result.syntax_rate),
        "metrics": metrics,
        "answers": answers,
    }
    if metadata:
        payload["metadata"] = metadata
    return payload


def baseline_payload_from_success_report(
    report: dict[str, Any],
    *,
    dataset: str = "gsm_symbolic",
) -> dict[str, Any]:
    """Build baseline JSON from ``success_report.json``."""
    evaluation = report.get("evaluation_result") or {}
    samples = report.get("sample_outputs") or []

    answers = [
        compose_baseline_answer(
            dataset=dataset,
            row=sample,
            question=normalize_baseline_question(dataset, row=sample),
            prompt=sample.get("prompt"),
            generated=sample.get("generated") or sample.get("full_output") or "",
            extracted=sample.get("actual") if sample.get("actual") is not None else sample.get("scored_output") or "",
            correct=sample.get("is_correct"),
            syntax_valid=sample.get("is_syntax_valid"),
            num_tokens=sample.get("token_count"),
            generation_seconds=sample.get("time_seconds"),
        )
        for sample in samples
    ]

    metrics = build_metrics_from_eval_samples(
        samples,
        evaluator_total_time_seconds=evaluation.get("total_time_seconds"),
        evaluator_max_sample_time_seconds=evaluation.get("max_sample_time_seconds"),
    )

    return {
        "accuracy": float(evaluation.get("accuracy", 0.0)),
        "syntax_rate": float(evaluation.get("syntax_rate", 0.0)),
        "metrics": metrics,
        "answers": answers,
    }


def save_minimal_baseline_json(
    result: EvaluationResult,
    json_path: Path,
    *,
    dataset: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write baseline JSON from an evaluation result."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_minimal_baseline_record(result, dataset=dataset, metadata=metadata)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    return json_path


def build_minimal_baseline_from_rows(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    run_wall_time_seconds: float | None = None,
    extra_metrics: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build baseline JSON from per-example adapter rows."""
    metrics: dict[str, Any] = {"num_examples": len(rows)}
    metrics = _finalize_runtime_metrics(
        metrics,
        rows_or_samples=rows,
        run_wall_time_seconds=run_wall_time_seconds,
    )
    if extra_metrics:
        metrics.update(extra_metrics)

    answers = _answers_from_rows(rows, dataset=dataset)

    if not answers:
        payload: dict[str, Any] = {
            "accuracy": 0.0,
            "syntax_rate": 0.0,
            "metrics": metrics,
            "answers": [],
        }
        if metadata:
            payload["metadata"] = metadata
        return payload

    bench = "gsm_symbolic" if dataset in ("gsm", "gsm_symbolic") else dataset
    if bench == "smiles":
        from synthesis.evaluate.benchmarks.smiles.pooled_eval import (
            DEFAULT_SMILES_POOLED_SUCCESS_TARGET,
            aggregate_smiles_pooled_scores,
        )

        success_target = int(
            (metadata or {}).get("success_target")
            or (extra_metrics or {}).get("success_target")
            or DEFAULT_SMILES_POOLED_SUCCESS_TARGET
        )
        summary = aggregate_smiles_pooled_scores(rows, success_target=success_target)
        metrics.update(summary.as_dict())
        payload = {
            "accuracy": summary.accuracy,
            "syntax_rate": summary.syntax_rate,
            "metrics": metrics,
            "answers": answers,
            "accuracy_definition": "unique_in_class_over_success_target",
            "syntax_definition": "unique_syntax_valid_over_success_target",
            "accuracy_denominator": summary.success_target,
            "syntax_denominator": summary.success_target,
        }
    else:
        correct_vals = [1.0 if a["correct"] else 0.0 for a in answers]
        syntax_vals = [1.0 if a["syntax_valid"] else 0.0 for a in answers]
        payload = {
            "accuracy": sum(correct_vals) / max(1, len(correct_vals)),
            "syntax_rate": sum(syntax_vals) / max(1, len(syntax_vals)),
            "metrics": metrics,
            "answers": answers,
        }
    if metadata:
        payload["metadata"] = metadata
    return payload


def save_minimal_baseline_from_rows(
    rows: list[dict[str, Any]],
    json_path: Path,
    *,
    dataset: str,
    run_wall_time_seconds: float | None = None,
    extra_metrics: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write baseline JSON from adapter rows."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_minimal_baseline_from_rows(
        rows,
        dataset=dataset,
        run_wall_time_seconds=run_wall_time_seconds,
        extra_metrics=extra_metrics,
        metadata=metadata,
    )
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    return json_path
