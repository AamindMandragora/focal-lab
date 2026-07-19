"""Minimal baseline JSON storage helpers.

Baselines persist aggregate accuracy/syntax, optional timing and token metrics,
and per-row answers (question + generated text).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluator import EvaluationResult


def build_metrics_from_eval_samples(
    samples: list[dict[str, Any]],
    *,
    evaluator_total_time_seconds: float | None = None,
    evaluator_max_sample_time_seconds: float | None = None,
    run_wall_time_seconds: float | None = None,
) -> dict[str, Any]:
    """Aggregate optional ``time_seconds`` / ``token_count`` fields from evaluator samples."""
    metrics: dict[str, Any] = {"num_examples": len(samples)}
    times = [
        float(s["time_seconds"])
        for s in samples
        if s.get("time_seconds") is not None
    ]
    toks = [
        int(s["token_count"])
        for s in samples
        if s.get("token_count") is not None
    ]
    if times:
        metrics["total_generation_seconds"] = round(sum(times), 4)
        metrics["mean_generation_seconds_per_example"] = round(sum(times) / len(times), 6)
        metrics["examples_with_generation_timing"] = len(times)
    if toks:
        metrics["total_output_tokens"] = int(sum(toks))
        metrics["mean_output_tokens_per_example"] = round(sum(toks) / len(toks), 4)
        metrics["examples_with_token_counts"] = len(toks)
    if evaluator_total_time_seconds is not None:
        metrics["evaluator_total_time_seconds"] = round(float(evaluator_total_time_seconds), 4)
    if evaluator_max_sample_time_seconds is not None:
        metrics["evaluator_max_sample_time_seconds"] = round(
            float(evaluator_max_sample_time_seconds), 4
        )
    if run_wall_time_seconds is not None:
        metrics["run_wall_time_seconds"] = round(float(run_wall_time_seconds), 4)
    return metrics


def build_minimal_baseline_record(
    result: EvaluationResult,
    eval_split: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the minimal baseline payload from an evaluation result.

    ``eval_split`` is the split-provenance block (which split file/side the
    numbers were measured on — see synthesis/split_provenance.py). Callers
    should pass it so saved JSONs are self-describing; comparisons across
    split sides caused wrong win/loss verdicts on 2026-07-17.
    """
    samples = result.sample_outputs or []
    answers: list[dict[str, Any]] = []
    for sample in samples:
        question = str(sample.get("question", ""))
        generated_answer = sample.get("scored_output")
        if not generated_answer:
            generated_answer = sample.get("full_output", "")
        row: dict[str, Any] = {
            "question": question,
            "generated_answer": str(generated_answer),
        }
        for key in (
            "example_index",
            "source_index",
            "crane_source_index",
            "spider_source_index",
            "db_id",
            "id_orig",
            "id_shuffled",
        ):
            if sample.get(key) is not None:
                row[key] = sample.get(key)
        if sample.get("token_count") is not None:
            row["num_tokens"] = int(sample["token_count"])
        if sample.get("time_seconds") is not None:
            row["generation_seconds"] = round(float(sample["time_seconds"]), 6)
        # Per-example outcome flags so saved JSONs support offline diffing
        # against baseline-side annotations without regrading.
        for key in ("is_correct", "is_syntax_valid"):
            if sample.get(key) is not None:
                row[key] = bool(sample[key])
        answers.append(row)

    metrics = build_metrics_from_eval_samples(
        samples,
        evaluator_total_time_seconds=result.total_time_seconds,
        evaluator_max_sample_time_seconds=result.max_sample_time_seconds,
    )

    record = {
        "accuracy": float(result.accuracy),
        "syntax_rate": float(result.syntax_rate),
        "metrics": metrics,
        "answers": answers,
    }
    if eval_split is not None:
        record["eval_split"] = eval_split
    # Preserve the CARS-paper SMILES metrics (unique_valid_count, diversity_tanimoto,
    # validity_rdkit, samples_to_target_unique_valid) so saved JSONs carry the real
    # comparison axes instead of just the headline accuracy. Inert for other datasets.
    trial = (result.aux_metrics or {}).get("smiles_paper_trial")
    if trial:
        record["smiles_paper_trial"] = trial
    return record


def baseline_payload_from_success_report(report: dict[str, Any]) -> dict[str, Any]:
    """Build the same baseline JSON shape as legacy exports from ``success_report.json``."""
    evaluation = report.get("evaluation_result") or {}
    samples = report.get("sample_outputs") or []

    answers: list[dict[str, Any]] = []
    for sample in samples:
        question = str(sample.get("question", ""))
        generated_answer = sample.get("scored_output")
        if not generated_answer:
            generated_answer = sample.get("full_output", "")
        row: dict[str, Any] = {
            "question": question,
            "generated_answer": str(generated_answer),
        }
        if sample.get("token_count") is not None:
            row["num_tokens"] = int(sample["token_count"])
        if sample.get("time_seconds") is not None:
            row["generation_seconds"] = round(float(sample["time_seconds"]), 6)
        # Per-example outcome flags so saved JSONs support offline diffing
        # against baseline-side annotations without regrading.
        for key in ("is_correct", "is_syntax_valid"):
            if sample.get(key) is not None:
                row[key] = bool(sample[key])
        answers.append(row)

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
    eval_split: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a minimal baseline JSON file and return its path."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_minimal_baseline_record(result, eval_split=eval_split)
    if metadata:
        payload.update(metadata)
    temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(json_path)
    return json_path
