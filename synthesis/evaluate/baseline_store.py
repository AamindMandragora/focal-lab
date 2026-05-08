"""Minimal baseline JSON storage helpers.

Baselines are intentionally compact and only persist:
- accuracy
- syntax_rate
- generated answer for each benchmark question
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluator import EvaluationResult


def build_minimal_baseline_record(result: EvaluationResult) -> dict[str, Any]:
    """Build the minimal baseline payload from an evaluation result."""
    answers: list[dict[str, str]] = []
    for sample in result.sample_outputs or []:
        question = str(sample.get("question", ""))
        generated_answer = sample.get("actual")
        if generated_answer is None:
            generated_answer = sample.get("full_output", "")
        answers.append(
            {
                "question": question,
                "generated_answer": str(generated_answer),
            }
        )

    return {
        "accuracy": float(result.accuracy),
        "syntax_rate": float(result.syntax_rate),
        "answers": answers,
    }


def save_minimal_baseline_json(result: EvaluationResult, json_path: Path) -> Path:
    """Write a minimal baseline JSON file and return its path."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_minimal_baseline_record(result)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    return json_path
