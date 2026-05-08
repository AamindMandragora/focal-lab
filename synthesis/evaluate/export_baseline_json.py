"""Export a minimal baseline JSON file from a synthesis success report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_minimal_payload(report: dict) -> dict:
    evaluation = report.get("evaluation_result") or {}
    samples = report.get("sample_outputs") or []

    answers = []
    for sample in samples:
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
        "accuracy": float(evaluation.get("accuracy", 0.0)),
        "syntax_rate": float(evaluation.get("syntax_rate", 0.0)),
        "answers": answers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a minimal baseline JSON (accuracy, syntax_rate, answers)"
    )
    parser.add_argument(
        "--success-report",
        type=Path,
        required=True,
        help="Path to success_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output baseline JSON path (for example outputs/baselines/<strategy>/<model>/<benchmark>.json)",
    )
    args = parser.parse_args()

    report = json.loads(args.success_report.read_text())
    payload = _build_minimal_payload(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote baseline JSON: {args.output}")


if __name__ == "__main__":
    main()
