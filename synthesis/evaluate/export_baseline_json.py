"""Export a minimal baseline JSON file from a synthesis success report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthesis.evaluate.baseline_store import baseline_payload_from_success_report


def _build_minimal_payload(report: dict) -> dict:
    return baseline_payload_from_success_report(report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a minimal baseline JSON (accuracy, syntax_rate, metrics, answers)"
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
