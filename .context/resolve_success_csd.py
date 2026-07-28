#!/usr/bin/env python3
"""Resolve the compiled strategy for a deferred GSM success report."""

import json
import sys
from pathlib import Path


OUTPUT_NAMES = {
    "gsm-qwen35-9b": "post14b_rebar_gsm-qwen35-9b_0711",
    "gsm14b": "synth_gsm14b_z3bar_retry_0708_infraretry_kvfix_0711",
}


def resolve(repo: Path, cell: str) -> Path:
    output_name = OUTPUT_NAMES[cell]
    latest = repo / "outputs/generated" / output_name / "latest_run.txt"
    run_dir = Path(latest.read_text(encoding="utf-8").strip())
    report = json.loads((run_dir / "results/success_report.json").read_text())
    csd = Path(report["compiled_dir"]) / "GeneratedCSD.py"
    if not csd.is_file():
        raise FileNotFoundError(csd)
    return csd


if __name__ == "__main__":
    print(resolve(Path(__file__).resolve().parents[1], sys.argv[1]))
