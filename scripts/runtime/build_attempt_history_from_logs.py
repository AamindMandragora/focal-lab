#!/usr/bin/env python3
"""Build approved recovery history from synthesis logs without secret data."""

import argparse
import json
import re
from pathlib import Path


ATTEMPT_RE = re.compile(r"^Attempt (\d+)/\d+$", re.MULTILINE)


def parse_logs(paths: list[Path], before_attempt: int, num_examples: int) -> list[dict]:
    records: dict[int, dict] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = list(ATTEMPT_RE.finditer(text))
        for index, match in enumerate(matches):
            attempt = int(match.group(1))
            if attempt >= before_attempt:
                continue
            block_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            block = text[match.end():block_end]
            strategy_start = block.find("Strategy: ")
            stage_start = block.find("\n\n[1/4] Verifying with Dafny...", strategy_start)
            accuracy = re.search(r"^\s*Accuracy:\s*([0-9.]+)%", block, re.MULTILINE)
            syntax = re.search(r"^\s*Syntax:\s*([0-9.]+)%", block, re.MULTILINE)
            delimiters = re.search(r"^\s*Contains << >>:\s*(yes|no)", block, re.MULTILINE)
            if min(strategy_start, stage_start) < 0 or not accuracy or not syntax:
                continue
            accuracy_value = float(accuracy.group(1)) / 100.0
            records[attempt] = {
                "attempt_number": attempt,
                "strategy_code": block[
                    strategy_start + len("Strategy: "):stage_start
                ].strip()
                + "\n",
                "accuracy": accuracy_value,
                "syntax_rate": float(syntax.group(1)) / 100.0,
                "num_examples": num_examples,
                "num_correct": round(accuracy_value * num_examples),
                "contains_delimiters": delimiters is None or delimiters.group(1) == "yes",
            }
    return [records[number] for number in sorted(records)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="append", type=Path, required=True)
    parser.add_argument("--before-attempt", type=int, required=True)
    parser.add_argument("--num-examples", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = parse_logs(args.log, args.before_attempt, args.num_examples)
    if not records:
        raise SystemExit("no evaluated attempt history recovered")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote={args.output} attempts={len(records)} "
        f"range={records[0]['attempt_number']}-{records[-1]['attempt_number']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
