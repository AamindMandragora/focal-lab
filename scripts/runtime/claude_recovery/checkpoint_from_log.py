#!/usr/bin/env python3
"""Build a restart checkpoint from completed attempt blocks in a live log."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path


LOGGER = logging.getLogger("claude-recovery-checkpoint")
_ATTEMPT_START = re.compile(r"(?m)^Attempt (?P<number>\d+)/(?P<total>\d+)\s*$")
_STRATEGY = re.compile(r"(?ms)^Strategy: (?P<strategy>.*?)(?=\n\[1/4\] Verifying with Dafny\.\.\.)")
_ACCURACY = re.compile(r"(?m)^\s+Accuracy: (?P<percent>[0-9.]+)%")
_SYNTAX = re.compile(r"(?m)^\s+Syntax: (?P<percent>[0-9.]+)%")
_DELIMITERS = re.compile(r"(?m)^\s+Contains << >>: (?P<value>yes|no)")


def _attempt_blocks(log_text: str) -> dict[int, list[str]]:
    matches = list(_ATTEMPT_START.finditer(log_text))
    blocks: dict[int, list[str]] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(log_text)
        blocks.setdefault(int(match.group("number")), []).append(
            log_text[match.start():end]
        )
    return blocks


def _strategy(block: str, attempt: int) -> str:
    match = _STRATEGY.search(block)
    if match is None:
        raise ValueError(f"attempt {attempt} has no complete strategy block")
    return match.group("strategy").rstrip()


def _evaluated_record(
    block: str,
    attempt: int,
    num_examples: int,
    *,
    require_delimiters: bool = True,
) -> dict:
    accuracy = _ACCURACY.search(block)
    syntax = _SYNTAX.search(block)
    delimiters = _DELIMITERS.search(block)
    if accuracy is None or syntax is None or (require_delimiters and delimiters is None):
        raise ValueError(f"attempt {attempt} has no complete evaluation summary")
    num_correct = round(float(accuracy.group("percent")) * num_examples / 100)
    num_syntax = round(float(syntax.group("percent")) * num_examples / 100)
    return {
        "attempt_number": attempt,
        "strategy_code": _strategy(block, attempt),
        "accuracy": num_correct / num_examples,
        "contains_delimiters": (
            delimiters is not None and delimiters.group("value") == "yes"
        ),
        "syntax_rate": num_syntax / num_examples,
        "num_examples": num_examples,
        "num_correct": num_correct,
        "timestamp": "restored-from-live-log",
    }


def build_checkpoint(
    *,
    log_path: Path,
    prior_history_path: Path,
    first_finished_attempt: int,
    last_finished_attempt: int,
    active_attempt: int,
    num_examples: int,
) -> tuple[list[dict], str]:
    """Return merged evaluated history and the already-authored active strategy."""
    prior = json.loads(prior_history_path.read_text(encoding="utf-8"))
    blocks = _attempt_blocks(log_path.read_text(encoding="utf-8", errors="replace"))
    restored = []
    for attempt in range(first_finished_attempt, last_finished_attempt + 1):
        for block in reversed(blocks.get(attempt, [])):
            try:
                restored.append(_evaluated_record(block, attempt, num_examples))
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"attempt {attempt} has no complete evaluation block")

    for block in reversed(blocks.get(active_attempt, [])):
        try:
            seed = _strategy(block, active_attempt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"attempt {active_attempt} has no complete strategy block")
    replaced = {row["attempt_number"] for row in restored}
    history = [row for row in prior if row["attempt_number"] not in replaced] + restored
    history.sort(key=lambda row: row["attempt_number"])
    actual = [row["attempt_number"] for row in history]
    if actual != sorted(set(actual)) or actual[-1] != last_finished_attempt:
        raise ValueError(
            "checkpoint history must contain unique, increasing evaluated attempts "
            f"ending at {last_finished_attempt}; got {actual}"
        )
    LOGGER.info(
        "[claude-recovery-checkpoint] source=%s history=1..%d seed_attempt=%d",
        log_path,
        last_finished_attempt,
        active_attempt,
    )
    return history, seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--prior-history", type=Path, required=True)
    parser.add_argument("--history-out", type=Path, required=True)
    parser.add_argument("--seed-out", type=Path, required=True)
    parser.add_argument("--first-finished-attempt", type=int, required=True)
    parser.add_argument("--last-finished-attempt", type=int, required=True)
    parser.add_argument("--active-attempt", type=int, required=True)
    parser.add_argument("--num-examples", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    history, seed = build_checkpoint(
        log_path=args.log,
        prior_history_path=args.prior_history,
        first_finished_attempt=args.first_finished_attempt,
        last_finished_attempt=args.last_finished_attempt,
        active_attempt=args.active_attempt,
        num_examples=args.num_examples,
    )
    args.history_out.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    args.seed_out.write_text(seed + "\n", encoding="utf-8")
    LOGGER.info(
        "[claude-recovery-checkpoint] wrote history=%s seed=%s",
        args.history_out,
        args.seed_out,
    )


if __name__ == "__main__":
    main()
