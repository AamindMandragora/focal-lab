"""Accuracy parsing when a crashed run restarts attempt numbering in one log."""

from scripts.runtime.zero_acc_babysitter.accuracy import (
    attempt_block,
    parse_finished_attempt_accuracy,
)

RESTARTED_LOG = """\
Attempt 1/40
some synthesis output
[vllm] Engine init failed: RuntimeError: Engine core initialization failed.
Attempt 1/40
more synthesis output
  ✓ Evaluation passed:
    Accuracy: 2.0%
    Syntax: 100.0%
SUCCESS after 1 attempt(s)
"""


def test_attempt_block_returns_last_duplicate_block():
    block = attempt_block(RESTARTED_LOG, 1)
    assert block is not None
    assert "Accuracy: 2.0%" in block
    assert "Engine init failed" not in block


def test_parse_finished_attempt_accuracy_uses_the_restarted_run():
    result = parse_finished_attempt_accuracy(RESTARTED_LOG, 1, finished=True)
    assert result.accuracy_pct == 2.0
