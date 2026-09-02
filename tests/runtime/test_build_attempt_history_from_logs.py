from pathlib import Path
import importlib.util
import sys


MODULE_PATH = Path(__file__).parents[2] / "scripts/runtime/build_attempt_history_from_logs.py"
SPEC = importlib.util.spec_from_file_location("build_attempt_history", MODULE_PATH)
history = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = history
SPEC.loader.exec_module(history)


def test_parse_logs_restores_evaluated_attempt_before_replay(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        """Attempt 3/40
Strategy: method Main() {
}

[1/4] Verifying with Dafny...
  ✗ Evaluation below threshold:
    Accuracy: 42.9% (min: 59.2%)
    Contains << >>: yes (required: yes)
    Syntax: 87.8% (min: 85.0%)
Attempt 4/40
Strategy: method Main() {}

[1/4] Verifying with Dafny...
"""
    )

    records = history.parse_logs([log], before_attempt=4, num_examples=49)

    assert [record["attempt_number"] for record in records] == [3]
    assert records[0]["num_correct"] == 21
    assert records[0]["syntax_rate"] == 0.878
