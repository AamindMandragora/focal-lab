import json

from synthesis.run_synthesis import _load_initial_attempt_history


def test_load_initial_attempt_history_restores_metrics_and_strategy(tmp_path):
    history = tmp_path / "history.json"
    history.write_text(
        json.dumps(
            [
                {
                    "attempt_number": 3,
                    "strategy_code": "method Main() {}",
                    "accuracy": 0.42,
                    "syntax_rate": 0.91,
                    "num_examples": 49,
                    "num_correct": 21,
                    "contains_delimiters": True,
                }
            ]
        )
    )

    attempts = _load_initial_attempt_history(history)

    assert len(attempts) == 1
    assert attempts[0].attempt_number == 3
    assert attempts[0].strategy_code == "method Main() {}"
    assert attempts[0].eval_result.accuracy == 0.42
    assert attempts[0].eval_result.syntax_rate == 0.91
    assert attempts[0].eval_result.num_correct == 21
