"""Tests for SynthesisPipeline._pareto_best_prunable_helpers (A3/A4 fix).

The bandit protects (never prunes) the helpers used by its "best" attempt.
Old behavior seeded that keep-set from the SCALAR-best attempt, while the
author is told to beat the PARETO-best (threshold-shortfall) anchor. When the
two differ, the protected helpers belonged to the wrong branch. The fix seeds
the keep-set from the pareto-best attempt -- the same anchor shown to the
author -- so protection matches the branch being chased. Falls back to the
scalar-best attempt when no pareto anchor exists.

Called unbound on a duck-typed stand-in.
"""
from types import SimpleNamespace

from synthesis.evaluate.feedback_loop import SynthesisPipeline


def _pipe(helper_map):
    fake = SimpleNamespace(
        min_accuracy=0.9,
        min_syntax_rate=0.9,
        require_delimiters=False,
        eval_max_seconds_per_example=None,  # scalar = 3*acc + syn + 0.5
    )
    fake._get_helper_calls_for_evaluation_history = lambda code: helper_map[code]
    fake._evaluation_scalar_score = lambda result: SynthesisPipeline._evaluation_scalar_score(fake, result)
    fake._compute_pareto_best = lambda attempts: SynthesisPipeline._compute_pareto_best(fake, attempts)
    return fake


def _att(number, code, accuracy, syntax_rate, num_examples=300):
    return SimpleNamespace(
        attempt_number=number,
        strategy_code=code,
        eval_result=SimpleNamespace(
            accuracy=accuracy, syntax_rate=syntax_rate, num_examples=num_examples,
            contains_delimiters=False, max_sample_time_seconds=0.0, sample_outputs=[],
        ),
    )


def _seed(pipe, attempts, pool):
    return SynthesisPipeline._pareto_best_prunable_helpers(pipe, attempts, pool)


def test_seeds_from_pareto_best_not_scalar_best():
    # P: balanced, closest to thresholds (shortfall 0.10) -> pareto-best.
    # S: very high accuracy but low syntax (shortfall 0.45) -> scalar-best (3x acc).
    helper_map = {"p": {"ParetoOnly"}, "s": {"ScalarOnly"}}
    pipe = _pipe(helper_map)
    P = _att(1, "p", 0.80, 1.00)   # shortfall 0.10 ; scalar 3.90
    S = _att(2, "s", 1.00, 0.45)   # shortfall 0.45 ; scalar 3.95

    # Sanity: confirm the two anchors really diverge.
    assert pipe._compute_pareto_best([P, S])[0] == 1, "pareto-best must be P"
    assert max([P, S], key=lambda a: pipe._evaluation_scalar_score(a.eval_result)) is S

    pool = ["ParetoOnly", "ScalarOnly"]
    seed = _seed(pipe, [P, S], pool)
    assert seed == {"ParetoOnly"}, f"expected pareto-best's helper, got {seed}"


def test_falls_back_to_scalar_best_when_no_pareto_anchor():
    # No attempt has a completed eval (num_examples 0) -> pareto returns None.
    helper_map = {"x": {"HelperX"}}
    pipe = _pipe(helper_map)
    X = _att(1, "x", 0.5, 0.5, num_examples=0)
    assert pipe._compute_pareto_best([X])[0] is None
    seed = _seed(pipe, [X], ["HelperX"])
    assert seed == {"HelperX"}, "fallback must use scalar-best helpers"


if __name__ == "__main__":
    test_seeds_from_pareto_best_not_scalar_best()
    test_falls_back_to_scalar_best_when_no_pareto_anchor()
    print("All tests passed.")
