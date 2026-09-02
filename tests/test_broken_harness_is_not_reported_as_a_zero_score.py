"""A broken harness must not be reported as a score of zero.

The bug this pins
-----------------
Spider scored 0% accuracy AND 0% syntax on every iteration. The cause was not
the model, the grammar, or the strategy. It was this:

  1. evaluator.py routes some datasets through a parallel worker pool:
         POOLABLE_DATASETS = {"gsm_symbolic", "spider"}          (evaluator.py:156)

  2. that branch imports the pool with no guard:
         from synthesis.scripts.eval_worker_pool import get_synthesis_eval_pool

  3. synthesis/scripts/eval_worker_pool.py DOES NOT EXIST on this branch --
     nor do its two siblings, sharded_eval_core.py and eval_worker_main.py.
     They exist on other branches; they are absent here.

  4. ModuleNotFoundError is an Exception, so the broad `except Exception` that
     wraps the whole method catches it and returns:
         accuracy=0.0, syntax_rate=0.0, num_examples=0     (evaluator.py:3062)

So a missing file was displayed as "the model got everything wrong". Nothing
downstream can tell those two apart, and no amount of fixing the decoder,
grammar, or prompt could ever have moved the number, because not one example
was ever evaluated.

`num_examples=0` is the tell: a real 0% run still evaluates its examples.

What must be true instead
-------------------------
The pool is a speed optimisation -- it exists because loading the vLLM engine
takes ~24s and the sequential path reloads it each iteration. There is already
a working sequential path in the same method's `else` branch. So a missing or
broken pool must degrade to the slower correct path, not to a fake zero.
"""

from __future__ import annotations

import importlib.util

import pytest


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def test_falling_back_to_the_sequential_path_is_announced_loudly(capsys):
    """Degrading to the slow path must be visible, never silent.

    Silence is how the original bug survived: the harness quietly produced
    0% and everyone read it as a modelling problem. Whatever the pool's
    availability, the run must say which path it took, so a slow run is
    recognisable as "the optimisation is missing" rather than a mystery.
    """
    from synthesis.evaluate.evaluator import _resolve_eval_pool_loader

    loader = _resolve_eval_pool_loader()
    printed = capsys.readouterr().out

    if loader is not None:
        # The pool is present; nothing to announce.
        return

    assert printed.strip(), (
        "The eval worker pool is unavailable and evaluation silently dropped "
        "to the sequential path. Say so on stdout: a silent downgrade is how "
        "the original fake-0% bug went unnoticed."
    )
    assert "sequential" in printed.lower(), (
        f"The fallback message must name the path actually taken. Got: {printed!r}"
    )


def test_the_pool_is_reported_as_present_or_absent_but_never_guessed():
    """Records the known gap: the pool modules are absent on this branch.

    synthesis/scripts/eval_worker_pool.py (and its siblings sharded_eval_core.py
    and eval_worker_main.py) exist on other branches but not this one. With the
    fallback in place that costs SPEED, not correctness -- the sequential path
    reloads the vLLM engine (~24s) each iteration.

    This test does not demand the pool exist. It pins the thing that must stay
    true either way: what the code believes about the pool matches reality.
    If someone restores the modules, this keeps passing.
    """
    from synthesis.evaluate.evaluator import POOLABLE_DATASETS, _resolve_eval_pool_loader

    assert POOLABLE_DATASETS, "no datasets are pooled; this test needs updating"

    believed_available = _resolve_eval_pool_loader() is not None
    really_available = _module_exists("synthesis.scripts.eval_worker_pool")

    assert believed_available == really_available, (
        f"The code believes the eval pool is "
        f"{'available' if believed_available else 'unavailable'}, but it is "
        f"really {'available' if really_available else 'unavailable'}. Those "
        "must agree, or evaluation picks a path that cannot run."
    )


def test_an_unavailable_pool_degrades_to_the_sequential_path():
    """The choice of eval path must not be able to zero the score.

    `_resolve_eval_pool_loader` returns the pool's entry point, or None when
    the pool is unavailable. None means "use the sequential path", which is
    slower but correct. It must never raise, because raising here is caught
    far upstream and laundered into a 0% result.
    """
    from synthesis.evaluate.evaluator import _resolve_eval_pool_loader

    loader = _resolve_eval_pool_loader()

    assert loader is None or callable(loader), (
        f"Expected the pool entry point or None, got {loader!r}."
    )


def test_choosing_the_eval_path_never_raises(monkeypatch):
    """Even if importing the pool blows up in a new way, evaluation continues."""
    import synthesis.evaluate.evaluator as evaluator_module

    def _explode(*_args, **_kwargs):
        raise RuntimeError("simulated: pool import blew up in some new way")

    monkeypatch.setattr(importlib, "import_module", _explode)

    try:
        loader = evaluator_module._resolve_eval_pool_loader()
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"Choosing the eval path raised {type(exc).__name__}: {exc}. That "
            "exception is caught upstream and reported as accuracy=0.0, which "
            "hides the real failure."
        )

    assert loader is None, "A pool that cannot be imported must mean 'go sequential'."
