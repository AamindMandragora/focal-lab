"""What the author is TOLD about the decoding surface must match what happens.

The bug this guards
-------------------
Generation can reach the constrained region two ways:

  - visible-delimiter surface: generation starts outside the constrained
    region, and the strategy calls `OpenConstrainedSpan`, which puts a literal
    `<<` into the output.
  - observed surface: generation starts already inside, `EnterObservedConstrainedSpan`
    emits nothing, and no `<<` is ever produced by anyone.

Spider runs on the observed surface. Authors kept writing strategies whose only
route into constrained mode was `if next == "<<"`, which on Spider never fires,
so nothing was ever constrained.

The fix tells the author which surface it is on. But that only helps if the
claim is TRUE. Today two separate pieces of code decide it:

  - `starts_inside_constrained()`      -> what the author's prompt is told
  - `get_generation_runner()`          -> what evaluation actually does

They are kept in agreement by a code comment and nothing else. A comment does
not fail a build. If they ever drift, the author gets confidently told the
wrong surface and writes a strategy that cannot work -- the same silent
failure as before, now wearing a fix.

So this file does not check the prompt wording (that is covered elsewhere). It
checks the claim against the behaviour, under both settings of the mode switch.
"""

from __future__ import annotations

import importlib

import pytest


TOKEN0_ENV = "SPIDER_TOKEN0_CONSTRAINED"


def _spider_eval_logic():
    """Freshly imported, so a changed env var is actually picked up."""
    module = importlib.import_module(
        "synthesis.evaluate.benchmarks.sql_spider.eval_logic"
    )
    return importlib.reload(module)


def _surface_actually_used(monkeypatch) -> bool:
    """Run the real generation runner and observe what it asks for.

    Returns whether `start_inside_constrained` was actually requested. The
    runner imports `run_crane_csd` when it is called, not at module load, so
    replacing it on the generation module beforehand is enough to intercept
    the call without loading a model.
    """
    eval_logic = _spider_eval_logic()
    generation = importlib.import_module(
        "synthesis.evaluate.benchmarks.sql_spider.generation"
    )

    seen: dict = {}

    def _capture(*args, **kwargs):
        seen.update(kwargs)
        return ("", 0, 0.0, [], [])

    monkeypatch.setattr(generation, "run_crane_csd", _capture)

    runner = eval_logic.get_generation_runner()
    runner()
    return bool(seen.get("start_inside_constrained", False))


@pytest.mark.parametrize(
    "token0_setting, expected_surface",
    [
        ("1", True),   # default mode: constrained from token 0, no visible "<<"
        ("0", False),  # legacy mode: visible "<<" ... ">>" span
    ],
)
def test_the_author_is_told_the_surface_that_is_actually_used(
    monkeypatch, token0_setting, expected_surface
):
    monkeypatch.setenv(TOKEN0_ENV, token0_setting)

    claimed = _spider_eval_logic().starts_inside_constrained()
    actual = _surface_actually_used(monkeypatch)

    assert claimed == actual, (
        f"With {TOKEN0_ENV}={token0_setting}, the author's prompt is told "
        f"start_inside_constrained={claimed}, but evaluation actually runs with "
        f"start_inside_constrained={actual}. The author will write a strategy "
        "for the wrong decoding surface and silently constrain nothing."
    )
    assert actual == expected_surface, (
        f"{TOKEN0_ENV}={token0_setting} was expected to produce "
        f"start_inside_constrained={expected_surface}, but produced {actual}. "
        "If this mode's meaning changed on purpose, update this test and the "
        "author prompt together -- they must not drift apart."
    )


def test_the_benchmark_registry_exposes_the_surface_to_the_feedback_loop():
    """The feedback loop finds this hook by name through the registry.

    It looks the benchmark up with `get_logic(dataset_name)` and then reads
    `starts_inside_constrained` off it with getattr, falling back to False when
    absent. A rename would therefore not raise -- it would quietly report the
    visible-delimiter surface for every benchmark, which is the pre-fix bug.
    """
    from synthesis.evaluate.benchmarks.registry import get_logic

    logic = get_logic("spider")
    hook = getattr(logic, "starts_inside_constrained", None)

    assert hook is not None, (
        "spider no longer exposes starts_inside_constrained(). The feedback "
        "loop's getattr lookup will silently fall back to False and tell every "
        "author it is on the visible-delimiter surface."
    )
    assert isinstance(hook(), bool)
