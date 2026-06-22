"""Tests for FIX A: per-example timeout is scored as failure and eval continues.

New behavior (2026-06-12):
- A timed-out example is recorded with accuracy=0, syntax=0 (already true).
- Eval continues to the next example instead of stopping immediately.
- A hard cap of 10 timed-out examples stops the eval early, but those 10
  examples still count as failures in the reported scores.
- The feedback summary reports how many examples timed out.

Test strategy: AST + exec (no vllm/torch imports needed).
"""
import ast
import time
from pathlib import Path

EVALUATOR = Path(__file__).resolve().parent.parent / "synthesis" / "evaluate" / "evaluator.py"


# ---------------------------------------------------------------------------
# Helper: extract and exec just the timer-class code from evaluator.py
# ---------------------------------------------------------------------------

def _get_timer_classes():
    src = EVALUATOR.read_text()
    tree = ast.parse(src)
    pieces = [
        ast.get_source_segment(src, n)
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name in ("PerExampleTimeout", "_PerExampleTimer")
    ]
    assert len(pieces) == 2, "timer classes not found in evaluator.py"
    ns = {}
    exec("import signal\nfrom typing import Optional\n" + "\n\n".join(p for p in pieces if p), ns)
    return ns["PerExampleTimeout"], ns["_PerExampleTimer"]


# ---------------------------------------------------------------------------
# AST tests — structural checks without loading heavy deps
# ---------------------------------------------------------------------------

def _src():
    return EVALUATOR.read_text()


def _find_eval_fn(src: str):
    """Return the AST node for the method that contains the eval example loop."""
    tree = ast.parse(src)
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            calls = {
                (c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", ""))
                for c in ast.walk(fn)
                if isinstance(c, ast.Call)
            }
            if "run_crane_csd" in calls:
                return fn
    return None


def test_timed_out_does_not_immediately_stop_eval():
    """After a PerExampleTimeout, the old unconditional early-stop branch must be gone.

    Old code pattern to look for (must NOT exist):
        if timed_out and early_stop_enabled:
            reason = "..."
            return build_result(reason)

    We check this structurally: inside the eval loop, there must be no If whose
    test is (timed_out AND early_stop_enabled) that leads to a return call.
    """
    src = _src()
    tree = ast.parse(src)
    fn = _find_eval_fn(src)
    assert fn is not None, "eval loop function not found"

    def _names_in_test(if_node: ast.If) -> set:
        return {
            (n.id if isinstance(n, ast.Name) else "")
            for n in ast.walk(if_node.test)
        }

    def _has_return_in_body(if_node: ast.If) -> bool:
        return any(isinstance(n, ast.Return) for n in ast.walk(ast.Module(body=if_node.body, type_ignores=[])))

    # Look for: if timed_out and early_stop_enabled: ... return build_result(...)
    # This pattern is the one we removed.
    offending = []
    for node in ast.walk(fn):
        if isinstance(node, ast.If):
            names = _names_in_test(node)
            if "timed_out" in names and "early_stop_enabled" in names and _has_return_in_body(node):
                offending.append(node)

    assert not offending, (
        f"Found {len(offending)} place(s) where a timed-out example still causes "
        f"an immediate early-stop return gated on early_stop_enabled. "
        f"This block must be removed so eval continues after a timeout."
    )


def test_timeout_cap_constant_present():
    """The 10-timeout pathological-strategy cap must exist in the source."""
    src = _src()
    # Check that the cap value 10 is referenced near "timeout" in some form.
    # We look for the string in the source to avoid false positives.
    assert "10" in src, "Source does not contain the cap value 10 at all"
    # More specifically: there should be a variable or comment referencing a timeout cap.
    # Accept any of these patterns:
    assert (
        "_MAX_TIMEOUT_CAP" in src
        or "timeout_cap" in src
        or "max_timeout" in src
        or "_TIMEOUT_HARD_CAP" in src
        or "_MAX_TIMEOUTS" in src
        or "max_timeouts" in src
    ), (
        "Could not find a named timeout-cap constant or variable in evaluator.py. "
        "Add a named constant (e.g. _MAX_TIMEOUTS = 10) for the pathological-strategy guard."
    )


def test_timeout_cap_stops_with_failure_count_in_reason():
    """When the timeout cap fires, the early-stop reason must mention the timeout count and total."""
    src = _src()
    # The reason string produced when the cap fires should mention timed-out examples.
    # Look for a string template that references the timeout counter and total in the
    # early-stop reason context.
    assert "timed out" in src or "timeout" in src.lower(), (
        "No 'timed out' / 'timeout' message found in evaluator.py"
    )
    # The feedback must report N/total timeouts — look for something like
    # "f"...{n_timeouts}..." near "timed" or similar patterns.
    assert (
        "n_timeouts" in src
        or "timeout_count" in src
        or "num_timeouts" in src
        or "timed_out_count" in src
        or "_timeout_counter" in src
    ), (
        "Could not find a timeout counter variable in evaluator.py. "
        "The eval loop must count timed-out examples to report in the early-stop reason."
    )


def test_feedback_summary_reports_timeout_count():
    """get_feedback_summary must report the number of timed-out examples."""
    src = _src()
    tree = ast.parse(src)
    # Find get_feedback_summary method
    feedback_fn = None
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == "get_feedback_summary":
            feedback_fn = fn
            break
    assert feedback_fn is not None, "get_feedback_summary not found in evaluator.py"

    fn_src = ast.get_source_segment(src, feedback_fn) or ""
    assert (
        "timeout" in fn_src.lower() or "timed" in fn_src.lower()
    ), (
        "get_feedback_summary does not mention timeouts. "
        "It should report how many examples timed out (e.g. '7/100 examples timed out')."
    )


# ---------------------------------------------------------------------------
# Mechanism test: timer still works (sanity check that we didn't break the timer)
# ---------------------------------------------------------------------------

def test_timer_still_interrupts_spin():
    """Sanity check: _PerExampleTimer still raises PerExampleTimeout."""
    PerExampleTimeout, _PerExampleTimer = _get_timer_classes()
    start = time.time()
    try:
        with _PerExampleTimer(0.3):
            while True:
                pass
    except PerExampleTimeout:
        pass
    else:
        raise AssertionError("timer did not interrupt the spin")
    assert time.time() - start < 5


# ---------------------------------------------------------------------------
# Integration-level AST: early_stop_reason_if_any must reference timeout count
# ---------------------------------------------------------------------------

def test_early_stop_reason_references_timeout_total():
    """The early-stop-reason string produced for the timeout cap must mention N and total.

    We look for a string template that interpolates both the counted timeouts and
    the planned total inside the body of evaluate_sample or a helper it calls.
    """
    src = _src()
    tree = ast.parse(src)
    fn = _find_eval_fn(src)
    assert fn is not None
    fn_src = ast.get_source_segment(src, fn) or ""
    # The reason string should say something like "10 timed-out; N=<n> of <total>"
    # or "eval stopped early after 10 timeouts".
    # At minimum, the counted-timeouts variable and planned_num_examples (or similar)
    # must both appear in the function body.
    has_timeout_var = (
        "n_timeouts" in fn_src
        or "timeout_count" in fn_src
        or "num_timeouts" in fn_src
        or "timed_out_count" in fn_src
        or "_timeout_counter" in fn_src
    )
    has_total_ref = "planned_num_examples" in fn_src or "len(dataset)" in fn_src
    assert has_timeout_var, "eval function does not count timeouts"
    assert has_total_ref, "eval function does not reference planned_num_examples for the reason string"
