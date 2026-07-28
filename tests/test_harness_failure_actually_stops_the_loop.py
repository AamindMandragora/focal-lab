"""Classifying a harness failure is useless unless the loop acts on it.

Why this is separate from test_harness_failure_is_not_blamed_on_the_strategy
---------------------------------------------------------------------------
That file tests `classify_eval_failure` on its own: given a result with zero
examples, does it say HARNESS? This file tests the thing that actually costs
money -- whether the retry loop *uses* the answer.

A classifier that returns HARNESS correctly, while the loop below it still
falls through to "Refining based on evaluation error...", changes nothing. The
missing module still gets sent to the strategy-writing model, every attempt,
at full API price. So the classifier passing its own test is not evidence the
bug is fixed.

Why this test reads the source instead of running the loop
----------------------------------------------------------
The branch lives inside a long method that needs a configured evaluator, a
model, and a run directory to reach. Standing all that up would test the
scaffolding more than the rule. So this checks the shape of the code, which is
weaker but honest about it, and pins the two things that can regress:

  1. the HARNESS branch raises (stops the run) rather than logging and going on
  2. it comes BEFORE the refinement call, so refinement is unreachable for a
     harness failure

`test_the_check_rejects_the_old_broken_shape` feeds the checker the code as it
was before the fix. If that pre-fix code passes, the checker is not checking
anything and the green result here is meaningless.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_LOOP = REPO_ROOT / "synthesis" / "evaluate" / "feedback_loop.py"

REFINE_MARKER = "Refining based on evaluation error"


def _is_eval_failed_test(node: ast.expr) -> bool:
    """Match the `if not eval_result.success:` condition."""
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Attribute)
        and node.operand.attr == "success"
        and isinstance(node.operand.value, ast.Name)
        and node.operand.value.id == "eval_result"
    )


def _mentions(node: ast.AST, needle: str) -> bool:
    """True if any string constant or attribute chain under `node` mentions it."""
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if needle in child.value:
                return True
        if isinstance(child, ast.Attribute) and child.attr == needle:
            return True
    return False


def check_harness_failure_stops_the_loop(source: str) -> tuple[bool, str]:
    """Does the eval-failure block stop on a harness failure before refining?

    Returns (ok, reason). Kept as a plain function so the test below can run it
    against the pre-fix code and confirm it says no.
    """
    tree = ast.parse(source)

    blocks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and _is_eval_failed_test(node.test)
    ]
    if not blocks:
        return False, "found no `if not eval_result.success:` block at all"

    for block in blocks:
        harness_at = None
        refine_at = None
        for index, statement in enumerate(block.body):
            if (
                harness_at is None
                and isinstance(statement, ast.If)
                and _mentions(statement.test, "HARNESS")
            ):
                harness_at = index
                raises = any(
                    isinstance(inner, ast.Raise) for inner in ast.walk(statement)
                )
                if not raises:
                    return False, (
                        "the HARNESS branch does not raise, so the loop carries "
                        "on and refines against a broken harness anyway"
                    )
            if refine_at is None and _mentions(statement, REFINE_MARKER):
                refine_at = index

        if harness_at is None:
            return False, (
                "the eval-failure block never branches on FailureStage.HARNESS, "
                "so a missing module is handled exactly like a bad strategy"
            )
        if refine_at is not None and harness_at > refine_at:
            return False, (
                f"refinement (statement {refine_at}) happens before the HARNESS "
                f"check (statement {harness_at}), so the model is asked to fix "
                "the missing module before anyone notices it is a harness fault"
            )
        return True, "harness failure stops the loop before refinement"

    return False, "no eval-failure block was checkable"


OLD_BROKEN_SHAPE = '''
def run(self, eval_result, attempt, attempts):
    if not eval_result.success:
        print(f"  x Evaluation failed: {eval_result.error}")
        attempt.failed_at = FailureStage.EVALUATION
        attempt.error_summary = eval_result.error or "Evaluation failed"
        attempts.append(attempt)
        self._unload_evaluator_runtime_before_refinement()
        print("  Refining based on evaluation error...")
        evaluation_feedback = eval_result.get_feedback_summary(True)
        continue
'''


def test_the_check_rejects_the_old_broken_shape():
    """Guard the guard.

    This is the code that shipped before the fix. If the checker calls it fine,
    then the checker cannot tell the fix from its absence, and the real test
    below is decoration.
    """
    ok, reason = check_harness_failure_stops_the_loop(OLD_BROKEN_SHAPE)

    assert not ok, (
        "The checker approved the pre-fix code, which sent missing-module "
        "errors to the strategy-writing model. It is not detecting anything, "
        f"so a green result from it means nothing. It said: {reason}"
    )


def test_a_harness_failure_stops_the_run_instead_of_being_refined():
    source = FEEDBACK_LOOP.read_text()

    ok, reason = check_harness_failure_stops_the_loop(source)

    assert ok, (
        f"{FEEDBACK_LOOP.relative_to(REPO_ROOT)} does not stop on a harness "
        f"failure: {reason}.\n\n"
        "Consequence: an evaluation that ran zero examples -- a missing Python "
        "file, not a bad strategy -- is fed back to the strategy-writing model "
        "as though the strategy were at fault. Every attempt pays full API cost "
        "to rewrite a strategy against a fault no strategy can fix."
    )
