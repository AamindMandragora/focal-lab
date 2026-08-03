"""A run that quietly stopped constraining must not claim it was constrained.

The defect
----------
`SyncodeLogitsProcessor.__call__` (grammar_decoder.py) has two paths that skip
the masking step and let the model pick any token at all:

    except Exception as e:              # :160
        ...
        continue                        # never reaches scores[idx].masked_fill

    else:                               # :182  no acceptable tokens
        self.logger.log('No acceptable tokens...')
        # masks nothing

Either one means that decoding step was UNCONSTRAINED. The grammar was not
applied. The whole point of this system is that output is grammar-constrained,
so a step that skipped the constraint is not a detail -- it is the measurement
failing.

Why this direction is the dangerous one
---------------------------------------
Unconstrained decoding does not necessarily produce bad output. The model often
writes valid SQL on its own. So a run can drop the constraint for some steps,
still score high on syntactic validity, and the number looks like evidence that
constrained decoding works. The failure makes the result look BETTER, which is
why nobody goes looking for it.

What is there now, and why it does not count
--------------------------------------------
`self.parse_failed` exists and looks like a record. It is not one:
  - it is a bool, so 1 failed step and 40,000 failed steps look identical
  - it is written in 5 places and read in exactly ONE -- its own print guard
  - nothing outside the class ever reads it, so it never reaches a result
  - the second fallback path (`no acceptable tokens`) does not set it at all

So the tell exists and is thrown away. That is the bug class this sweep is
about.

What this file pins
-------------------
1. a counter that is readable from outside, counts steps, and separates reasons
2. both fallback paths in the real source actually record into it

Test 2 reads the source rather than running the decoder, because torch is not
installed in this environment. It is guarded by a check that the pre-fix source
FAILS it, so a green result is not free.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DECODER = REPO_ROOT / "synthesis" / "evaluate" / "syncode" / "syncode" / "grammar_decoder.py"

MODULE = "synthesis.evaluate.benchmarks.common.constraint_audit"


def _audit():
    import importlib

    return importlib.import_module(MODULE).ConstraintAudit()


def test_a_fresh_run_is_considered_fully_constrained():
    audit = _audit()

    assert audit.was_fully_constrained is True
    assert audit.total_unconstrained_steps == 0


def test_one_skipped_step_is_enough_to_lose_the_claim():
    """There is no acceptable number of unconstrained steps above zero.

    The claim being made is 'output is grammar-constrained'. One step where the
    grammar was not applied makes that claim false for the sample.
    """
    audit = _audit()

    audit.record_unconstrained_step("parse_error", "unexpected token at line 1")

    assert audit.was_fully_constrained is False, (
        "A step decoded without the grammar applied, but the run still reports "
        "itself as fully constrained. Any validity rate it produces is not "
        "evidence about constrained decoding."
    )
    assert audit.total_unconstrained_steps == 1


def test_steps_are_counted_not_just_flagged():
    """A bool cannot tell a one-off hiccup from a run that never constrained.

    This is the concrete weakness of the existing `self.parse_failed` flag.
    """
    audit = _audit()

    for index in range(2500):
        audit.record_unconstrained_step("parse_error", f"step {index}")

    assert audit.total_unconstrained_steps == 2500, (
        f"Recorded {audit.total_unconstrained_steps} for 2500 unconstrained "
        "steps. Losing the count means a run that abandoned the grammar almost "
        "immediately is indistinguishable from one that slipped once."
    )


def test_the_two_causes_are_kept_apart():
    """They need different fixes, so they must not collapse into one number.

    A broken grammar file (parse_error) is a setup fault. A dead end with no
    acceptable tokens (no_valid_tokens) means the grammar and the model
    disagree about what can come next.
    """
    audit = _audit()

    audit.record_unconstrained_step("parse_error", "bad grammar")
    audit.record_unconstrained_step("no_valid_tokens", "dead end")
    audit.record_unconstrained_step("no_valid_tokens", "dead end again")

    assert audit.counts_by_reason == {"parse_error": 1, "no_valid_tokens": 2}


def test_the_summary_states_the_problem_in_words_a_human_will_notice():
    audit = _audit()
    audit.record_unconstrained_step("parse_error", "bad grammar")

    summary = audit.summary()

    assert summary, "A run that lost the constraint produced an empty summary."
    assert "1" in summary, f"The summary hides the count: {summary!r}"
    assert "parse_error" in summary, f"The summary hides the cause: {summary!r}"


def test_a_clean_run_says_so_without_crying_wolf():
    audit = _audit()

    assert audit.was_fully_constrained is True
    assert audit.counts_by_reason == {}


# ---------------------------------------------------------------------------
# The wiring: the pure counter above is worthless if the decoder ignores it.
# ---------------------------------------------------------------------------

FALLBACK_RECORDER = "record_unconstrained_step"


def _call_method(source: str) -> ast.FunctionDef | None:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "__call__":
            return node
    return None


def check_both_fallbacks_are_recorded(source: str) -> tuple[bool, str]:
    """Do both unconstrained paths in `__call__` record themselves?

    A plain function so the test below can run it against the pre-fix source and
    confirm it says no.
    """
    call = _call_method(source)
    if call is None:
        return False, "no __call__ method found in the source"

    def records(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Attribute) and child.attr == FALLBACK_RECORDER
            for child in ast.walk(node)
        )

    handlers = [n for n in ast.walk(call) if isinstance(n, ast.ExceptHandler)]
    if not handlers:
        return False, "no exception handler found in __call__"
    if not any(records(handler) for handler in handlers):
        return False, (
            "the parse-failure handler falls through to unconstrained decoding "
            f"without calling {FALLBACK_RECORDER}, so the skipped constraint "
            "leaves no trace in the results"
        )

    # The 'no acceptable tokens' path is the `else:` of the mask-sum test.
    orelses = [
        n.orelse
        for n in ast.walk(call)
        if isinstance(n, ast.If) and n.orelse
    ]
    if not any(records(stmt) for body in orelses for stmt in body):
        return False, (
            "the 'no acceptable tokens' branch masks nothing and does not call "
            f"{FALLBACK_RECORDER}. That step decoded freely and nothing records "
            "it; this path never even set the old parse_failed flag"
        )

    return True, "both unconstrained paths record themselves"


PRE_FIX_SHAPE = '''
class SyncodeLogitsProcessor:
    def __call__(self, input_ids, scores):
        partial_codes = self._get_partial_codes(input_ids)
        for idx, partial_code in enumerate(partial_codes):
            try:
                r = self.inc_parser.get_acceptable_next_terminals(partial_code)
                self.update_valid_state(partial_code, idx, r)
            except Exception as e:
                if self.dev_mode == True:
                    raise e
                elif self.parse_failed == False:
                    self.parse_failed = True
                    print("Parsing failed! Falling back to unconstrained decoding.")
                continue
            accept_mask = self.dfa_mask_store.get_accept_mask(r, logger=self.logger)
            if torch.sum(accept_mask) != 0:
                scores[idx] = scores[idx].masked_fill(~accept_mask, -float("inf"))
            else:
                self.logger.log('No acceptable tokens for the current partial code!')
                self._log_current_status(partial_code, r)
        return scores
'''


def test_the_wiring_check_rejects_the_pre_fix_decoder():
    """Guard the guard: if the old code passes, the check proves nothing."""
    ok, reason = check_both_fallbacks_are_recorded(PRE_FIX_SHAPE)

    assert not ok, (
        "The check approved the decoder as it was before the fix, where both "
        "fallbacks were silent. It is not detecting anything, so the test "
        f"below would pass for free. It said: {reason}"
    )


def test_the_real_decoder_records_every_unconstrained_step():
    ok, reason = check_both_fallbacks_are_recorded(DECODER.read_text())

    assert ok, (
        f"{DECODER.relative_to(REPO_ROOT)} can decode without the grammar and "
        f"leave no record: {reason}.\n\n"
        "Consequence: the run reports a syntactic validity rate that reads as "
        "evidence for constrained decoding, when some or all of those tokens "
        "were chosen with no constraint applied at all."
    )


def test_the_real_decoder_constructs_its_constraint_audit():
    """Recording cannot work unless every decoder instance owns a counter."""
    tree = ast.parse(DECODER.read_text())
    decoder_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SyncodeLogitsProcessor"
    )
    constructor = next(
        node
        for node in decoder_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    initializes_audit = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "constraint_audit"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "ConstraintAudit"
        for node in ast.walk(constructor)
    )

    assert initializes_audit, (
        "SyncodeLogitsProcessor records fallback steps through self.constraint_audit "
        "but never constructs that object, so the first fallback crashes the run."
    )
