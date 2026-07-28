"""A grammar that fails to load must not be reported as 100% valid syntax.

The defect
----------
`Evaluator._check_syntax_validity` (evaluator.py:2414-2423):

    try:
        parser = self._get_syntax_parser(example)
        for match in matches:
            try:
                parser.parse(match.strip())
                segments.append((match, True))
            except LarkError:
                segments.append((match, False))
    except Exception:
        return True, [(m, True) for m in matches]   # <-- every segment "valid"

The inner `except LarkError` is correct and narrow: that segment really did fail
to parse. The OUTER catch is the problem. It wraps `_get_syntax_parser`, so if
the parser cannot be built at all -- a missing grammar file, a syntax error in
the grammar, a bad import -- every segment is marked valid and the method
reports success.

Nothing was checked. The result is 100% syntactic validity.

Why this one is worse than a fake zero
--------------------------------------
A fake 0% gets investigated, which is how the Spider bug was eventually found.
A fake 100% is the number you were hoping for. It confirms the hypothesis, so
it goes in the results table and nobody asks it any questions.

Unchanged since the file's first commit.

Why raising is the right fix, not a safer default
-------------------------------------------------
There is no honest value to return here. The contract is (all_valid, segments),
which has no way to say "I could not check". Returning True claims validity that
was never tested; returning False invents failures that were never observed.

A parser that will not build is a setup fault of the same family as the missing
module that caused the Spider zero -- it must stop the run, not produce a
number. This path is only reached for datasets that have a real grammar: smiles
and gsm_symbolic both return earlier (evaluator.py:2362, 2401), so nothing
legitimately arrives here without a parser.
"""

from __future__ import annotations

import pytest


def _evaluator_with(parser, segments):
    """A bare Evaluator wired with just what _check_syntax_validity touches.

    Built with object.__new__ so no model, GPU, or dataset is needed -- the
    method under test only reads dataset_name and calls two helpers.
    """
    from synthesis.evaluate.evaluator import Evaluator

    evaluator = object.__new__(Evaluator)
    evaluator.dataset_name = "spider"
    evaluator._extract_constrained_content = lambda output: segments
    if isinstance(parser, BaseException):
        def explode(example=None):
            raise parser
        evaluator._get_syntax_parser = explode
    else:
        evaluator._get_syntax_parser = lambda example=None: parser
    return evaluator


class _ParserThatWorks:
    def parse(self, text):
        return True


class _ParserThatRejects:
    """Stands in for a real grammar rejecting a genuinely malformed segment."""

    def parse(self, text):
        from lark.exceptions import LarkError

        raise LarkError("unexpected token")


def test_a_grammar_that_will_not_load_does_not_report_perfect_syntax():
    """The core rule: unable to check is not the same as checked and passed."""
    evaluator = _evaluator_with(
        FileNotFoundError("spider.lark: no such file"),
        ["SELECT * FROM t"],
    )

    with pytest.raises(Exception) as caught:
        evaluator._check_syntax_validity("SELECT * FROM t", example={})

    assert not isinstance(caught.value, AssertionError), (
        "The grammar failed to load and the method returned normally instead of "
        "raising. Every segment was marked valid, so this run reports 100% "
        "syntactic validity without a single segment having been parsed."
    )


def test_the_original_cause_survives_so_it_can_be_diagnosed():
    """Replacing the cause with a generic error repeats the original mistake.

    The Spider zero cost weeks partly because the real error text never reached
    anyone. Whatever stopped the parser from loading must still be readable.
    """
    evaluator = _evaluator_with(
        FileNotFoundError("spider.lark: no such file"),
        ["SELECT * FROM t"],
    )

    with pytest.raises(FileNotFoundError, match="spider.lark"):
        evaluator._check_syntax_validity("SELECT * FROM t", example={})


def test_a_working_parser_still_marks_good_segments_valid():
    """Guard against over-fixing: normal grading must be untouched."""
    evaluator = _evaluator_with(_ParserThatWorks(), ["SELECT * FROM t"])

    all_valid, segments = evaluator._check_syntax_validity("...", example={})

    assert all_valid is True
    assert segments == [("SELECT * FROM t", True)]


def test_a_segment_the_grammar_rejects_is_still_marked_invalid():
    """The inner narrow catch must keep working: this is a real parse failure,
    not a broken setup, and it should produce a result rather than raise."""
    evaluator = _evaluator_with(_ParserThatRejects(), ["SELECT FROM WHERE"])

    all_valid, segments = evaluator._check_syntax_validity("...", example={})

    assert all_valid is False
    assert segments == [("SELECT FROM WHERE", False)]


def test_no_constrained_segments_is_still_not_an_error():
    """Existing early return at evaluator.py:2411 must stay as it is."""
    evaluator = _evaluator_with(_ParserThatWorks(), [])

    all_valid, segments = evaluator._check_syntax_validity("...", example={})

    assert all_valid is True
    assert segments == []
