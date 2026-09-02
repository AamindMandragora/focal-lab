"""The one error meaning "this example cannot be graded at all"."""

from __future__ import annotations


class UngradableExample(ValueError):
    """A dataset row that cannot be scored the way the benchmark scores.

    Raised when a field the grader needs is missing or unreadable -- a GSM row
    with no `variable_types`, or a `variable_types` that is not a mapping. It is
    not "the model got this wrong". There is no verdict to record, so the run
    stops instead of averaging in a number produced by guessing.

    `Evaluator._evaluate_one_example` re-raises this by name past its blanket
    catch. That is why it exists as its own class rather than the plain
    TypeError and ValueError these graders used to raise: re-raising those by
    type would also re-raise every incidental bug in the generation path (a
    None where a string was expected, a `float("abc")`), and one ordinary bug
    would abort an entire evaluation run.

    It subclasses ValueError so any existing `except ValueError` behaves
    exactly as it did before.
    """
