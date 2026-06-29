"""Tests for choose_denominator_basis — denominator selection on early-stop."""
from synthesis.evaluate.metrics import choose_denominator_basis  # adjust import path if helper is elsewhere


def test_runtime_early_stop_uses_planned():
    # Runtime budget exceeded: denominator must be full planned count, not truncated count
    result = choose_denominator_basis(
        "exceeded the 300.00s runtime budget", planned_num_examples=300, evaluated_count=28
    )
    assert result == 300, f"Expected 300 (planned), got {result}"


def test_full_run_uses_evaluated():
    # No early stop: denominator is evaluated_count (which equals planned for a full run)
    result = choose_denominator_basis(None, planned_num_examples=300, evaluated_count=300)
    assert result == 300, f"Expected 300, got {result}"


def test_accuracy_early_stop_uses_planned():
    # Accuracy early stop: denominator must be planned count
    result = choose_denominator_basis(
        "max possible accuracy below target accuracy", planned_num_examples=300, evaluated_count=50
    )
    assert result == 300, f"Expected 300 (planned), got {result}"


def test_no_early_stop_partial_uses_evaluated():
    # DISCRIMINATING CASE: no early stop but only 28 of 300 ran (shouldn't normally
    # happen, but proves the helper returns the EVALUATED count, not planned, when
    # there's no early-stop reason. A buggy "always return planned" would fail here.
    result = choose_denominator_basis(None, planned_num_examples=300, evaluated_count=28)
    assert result == 28, f"Expected 28 (evaluated), got {result}"


def test_empty_reason_string_is_not_early_stop():
    # An empty string is falsy -> treated as no early stop -> evaluated count.
    result = choose_denominator_basis("", planned_num_examples=300, evaluated_count=28)
    assert result == 28, f"Expected 28 (evaluated), got {result}"


if __name__ == "__main__":
    test_runtime_early_stop_uses_planned()
    test_full_run_uses_evaluated()
    test_accuracy_early_stop_uses_planned()
    test_no_early_stop_partial_uses_evaluated()
    test_empty_reason_string_is_not_early_stop()
    print("All tests passed.")
