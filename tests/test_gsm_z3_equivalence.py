"""Tests for GSM-Symbolic Z3 equivalence checking."""

from __future__ import annotations

import pytest

from synthesis.evaluate.benchmarks.gsm_symbolic.z3_equivalence import (
    gsm_symbolic_z3_equivalence,
)


@pytest.mark.parametrize(
    ("model_expr", "expected_expr", "variable_types", "expected"),
    [
        ("x + y", "y + x", {"x": "int", "y": "int"}, True),
        ("x * 2", "x + x", {"x": "int"}, True),
        ("x + 1", "x + 2", {"x": "int"}, False),
        ("int(x / y)", "x // y", {"x": "int", "y": "int"}, True),
    ],
)
def test_gsm_symbolic_z3_equivalence(model_expr, expected_expr, variable_types, expected):
    z3 = pytest.importorskip("z3")
    del z3  # presence check only
    assert (
        gsm_symbolic_z3_equivalence(model_expr, expected_expr, variable_types) is expected
    )
