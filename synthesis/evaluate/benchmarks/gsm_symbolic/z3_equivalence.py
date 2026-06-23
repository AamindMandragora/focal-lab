"""GSM-Symbolic equivalence via Z3 only (no substitution fallback)."""

from __future__ import annotations

import re
from typing import Any

from synthesis.evaluate.benchmarks.gsm_symbolic.expression_normalize import (
    normalize_gsm_symbolic_for_equivalence,
    reserved_equivalence_names,
)

_Z3_TIMEOUT_MS = 5000


def _floor_div_replacer(expression: str) -> str:
    pattern = r"(?P<left>.+?)\s*//\s*(?P<right>.+)"
    prev = None
    current = expression
    while prev != current and "//" in current:
        prev = current
        current = re.sub(
            pattern,
            lambda m: f"z3_floor_div({m.group('left').strip()}, {m.group('right').strip()})",
            current,
        )
    return current


def _integer_check(x: Any) -> Any:
    from z3 import And

    return And(x == _floor(x), x == _ceiling(x))


def _floor(x: Any) -> Any:
    from z3 import If, ToInt, ToReal

    return If(
        x >= 0,
        ToInt(x),
        ToInt(x) - If(ToReal(ToInt(x)) == x, 0, 1),
    )


def _ceiling(x: Any) -> Any:
    from z3 import If, ToInt, ToReal

    return If(
        x >= 0,
        ToInt(x) + If(ToReal(ToInt(x)) == x, 0, 1),
        ToInt(x),
    )


def gsm_symbolic_z3_equivalence(
    model_expr: str | None,
    expected_expr: str,
    variable_types: dict[str, str],
) -> bool:
    """Return whether two GSM symbolic expressions are Z3-proven equivalent."""
    if model_expr is None:
        return False

    original_model = normalize_gsm_symbolic_for_equivalence(model_expr)
    original_expected = normalize_gsm_symbolic_for_equivalence(expected_expr)
    if not original_model.strip() or not original_expected.strip():
        return False

    var_names = set(
        re.findall(r"\b[a-zA-Z_]\w*\b", original_model + " " + original_expected)
    )
    var_names -= reserved_equivalence_names()

    for name in var_names:
        if name not in variable_types:
            return False

    try:
        from z3 import If, Real, Solver, ToInt, unsat
    except ImportError:
        return False

    vars_dict: dict[str, Any] = {}
    constraints: list[Any] = []
    for name in var_names:
        var = Real(name)
        vars_dict[name] = var
        var_type = variable_types.get(name, "str")
        if var_type == "float between 0 and 1":
            constraints.extend((var > 0, var <= 1))
        elif var_type == "float":
            constraints.append(var > 0)
        elif var_type == "int":
            constraints.extend((var > 0, _integer_check(var)))
        else:
            return False

    expr1 = re.sub(r"\bint\(", "ToInt(", original_model)
    expr2 = re.sub(r"\bint\(", "ToInt(", original_expected)

    if "round(" in expr1:
        return False
    expr2 = re.sub(r"\bround\(", "ToInt(", expr2)

    if "//" in expr1:
        expr1 = _floor_div_replacer(expr1)
    if "//" in expr2:
        expr2 = _floor_div_replacer(expr2)

    def z3_floor_div(x: Any, y: Any) -> Any:
        return If(y != 0, ToInt(x / y), 0)

    eval_env = {
        **vars_dict,
        "ToInt": ToInt,
        "z3_floor_div": z3_floor_div,
        "Floor": _floor,
        "Ceiling": _ceiling,
    }

    def safe_eval(expr: str) -> Any:
        return eval(expr, {"__builtins__": None}, eval_env)

    try:
        expr2_z3 = safe_eval(expr2)
        expr1_z3 = safe_eval(expr1)
    except Exception:
        return False

    solver = Solver()
    solver.set("timeout", _Z3_TIMEOUT_MS)
    solver.add(constraints)
    try:
        solver.add(expr1_z3 != expr2_z3)
    except Exception:
        return False

    result = solver.check()
    return result == unsat
