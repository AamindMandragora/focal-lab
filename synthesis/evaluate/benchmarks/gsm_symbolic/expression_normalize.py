"""Normalize GSM-Symbolic expressions for equivalence checking.

Decoder grammars emit ``int(...)`` and ``//``. CRANE's Z3 checker rewrites those to
``ToInt(...)`` and ``z3_floor_div(left, right)`` before comparison. Apply the same
rewrite in the synthesis evaluator so model and gold expressions compare under
one dialect.
"""

from __future__ import annotations

import ast
import re
from typing import Iterable

# Names reserved by the equivalence dialect (not problem variables).
EQUIVALENCE_RESERVED_NAMES: frozenset[str] = frozenset(
    {"int", "ToInt", "z3_floor_div", "round"}
)

# Grammar-allowed call form vs checker dialect (see gsm.lark and legacy CRANE parser).
GRAMMAR_TO_CHECKER_REWRITES: tuple[tuple[str, str], ...] = (
    ("int", "ToInt"),
)


class _GsmEquivalenceNormalizer(ast.NodeTransformer):
    """Rewrite ``int(…)`` and ``//`` into CRANE's Z3-eval surface form."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "int"
            and len(node.args) == 1
            and not node.keywords
        ):
            return ast.Call(
                func=ast.Name(id="ToInt", ctx=ast.Load()),
                args=node.args,
                keywords=[],
            )
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "round"
            and len(node.args) == 1
            and not node.keywords
        ):
            return ast.Call(
                func=ast.Name(id="ToInt", ctx=ast.Load()),
                args=node.args,
                keywords=[],
            )
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.FloorDiv):
            return ast.Call(
                func=ast.Name(id="z3_floor_div", ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[],
            )
        return node


def _floor_div_replacer(expression: str) -> str:
    """Regex fallback matching legacy CRANE ``gsm_symbolic.py``."""
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


def _normalize_gsm_symbolic_regex(expr: str) -> str:
    out = expr.strip()
    out = re.sub(r"\bint\s*\(", "ToInt(", out)
    out = re.sub(r"\bround\s*\(", "ToInt(", out)
    if "//" in out:
        out = _floor_div_replacer(out)
    return out


def normalize_gsm_symbolic_for_equivalence(expr: str | None) -> str:
    """Map grammar-style GSM expressions into CRANE/Z3 equivalence dialect."""
    if expr is None:
        return ""
    text = str(expr).strip()
    if not text:
        return text
    try:
        tree = ast.parse(text, mode="eval")
        normalized = _GsmEquivalenceNormalizer().visit(tree)
        return ast.unparse(normalized).strip()
    except SyntaxError:
        return _normalize_gsm_symbolic_regex(text)


def reserved_equivalence_names(extra: Iterable[str] = ()) -> set[str]:
    names = set(EQUIVALENCE_RESERVED_NAMES)
    names.update(extra)
    return names


def has_unbound_problem_variables(expr: str, extra_reserved: Iterable[str] = ()) -> bool:
    """True when ``expr`` still contains identifiers other than checker builtins."""
    ids = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr))
    return bool(ids - reserved_equivalence_names(extra_reserved))
