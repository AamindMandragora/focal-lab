"""The attempt time cap must be settable without editing source.

Why this matters
----------------
SynthesisPipeline takes `max_attempt_seconds` (default 3600s) to stop one
pathological attempt from hanging a whole run. But run_synthesis.py -- the only
real entry point -- never passed it, so the default was the ONLY value any run
could ever have. Changing it meant editing library source.

3600s is a guess, not a measured number, and the right value moves with the
benchmark, the model and the machine. It got more load-bearing after the eval
worker pool turned out to be missing on this branch: evaluation now runs
sequentially and reloads the vLLM engine (~24s) each iteration, so attempts are
slower than when 3600 was chosen.

A guessed default that cannot be overridden is the problem. This pins that it
can be.

Note on shape: the parser is built inside main(), so it cannot be imported and
inspected directly. These tests read the source structure instead, which still
fails if the flag or its wiring is removed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ENTRY_POINT = Path(__file__).resolve().parents[1] / "synthesis" / "run_synthesis.py"


def _source_tree() -> ast.AST:
    return ast.parse(ENTRY_POINT.read_text())


def _added_cli_flags(tree: ast.AST) -> set[str]:
    """Every string literal passed to an add_argument(...) call."""
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    flags.add(arg.value)
    return flags


def _pipeline_keywords(tree: ast.AST) -> dict[str, ast.AST]:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SynthesisPipeline"
        ):
            return {kw.arg: kw.value for kw in node.keywords if kw.arg}
    pytest.fail("No SynthesisPipeline(...) call found in run_synthesis.py")


def test_the_entry_point_exists_where_this_test_thinks_it_does():
    assert ENTRY_POINT.is_file(), f"Expected the entry point at {ENTRY_POINT}"


def test_the_cap_has_a_command_line_flag():
    flags = _added_cli_flags(_source_tree())

    assert "--max-attempt-seconds" in flags, (
        "run_synthesis.py has no --max-attempt-seconds flag, so every run is "
        "stuck with the hard-coded 3600s default and the only way to change it "
        "is editing library source. Sorted flags found: "
        f"{sorted(f for f in flags if f.startswith('--'))[:12]}..."
    )


def test_the_flag_is_actually_passed_to_the_pipeline():
    """A flag that is parsed but never forwarded is worse than no flag.

    It would accept --max-attempt-seconds on the command line, print nothing,
    and silently keep using 3600.
    """
    keywords = _pipeline_keywords(_source_tree())

    assert "max_attempt_seconds" in keywords, (
        "SynthesisPipeline is constructed without max_attempt_seconds, so the "
        "CLI flag (if any) is ignored and the default silently wins."
    )

    value = keywords["max_attempt_seconds"]
    assert isinstance(value, ast.Attribute) and value.attr == "max_attempt_seconds", (
        "max_attempt_seconds should be forwarded from the parsed arguments "
        "(args.max_attempt_seconds), not hard-coded at the call site."
    )
