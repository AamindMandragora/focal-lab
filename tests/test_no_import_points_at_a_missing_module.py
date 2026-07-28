"""Shipped code must not import a module that does not exist.

Why this test exists
--------------------
Spider reported 0% accuracy and 0% syntax for weeks. The cause was a missing
file, not a bad model:

    POOLABLE_DATASETS routes spider through a worker pool
        -> `import synthesis.scripts.eval_worker_pool`  (file absent)
        -> ModuleNotFoundError is an Exception
        -> a broad `except Exception` catches it
        -> returns accuracy=0.0, syntax_rate=0.0, num_examples=0

A missing file was displayed as "the model got everything wrong". That first
instance was fixed -- and the fix routed evaluation to a fallback path that
imported ANOTHER missing module (vllm_startup), which produced the identical
fake zero. Finding these one crash at a time is the slow way.

This test finds them all at once, statically, before anything runs.

Known limitation, stated plainly: this reads `import` statements from the
source. A module loaded dynamically by name -- importlib.import_module("...") --
is invisible here. So this is a floor on the problem, not a ceiling.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "synthesis"

# Vendored third-party tree; not our import hygiene to enforce.
SKIP_PARTS = {"syncode", "__pycache__", ".venv"}


def _module_exists(dotted: str) -> bool:
    """True if `dotted` resolves to a real .py file or package in this repo."""
    parts = dotted.split(".")
    for base in (REPO_ROOT, PACKAGE_ROOT):
        candidate = base.joinpath(*parts)
        if candidate.with_suffix(".py").is_file():
            return True
        if (candidate / "__init__.py").is_file():
            return True
    return False


def _resolve_relative(path: Path, node: ast.ImportFrom) -> str | None:
    """Turn `from ..foo import bar` into the absolute module it points at.

    Relative imports matter as much as absolute ones: the dead
    synthesis/feedback_loop.py reaches five non-existent modules purely through
    `from .compiler import ...` style lines.
    """
    # Dropping the final part gives the containing package in both cases:
    # for foo/bar.py that is foo, and for foo/__init__.py it is also foo,
    # since a relative import there resolves against the package itself.
    parts = path.relative_to(REPO_ROOT).with_suffix("").parts
    package = parts[:-1]
    # level 1 is the containing package, level 2 its parent, and so on.
    trimmed = package[: len(package) - (node.level - 1)]
    if not trimmed:
        return None
    tail = node.module.split(".") if node.module else []
    return ".".join([*trimmed, *tail])


def _in_repo_imports() -> list[tuple[str, str]]:
    """Every in-repo module imported by shipped code, with its location.

    Covers both `import synthesis.x` / `from synthesis.x import y` and the
    relative `from .x import y` form.
    """
    found: list[tuple[str, str]] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            where = f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    target = _resolve_relative(path, node)
                    if target:
                        found.append((target, where))
                elif node.module and node.module.split(".")[0] == "synthesis":
                    found.append((node.module, where))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "synthesis":
                        found.append((alias.name, where))
    return found


def test_the_scan_actually_finds_imports():
    """Guard the guard: a scan that silently finds nothing always passes."""
    imports = _in_repo_imports()

    assert len(imports) > 20, (
        f"Only found {len(imports)} in-repo imports under {PACKAGE_ROOT}. "
        "The scan is probably broken or pointed at the wrong directory, which "
        "would make the real test below pass for the wrong reason."
    )


def test_every_imported_module_exists():
    missing: dict[str, list[str]] = {}
    for module, where in _in_repo_imports():
        if not _module_exists(module):
            missing.setdefault(module, []).append(where)

    assert not missing, (
        "Shipped code imports modules that do not exist. Each one raises "
        "ModuleNotFoundError at run time, and this repo has broad `except "
        "Exception` handlers that turn that into a normal-looking score "
        "(accuracy=0.0) instead of a crash:\n\n"
        + "\n".join(
            f"  {module}\n" + "".join(f"      imported at {w}\n" for w in sites)
            for module, sites in sorted(missing.items())
        )
    )
