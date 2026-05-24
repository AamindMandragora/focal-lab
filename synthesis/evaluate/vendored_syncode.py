"""Bootstrap imports for the repo-vendored SynCode package.

Shared conda envs may still have an editable ``syncode`` install that points at
another user's checkout (for example ``/home/aadivyar/CRANE/syncode``). Python's
editable meta-path finder runs before normal ``sys.path`` lookup and raises
``PermissionError`` when that tree is unreadable. Call
:func:`ensure_vendored_syncode_importable` before any ``import syncode`` in this
repository.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_BOOTSTRAPPED = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _vendored_syncode_paths() -> tuple[Path, Path]:
    syncode_root = Path(__file__).resolve().parent / "syncode"
    return syncode_root, syncode_root / "syncode"


def _mapping_points_to_unreadable_syncode(mapping: dict[str, str]) -> bool:
    pkg_path = mapping.get("syncode")
    if not pkg_path:
        return False
    init_py = Path(pkg_path) / "__init__.py"
    try:
        return not init_py.is_file() or not os.access(init_py, os.R_OK)
    except OSError:
        return True


def _drop_broken_syncode_editable_finders() -> None:
    kept = []
    for finder in sys.meta_path:
        mapping = getattr(finder, "MAPPING", None)
        if isinstance(mapping, dict) and _mapping_points_to_unreadable_syncode(mapping):
            continue
        kept.append(finder)
    if len(kept) != len(sys.meta_path):
        sys.meta_path[:] = kept


def _prepend_vendored_paths() -> None:
    syncode_root, syncode_pkg = _vendored_syncode_paths()
    for candidate in (syncode_pkg, syncode_root):
        candidate_str = str(candidate)
        if candidate_str in sys.path:
            sys.path.remove(candidate_str)
        sys.path.insert(0, candidate_str)


def _purge_foreign_syncode_modules() -> None:
    _, syncode_pkg = _vendored_syncode_paths()
    vendored_init = syncode_pkg / "__init__.py"
    if not vendored_init.is_file():
        return
    try:
        vendored_resolved = vendored_init.resolve()
    except OSError:
        return

    for name, module in list(sys.modules.items()):
        if name != "syncode" and not name.startswith("syncode."):
            continue
        mod_file = getattr(module, "__file__", None)
        if not mod_file:
            continue
        try:
            if Path(mod_file).resolve() != vendored_resolved:
                del sys.modules[name]
        except OSError:
            del sys.modules[name]


def ensure_vendored_syncode_importable() -> None:
    """Make ``import syncode`` resolve to ``synthesis/evaluate/syncode/syncode``."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _drop_broken_syncode_editable_finders()
    _prepend_vendored_paths()
    _purge_foreign_syncode_modules()
    _BOOTSTRAPPED = True
