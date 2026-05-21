"""Reuse SynCode DFA mask stores across tokenizer cache subdirectories."""

from __future__ import annotations

import hashlib
import os
import pickle
import shutil
from pathlib import Path
from typing import Any


def grammar_ebnf_hash(grammar_ebnf: str) -> str:
    """Match ``syncode.parsers.grammars.grammar.Grammar.hash()``."""
    return str(int(hashlib.sha256(grammar_ebnf.encode("utf-8")).hexdigest(), 16))[:10]


def mask_store_pickle_name(
    grammar_ebnf: str,
    *,
    mode: str,
    vocab_size: int,
) -> str:
    return f"{mode}_{grammar_ebnf_hash(grammar_ebnf)}_{vocab_size}.pkl"


def synocode_mask_stores_root() -> Path:
    import syncode.common as common

    return Path(common.SYNCODE_CACHE) / "mask_stores"


def find_mask_store_pickle(
    grammar_ebnf: str,
    tokenizer: Any,
    *,
    mode: str = "grammar_strict",
) -> Path | None:
    """
    Locate an on-disk mask store for ``grammar_ebnf``, searching every
    ``mask_stores/<TokenizerClass>/`` directory (e.g. ``CachedQwen2Tokenizer``).

    Matches on grammar hash and mode; vocab size in the filename may differ slightly
    across Transformers builds (e.g. ``151643`` vs ``151665`` for Qwen2.5-Coder).
    """
    grammar_hash = grammar_ebnf_hash(grammar_ebnf)
    root = synocode_mask_stores_root()
    if not root.is_dir():
        return None

    try:
        vocab_size = int(len(tokenizer))
    except Exception:
        vocab_size = int(getattr(tokenizer, "vocab_size", -1) or -1)

    hits: list[Path] = []
    exact_name = (
        mask_store_pickle_name(grammar_ebnf, mode=mode, vocab_size=vocab_size)
        if vocab_size > 0
        else None
    )
    glob_pattern = f"{mode}_{grammar_hash}_*.pkl"

    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue
        if exact_name is not None:
            candidate = subdir / exact_name
            if candidate.is_file():
                hits.append(candidate)
        for candidate in subdir.glob(glob_pattern):
            if candidate.is_file():
                hits.append(candidate)

    if not hits:
        return None

    # Deduplicate; prefer exact vocab, then canonical tokenizer dir, then largest file.
    unique = sorted(set(hits), key=lambda p: p.as_posix())
    tokenizer_name = type(tokenizer).__name__

    def _sort_key(p: Path) -> tuple:
        suffix = p.stem.rsplit("_", 1)[-1]
        vocab_match = 0 if vocab_size > 0 and suffix == str(vocab_size) else 1
        return (
            vocab_match,
            0 if p.parent.name == tokenizer_name else 1,
            -p.stat().st_size,
        )

    return sorted(unique, key=_sort_key)[0]


def mask_store_pickle_loadable(path: Path) -> bool:
    """True if ``pickle.load`` succeeds (same check SynCode uses before rebuilding)."""
    try:
        with path.open("rb") as handle:
            pickle.load(handle)
        return True
    except Exception:
        return False


def find_loadable_mask_store_pickle(
    grammar_ebnf: str,
    tokenizer: Any,
    *,
    mode: str = "grammar_strict",
) -> Path | None:
    """Like ``find_mask_store_pickle`` but ignore pickles that fail to unpickle."""
    found = find_mask_store_pickle(grammar_ebnf, tokenizer, mode=mode)
    if found is None:
        return None
    if mask_store_pickle_loadable(found):
        return found
    return None


def ensure_mask_store_pickle_visible(
    grammar_ebnf: str,
    tokenizer: Any,
    *,
    mode: str = "grammar_strict",
) -> Path | None:
    """
    Ensure SynCode's default cache path can load this grammar.

    If the pickle exists only under another tokenizer subdirectory, symlink or
    copy it into ``mask_stores/<type(tokenizer)>/<name>.pkl``.
    """
    found = find_loadable_mask_store_pickle(grammar_ebnf, tokenizer, mode=mode)
    if found is None:
        return None

    root = synocode_mask_stores_root()
    target_dir = root / type(tokenizer).__name__
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        vocab_size = int(len(tokenizer))
    except Exception:
        vocab_size = int(getattr(tokenizer, "vocab_size", -1) or -1)
    target_name = (
        mask_store_pickle_name(grammar_ebnf, mode=mode, vocab_size=vocab_size)
        if vocab_size > 0
        else found.name
    )
    target = target_dir / target_name
    if target.exists():
        return target
    try:
        os.symlink(found.resolve(), target)
    except OSError:
        shutil.copy2(found, target)
    return target

