"""SMILES-specific SynCode mask-store reuse helpers."""

from __future__ import annotations

from typing import Any

from synthesis.evaluate.benchmarks.common.mask_store_cache import (
    ensure_mask_store_pickle_visible,
    find_loadable_mask_store_pickle,
)
from synthesis.evaluate.benchmarks.smiles.grammar_helpers import build_smiles_tier1_body_grammar


def prepare_smiles_mask_store(
    example: dict[str, Any],
    tokenizer: Any,
    *,
    mode: str = "grammar_strict",
) -> str:
    """
    Return grammar EBNF to pass to legacy SynCode and link any existing pickle.

    Prefers tier-1 body grammar when a loadable pickle exists; otherwise falls back
    to raw ``.lark`` text (e.g. legacy caches). Pickles built by legacy IterGen may
    reference ``itergen`` modules and fail to load under vendored SynCode (GCD).
    """
    raw = str(example.get("grammar_text", ""))
    tier1 = build_smiles_tier1_body_grammar(raw)
    if find_loadable_mask_store_pickle(tier1, tokenizer, mode=mode):
        grammar = tier1
    elif find_loadable_mask_store_pickle(raw, tokenizer, mode=mode):
        grammar = raw
    else:
        grammar = tier1
    ensure_mask_store_pickle_visible(grammar, tokenizer, mode=mode)
    return grammar
