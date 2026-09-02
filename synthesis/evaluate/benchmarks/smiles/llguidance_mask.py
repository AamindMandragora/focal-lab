"""llguidance next-token accept masks for SMILES (CARS-aligned).

CARS uses ``LlguidanceTokenRecognizer.filter_vocab`` / llguidance bitmasks.
Syncode ``DFAMaskStore`` only allows BPE tokens that equal one grammar terminal,
so multi-atom tokens like ``CC`` are incorrectly hard-masked at the empty prefix.
"""

from __future__ import annotations

import os
from typing import Any

import torch


class SmilesLlguidanceMaskStore:
    """Resettable llguidance matcher → full-vocab bool accept mask."""

    def __init__(self, grammar_str: str, tokenizer: Any):
        import llguidance
        import llguidance.hf
        import llguidance.torch as llg_torch

        ll_grammar = llguidance.grammar_from("grammar", grammar_str)
        self._ll_tokenizer = llguidance.hf.from_tokenizer(tokenizer)
        err = llguidance.LLMatcher.validate_grammar(ll_grammar, self._ll_tokenizer)
        if err:
            raise ValueError(f"SMILES llguidance grammar error: {err}")
        self._matcher = llguidance.LLMatcher(
            self._ll_tokenizer,
            ll_grammar,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "0")),
        )
        self._bitmask = llg_torch.allocate_token_bitmask(1, self._ll_tokenizer.vocab_size)
        self._llg_torch = llg_torch
        self._tokenizer = tokenizer
        self._vocab_size = int(getattr(tokenizer, "vocab_size", None) or len(tokenizer))

    def reset(self) -> None:
        self._matcher.reset()

    def accept_mask_for_text(self, text: str) -> torch.Tensor:
        """Bool mask over tokenizer vocab: True = allowed next token.

        Prefer ``accept_mask_for_pieces`` when the prefix was built from emitted
        token strings — ``encode(full_text)`` can re-merge BPE pieces and leave
        the matcher unable to consume the last id (empty mask → early stop).
        """
        if not text:
            return self.accept_mask_for_token_ids([])
        return self.accept_mask_for_token_ids(
            self._tokenizer.encode(text, add_special_tokens=False)
        )

    def accept_mask_for_pieces(self, pieces: list[str]) -> torch.Tensor:
        """Advance with each emitted piece's token id(s), then return next-token mask."""
        ids: list[int] = []
        for piece in pieces:
            if not piece:
                continue
            ids.extend(self._tokenizer.encode(piece, add_special_tokens=False))
        return self.accept_mask_for_token_ids(ids)

    def accept_mask_for_token_ids(self, token_ids: list[int]) -> torch.Tensor:
        self.reset()
        if token_ids:
            consumed = self._matcher.try_consume_tokens(list(token_ids))
            if consumed < len(token_ids):
                return torch.zeros(self._vocab_size, dtype=torch.bool)
        self._llg_torch.fill_next_token_bitmask(self._matcher, self._bitmask, 0)
        return _bitmask_to_bool(self._bitmask[0], self._vocab_size)


def _bitmask_to_bool(bitmask_row: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Unpack llguidance int32 bitmask row into a length-``vocab_size`` bool tensor."""
    words = bitmask_row.detach().to(dtype=torch.int32, device="cpu")
    n_words = int(words.numel())
    # Vectorized unpack: each int32 → 32 bits
    shifts = torch.arange(32, dtype=torch.int32)
    bits = ((words.unsqueeze(1) >> shifts) & 1).bool().reshape(-1)
    if bits.numel() < vocab_size:
        out = torch.zeros(vocab_size, dtype=torch.bool)
        out[: bits.numel()] = bits
        return out
    return bits[:vocab_size].contiguous()


def smiles_llguidance_accept_mask(grammar_str: str, tokenizer: Any, text: str) -> torch.Tensor:
    """One-shot helper (builds a fresh matcher). Prefer ``SmilesLlguidanceMaskStore`` in hot paths."""
    return SmilesLlguidanceMaskStore(grammar_str, tokenizer).accept_mask_for_text(text)
