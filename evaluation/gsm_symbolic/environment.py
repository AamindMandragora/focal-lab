"""
Environment setup for GSM-Symbolic evaluation.

Thin wrapper around common; uses start_rule="csd_start" and supports quantization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from evaluation.common.environment import (
    load_compiled_modules,
    resolve_run_dir,
    setup_dafny_environment as _setup_dafny_environment,
    verify_critical_tokens,
)


def _attach_helper_fastpath(VerifiedDecoderAgent) -> None:
    """Replace the compiled helper's Python vocab scan with tensor-backed argmax.

    The compiled Dafny helper walks `lm.Logits[i]` in Python to find the argmax.
    Under our HuggingFace-backed evaluation LM that means a huge number of Python
    scalar reads for large vocabularies. Delegating to `lm.ChooseNextToken()`
    preserves the helper's postcondition while avoiding the scan.
    """
    helpers_cls = getattr(VerifiedDecoderAgent, "CSDHelpers", None)
    if helpers_cls is None or getattr(helpers_cls, "_fastpath_patched", False):
        return

    original = getattr(helpers_cls, "GetHighestLogitToken", None)
    if original is None:
        return

    def _fast_get_highest_logit_token(self, lm):
        return lm.ChooseNextToken()

    helpers_cls.GetHighestLogitToken = _fast_get_highest_logit_token
    helpers_cls._fastpath_patched = True


def setup_dafny_environment(
    run_dir: Path,
    model_name: str,
    device: str,
    vocab_size: int | None,
    grammar_file: Path,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> Dict[str, Any]:
    """Setup Dafny environment with GSM grammar start rule and optional quantization."""
    env = _setup_dafny_environment(
        run_dir=run_dir,
        model_name=model_name,
        device=device,
        vocab_size=vocab_size,
        grammar_file=grammar_file,
        start_rule="csd_start",
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        add_gsm_delimiter_tokens=True,
    )
    _attach_helper_fastpath(env["VerifiedDecoderAgent"])
    return env


__all__ = [
    "resolve_run_dir",
    "load_compiled_modules",
    "verify_critical_tokens",
    "setup_dafny_environment",
]
