"""Single-GPU Hugging Face model session for vendored SynCode constrained decoding.

Legacy GCD baselines used one :class:`syncode.infer.Syncode` per distinct grammar text.
GSM tier-1 grammars vary per example, which loaded the full model dozens of times and
triggered CUDA OOM. This module keeps one model and swaps only grammar decoders (DFA
mask stores), resetting parser state on every ``infer`` call.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Union

from synthesis.evaluate.vendored_syncode import ensure_vendored_syncode_importable

ensure_vendored_syncode_importable()

import syncode.common as syncode_common
from syncode.grammar_decoder import SyncodeLogitsProcessor
from syncode.language_model import HuggingFaceModel
from syncode.parsers.grammars import Grammar
from transformers import LogitsProcessorList


class SyncodeRunSession:
    """Reuse one HF causal LM; cache :class:`SyncodeLogitsProcessor` per grammar text."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        mode: str = "grammar_strict",
        quantize: bool = False,
        parse_output_only: bool = True,
        parser: str = "lalr",
        log_level: int = 0,
        dev_mode: bool = False,
        new_mask_store: bool = False,
        num_return_sequences: int = 1,
        opp: bool = False,
        **gen_kwargs: Any,
    ) -> None:
        if mode not in ("original", "grammar_mask", "grammar_strict"):
            raise ValueError(f"Unsupported SynCode mode: {mode}")

        self.model_name = model_name
        self.device = device
        self.mode = mode
        self.quantize = quantize
        self.parse_output_only = parse_output_only
        self.parser = parser
        self.log_level = log_level
        self.dev_mode = dev_mode
        self.new_mask_store = new_mask_store
        self.num_samples = num_return_sequences
        self.opp = opp
        self.gen_kwargs = dict(gen_kwargs)

        self._model = syncode_common.load_model(model_name, device, quantize)
        self._tokenizer = syncode_common.load_tokenizer(model_name)
        self._grammar_decoders: dict[str, SyncodeLogitsProcessor] = {}
        self._active_grammar_text: Optional[str] = None
        self._hf_runner: Optional[HuggingFaceModel] = None

    @property
    def loaded_model(self) -> Any:
        return self._model

    def apply_grammar(self, grammar_text: str) -> None:
        """Attach *grammar_text* for the next decode; does not retain prior parser state."""
        if self._active_grammar_text == grammar_text and self._hf_runner is not None:
            return

        grammar = Grammar(grammar_text)
        decoder = self._grammar_decoders.get(grammar_text)
        if decoder is None:
            decoder = SyncodeLogitsProcessor(
                grammar,
                tokenizer=self._tokenizer,
                use_cache=not self.new_mask_store,
                parse_output_only=self.parse_output_only,
                num_samples=self.num_samples,
                dev_mode=self.dev_mode,
                parser=self.parser,
                mode=self.mode,
            )
            self._grammar_decoders[grammar_text] = decoder

        if self._hf_runner is None:
            self._hf_runner = HuggingFaceModel(
                self._model,
                grammar=grammar,
                tokenizer=self._tokenizer,
                device=self.device,
                grammar_decoder=decoder,
                mode=self.mode,
                opp=self.opp,
                **self.gen_kwargs,
            )
        else:
            self._hf_runner.grammar = grammar
            self._hf_runner.grammar_decoder = decoder
            self._hf_runner.grammar_processor = (
                LogitsProcessorList([decoder]) if decoder is not None else None
            )

        self._active_grammar_text = grammar_text

    def infer(
        self,
        prompt: Union[str, list],
        *,
        stop_words: Optional[list[str]] = None,
        debug: bool = False,
    ) -> list[str]:
        if self._hf_runner is None:
            raise RuntimeError("call apply_grammar() before infer()")
        batch = self._hf_runner.generate_grammar_constrained_completion(
            prompt,
            self.num_samples,
            stop_words=stop_words,
            debug=debug,
        )
        return list(batch)

    def close(self) -> None:
        """Drop references so a long matrix subprocess can reclaim GPU memory."""
        self._grammar_decoders.clear()
        self._hf_runner = None
        self._active_grammar_text = None
        self._model = None
        self._tokenizer = None
        release_cuda_cache()


def release_cuda_cache() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
