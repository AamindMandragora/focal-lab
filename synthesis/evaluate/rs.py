"""RS (rejection sampling) baseline helpers.

Each example runs up to ``max_attempts`` full decodes at temperature 1 without
grammar masking (SynCode ``mode="original"``). The first syntax-valid completion
is kept; if none validate, the last attempt is scored.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

DEFAULT_RS_SEARCH_STEPS = 200
RS_TEMPERATURE = 1.0


def rs_sample_completion(
    syncode: Any,
    prompt: str,
    *,
    max_attempts: int,
    normalize_output: Callable[[str], str],
    is_syntax_valid: Callable[[str], bool],
    stop_words: list[str] | None = None,
) -> tuple[str, int]:
    """Return ``(completion, attempts_used)`` after RS decoding."""
    attempts = max(1, int(max_attempts))
    last_normalized = ""
    for attempt in range(1, attempts + 1):
        batch = syncode.infer(prompt, stop_words=stop_words)
        last_raw = batch[0] if batch else ""
        last_normalized = normalize_output(last_raw)
        if is_syntax_valid(last_normalized):
            return last_normalized, attempt
    return last_normalized, attempts


def build_rs_syncode(
    model_name: str,
    *,
    device: str,
    max_new_tokens: int,
) -> Any:
    """Create a SynCode instance configured for temperature-1 RS decoding."""
    from syncode.infer import Syncode

    return Syncode(
        model=model_name,
        mode="original",
        quantize=False,
        device=device,
        parse_output_only=True,
        log_level=0,
        max_new_tokens=max(1, int(max_new_tokens)),
        do_sample=True,
        temperature=RS_TEMPERATURE,
        num_return_sequences=1,
        opp=False,
    )
