"""RS (rejection sampling) baseline helpers.

Each example runs up to ``max_attempts`` full decodes at temperature 1 without
grammar masking (SynCode ``mode="original"``). The first syntax-valid completion
is kept; if none validate, the last attempt is scored.
"""

from __future__ import annotations

from collections.abc import Callable
from synthesis.evaluate.syncode_run_session import SyncodeRunSession

# Matches CARS default search budget in ``run_legacy_fixed_strategy``.
DEFAULT_RS_SEARCH_STEPS = 200

# Fixed sampling temperature for this baseline (same policy as CARS).
RS_TEMPERATURE = 1.0


def rs_sample_completion(
    session: SyncodeRunSession,
    prompt: str,
    *,
    max_attempts: int,
    normalize_output: Callable[[str], str],
    is_syntax_valid: Callable[[str], bool],
) -> tuple[str, int]:
    """Return ``(completion, attempts_used)`` after RS decoding."""
    attempts = max(1, int(max_attempts))
    last_raw = ""
    last_normalized = ""
    for attempt in range(1, attempts + 1):
        batch = session.infer(prompt, stop_words=None)
        last_raw = batch[0] if batch else ""
        last_normalized = normalize_output(last_raw)
        if is_syntax_valid(last_normalized):
            return last_normalized, attempt
    return last_normalized, attempts


def build_rs_session(
    model_name: str,
    *,
    device: str,
    max_new_tokens: int,
) -> SyncodeRunSession:
    """Create a SynCode session configured for temperature-1 RS decoding."""
    return SyncodeRunSession(
        model_name,
        device=device,
        mode="original",
        quantize=False,
        parse_output_only=True,
        log_level=0,
        max_new_tokens=max(1, int(max_new_tokens)),
        do_sample=True,
        temperature=RS_TEMPERATURE,
        num_return_sequences=1,
        opp=False,
    )
