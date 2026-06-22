"""Helpers to isolate model-generated text from full prompt+completion strings."""

from __future__ import annotations


def strip_prompt_prefix(prompt, text: str) -> str:
    """Return only the model completion when ``text`` repeats the prompt prefix."""
    if not text or not prompt:
        return text
    if not isinstance(prompt, str):
        return text
    if text.startswith(prompt):
        return text[len(prompt):]
    return text


def completion_for_scoring(prompt, raw_output: str) -> str:
    """Text passed to benchmark ``extract_actual`` / syntax checks (generation only)."""
    if not raw_output:
        return ""
    if prompt:
        return strip_prompt_prefix(prompt, raw_output)
    return raw_output
