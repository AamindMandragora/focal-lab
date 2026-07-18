"""Core contract for the shared prompt-rendering layer.

These tests are the spec for synthesis/prompt_rendering/base.py. They pin:
  1. the exact Jinja whitespace behavior (trim_blocks / lstrip_blocks /
     keep_trailing_newline) — stock Jinja would silently change what the model
     sees (blank lines around {% if %}, stripped final newline);
  2. the pydantic model contract: frozen (write-once) + extra="forbid"
     (a mistyped field is a loud error, not a silently missing prompt line).
"""
import pathlib

import pytest
from pydantic import ValidationError

from synthesis.prompt_rendering.base import (
    PromptModel,
    get_environment,
    render,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


class _WhitespaceModel(PromptModel):
    show: bool
    name: str


def test_whitespace_pinning_block_and_trailing_newline():
    """trim_blocks + lstrip_blocks remove the tag lines cleanly; the optional
    section renders with no stray blank lines, and the final newline is kept."""
    env = get_environment(searchpath=str(FIXTURES))
    out = render(_WhitespaceModel(show=True, name="A"), "ws.j2", env=env)
    assert out == "Header\n  line A\nFooter\n"


def test_whitespace_pinning_optional_section_absent():
    env = get_environment(searchpath=str(FIXTURES))
    out = render(_WhitespaceModel(show=False, name="A"), "ws.j2", env=env)
    assert out == "Header\nFooter\n"


class _StrictModel(PromptModel):
    x: int


def test_extra_field_forbidden():
    with pytest.raises(ValidationError):
        _StrictModel(x=1, y=2)


def test_model_is_frozen():
    m = _StrictModel(x=1)
    with pytest.raises(ValidationError):
        m.x = 5
