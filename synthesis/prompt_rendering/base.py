"""Shared core for rendering prompts from typed data models.

Every prompt in the pipeline should define a small `PromptModel` subclass
holding the exact data the prompt needs, then render it through a Jinja2
template with `render()`. Keeping this in one place pins two things that
must stay consistent across every prompt:

1. Jinja whitespace behavior (`trim_blocks`, `lstrip_blocks`,
   `keep_trailing_newline`) — stock Jinja defaults would silently change
   what the model sees (blank lines left around `{% if %}` tags, the final
   newline stripped from the rendered string).
2. The pydantic model contract: frozen (a prompt's data is set once and
   never mutated) and extra="forbid" (a typo'd field name is a loud error
   at construction time, not a silently missing line in the prompt).
"""
import pathlib

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, ConfigDict

_DEFAULT_TEMPLATES_DIR = str(pathlib.Path(__file__).parent / "templates")


class PromptModel(BaseModel):
    """Base class for a prompt's typed data. Frozen and extra fields forbidden."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def get_environment(searchpath: str | None = None) -> Environment:
    """Build the Jinja2 Environment every prompt template is rendered with.

    The five flags below are load-bearing and must not be changed: stock
    Jinja would strip the final newline and leave blank lines around
    `{% if %}` blocks, silently altering the rendered prompt.
    """
    return Environment(
        loader=FileSystemLoader(searchpath or _DEFAULT_TEMPLATES_DIR),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )


def render(model: PromptModel, template_name: str, env: Environment | None = None) -> str:
    """Render `template_name` from `env` (or the default environment) using
    `model`'s fields as the template context."""
    if env is None:
        env = get_environment()
    template = env.get_template(template_name)
    return template.render(**model.model_dump())
