"""SMILES must genuinely run without visible << >> delimiters.

What was wrong
--------------
SMILES declared itself delimiter-free -- `emits_visible_delimiters()` returns
False, and its comment says "SMILES has exactly one constrained span and no
<< >> markers around it". Its grammar agrees: `smiles_*.lark` starts at
`start: smiles`, with no delimiter anywhere in it.

But all three of its prompt builders told the model to produce delimiters:

    format_prompt                   "Wrap your answer molecule in << >> delimiters"
    format_prompt_expression_only   "Return exactly one line containing `<<SMILES>>`"
    format_prompt_chain_of_thought  "wrap your final SMILES in << >> delimiters"

and its generation runner never asked to start inside the constrained region,
so generation began outside it on the visible-delimiter surface.

So the benchmark said one thing and did another. That matters twice over:
`emits_visible_delimiters()` suppresses delimiter diagnostics, so real
delimiter failures were being hidden; and the strategy-writing AI is told which
surface it is on, so a wrong answer there makes it write a strategy that cannot
work.

The decision (2026-07-28) was to make the behaviour match the declaration:
SMILES is delimiter-free for real.

Note on extraction: `clean_smiles_output` in metrics.py strips `<<`/`>>` rather
than requiring them, so removing the delimiters does not break answer parsing.
"""

from __future__ import annotations

import importlib

import pytest


PROMPT_BUILDERS = [
    "format_prompt",
    "format_prompt_expression_only",
    "format_prompt_chain_of_thought",
]

EXAMPLE = {"prompt": "Give a molecule from the acrylates class."}


def _smiles_eval_logic():
    return importlib.import_module("synthesis.evaluate.benchmarks.smiles.eval_logic")


@pytest.mark.parametrize("builder_name", PROMPT_BUILDERS)
def test_no_prompt_asks_the_model_for_delimiters(builder_name):
    builder = getattr(_smiles_eval_logic(), builder_name)
    prompt = builder(None, EXAMPLE)

    assert "<<" not in prompt and ">>" not in prompt, (
        f"{builder_name} still tells the model to emit << >> delimiters, but "
        "SMILES reports emits_visible_delimiters() == False and its grammar has "
        "no delimiter in it. The prompt and the benchmark must agree."
    )
    assert "delimiter" not in prompt.lower(), (
        f"{builder_name} still mentions delimiters to the model."
    )


def test_smiles_reports_that_it_starts_inside_the_constrained_region():
    assert _smiles_eval_logic().starts_inside_constrained() is True, (
        "SMILES has no visible delimiters, so there is no `<<` for a strategy to "
        "wait for. Generation must therefore begin already inside the "
        "constrained region, and the author's prompt must be told so."
    )


def test_smiles_generation_actually_starts_inside_the_constrained_region(monkeypatch):
    """The claim above must match what evaluation really does.

    Declaring the surface is only useful if the run honours it. The runner
    imports `run_crane_csd` when called, not at module load, so replacing it on
    the generation module first intercepts the call without loading a model.
    """
    eval_logic = _smiles_eval_logic()
    generation = importlib.import_module(
        "synthesis.evaluate.benchmarks.smiles.generation"
    )

    seen: dict = {}

    def _capture(*args, **kwargs):
        seen.update(kwargs)
        return ("", 0, 0.0, [], [])

    monkeypatch.setattr(generation, "run_crane_csd", _capture)

    runner = eval_logic.get_generation_runner()
    runner()

    assert seen.get("start_inside_constrained") is True, (
        "SMILES evaluation still generates on the visible-delimiter surface. "
        f"start_inside_constrained was {seen.get('start_inside_constrained')!r}. "
        "The strategy will wait for a `<<` that no one emits and never constrain "
        "anything."
    )
