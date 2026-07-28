"""The author's prompt must state which decoding surface a run uses.

Background: there are two different ways generation can reach the
constrained region.

  - Visible-delimiter surface (start_inside_constrained=False): generation
    begins OUTSIDE the constrained region. The strategy enters constrained
    mode by calling `OpenConstrainedSpan`, which emits a literal `<<` token
    into the output.

  - Observed surface (start_inside_constrained=True): generation begins
    ALREADY INSIDE the constrained region. The strategy enters constrained
    mode by calling `EnterObservedConstrainedSpan`, which emits nothing --
    no `<<` token is ever produced by the model or anyone else.

Spider evaluation runs on the observed surface (see
synthesis/evaluate/benchmarks/sql_spider/eval_logic.py, which sets
start_inside_constrained=True), but the author's prompt never used to say
so. A strategy whose only way of entering constrained mode was "wait for
next == '<<'" would then never constrain anything on Spider, because that
token never arrives.

This test renders the real author prompt with both settings and checks the
prompt states the applicable surface and, for the observed surface, warns
that no `<<` will ever appear.
"""

from __future__ import annotations

from synthesis.generate.prompts import build_initial_prompt


def _render_user_prompt(start_inside_constrained: bool) -> str:
    _system_prompt, user_prompt = build_initial_prompt(
        task_description="Write a syntactically valid SQL query for the question.",
        start_inside_constrained=start_inside_constrained,
    )
    return user_prompt


def test_observed_surface_prompt_names_enter_observed_and_warns_no_delimiter():
    user = _render_user_prompt(start_inside_constrained=True)

    assert "EnterObservedConstrainedSpan" in user
    assert "starts inside the constrained region" in user
    assert "no `<<` token will ever appear" in user


def test_visible_delimiter_surface_prompt_describes_open_constrained_span_route():
    user = _render_user_prompt(start_inside_constrained=False)

    assert "OpenConstrainedSpan" in user
    assert "starts outside the constrained region" in user
    # This is the observed-surface-only warning; it must not leak into the
    # visible-delimiter surface's prompt.
    assert "no `<<` token will ever appear" not in user


def test_the_two_surface_renderings_differ():
    inside = _render_user_prompt(start_inside_constrained=True)
    outside = _render_user_prompt(start_inside_constrained=False)

    assert inside != outside
