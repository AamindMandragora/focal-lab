"""TDD guard for the IterGen-type-tool fairness fix (2026-06-17).

The synthesis study is controlled: prompts may carry MECHANISM-level tool info
(signature / role / mechanics / requires / returns / cost / control profile) but
NOT strategy guidance — no "when/how to use a specific tool", no worked example
promoting a tool, no feedback nudge naming a tool to adopt. (Project CLAUDE.md +
user ruling feedback_csd_guidance_allowed.md: general CSD-mechanism hints are
fair; task/tool-specific guidance is banned.)

Three IterGen-type-tool promotions were found and are removed here:
  1. prompts.py menu bullets for RegenerateUnitOnCheckFailure /
     RegenerateUnitOnGroundingFailure carried "When to use:", "How to use:",
     "Example call shape:", "Suggested starting values:" — stripped to
     mechanism-only (signatures + Mechanics/Cost/Control profile KEPT).
  2. prompts.py "Grounded-and-closed constrained CSD" worked example that
     promotes RegenerateUnitOnGroundingFailure — removed (it was already DEAD:
     its header didn't match _VERIFIED_EXAMPLE_PREFIXES so it was never injected;
     this also removes the dangling "// Grounded-unit constrained CSD." prefix).
  3. feedback_loop.py _unit_rewind_hint — a feedback nudge that named
     RegenerateUnitOnCheckFailure and told the author to adopt it — removed.

The two helpers REMAIN selectable (still scraped into the helper universe; the
bandit/mask still decides their visibility). We only remove the special promotion.

Run:  python -m pytest tests/fairness_cloud/test_itergen_promotion_removed.py -q
"""

import importlib.util
import pathlib

_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[2]  # .../_focal_pull (or csd-generation on focal)
_PROMPTS_PATH = _REPO / "generate" / "prompts.py"
_FEEDBACK_PATH = _REPO / "evaluate" / "feedback_loop.py"

# Fallback for the focal layout where the package root is synthesis/.
if not _PROMPTS_PATH.exists():
    _PROMPTS_PATH = _REPO / "synthesis" / "generate" / "prompts.py"
    _FEEDBACK_PATH = _REPO / "synthesis" / "evaluate" / "feedback_loop.py"


def _load_prompts():
    spec = importlib.util.spec_from_file_location("_prompts_under_test", _PROMPTS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BANNED_GUIDANCE_PHRASES = (
    "When to use:",
    "How to use:",
    "Example call shape:",
    "Suggested starting values:",
)

LIVE_EXAMPLE_TITLES = (
    "Guided-adaptive CSD.",
    "Simple delimiter-triggered CSD.",
    "Group-aware constrained CSD.",
    "Adaptive-narrowness CSD.",
    "Open-then-reliably-close CSD.",
)


def test_no_when_how_use_guidance_in_tool_reference():
    p = _load_prompts()
    ref = p._build_tool_reference_block(None)
    for phrase in BANNED_GUIDANCE_PHRASES:
        assert phrase not in ref, f"banned guidance phrase still present: {phrase!r}"


def test_grounding_helper_mechanism_kept():
    """Stripping guidance must NOT remove the mechanism-level info or the helper.

    NOTE (2026-06-18 menu prune): #14 RegenerateUnitOnCheckFailure was later
    pruned from the menu entirely (zero-usage clutter — see
    test_helper_menu_prune.py), so only #15 RegenerateUnitOnGroundingFailure
    must keep its mechanism-level signature here. This is the deliberate prune,
    not test-weakening.
    """
    p = _load_prompts()
    ref = p._build_tool_reference_block(None)
    for sig in ("RegenerateUnitOnGroundingFailure(",):
        assert sig in ref, f"helper signature lost from menu: {sig!r}"
    # Mechanism descriptors must survive for the kept helper.
    assert "Control profile:" in ref
    assert "Mechanics:" in ref


def test_helpers_still_in_universe():
    """The grounding helper (and ManagedStep) stay selectable via the bandit/mask.

    NOTE (2026-06-18 menu prune): #14 RegenerateUnitOnCheckFailure was pruned from
    the menu universe entirely (zero-usage clutter — see test_helper_menu_prune.py),
    so it is intentionally NOT asserted present here anymore. #15
    RegenerateUnitOnGroundingFailure (the IterGen-faithful one) and ManagedStep stay.
    """
    p = _load_prompts()
    for name in (
        "RegenerateUnitOnGroundingFailure",
        "ManagedStep",
    ):
        assert name in p._ALL_HELPER_NAMES, f"{name} dropped from helper universe"


def test_grounding_worked_example_removed():
    p = _load_prompts()
    block = p._build_verified_examples_block(None)
    assert "Grounded-and-closed" not in block
    # No few-shot example may DEMONSTRATE either IterGen-style helper (promotion).
    # The signatures stay in the tool-reference menu (fair mechanism) but must not
    # appear in a worked example. Check BOTH helpers, not just the grounding one.
    assert "RegenerateUnitOnGroundingFailure" not in block
    assert "RegenerateUnitOnCheckFailure" not in block
    # The live example library is otherwise intact.
    for title in LIVE_EXAMPLE_TITLES:
        assert title in block, f"live example unexpectedly missing: {title!r}"


def test_dangling_grounded_unit_prefix_removed():
    p = _load_prompts()
    assert "// Grounded-unit constrained CSD." not in p._VERIFIED_EXAMPLE_PREFIXES


def test_unit_rewind_hint_removed_from_feedback():
    src = _FEEDBACK_PATH.read_text()
    assert "_unit_rewind_hint" not in src, "the _unit_rewind_hint nudge must be gone"


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
