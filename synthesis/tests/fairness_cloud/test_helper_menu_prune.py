"""TDD guard for the IterGen-style helper menu prune (2026-06-18).

Context: the usefulness audit (saved-results/itergen-style-helper-audit-20260617.md)
grepped every recorded/accepted win body for `helpers.<Name>(` calls. Only #6
RollbackConstrainedToComplete (keystone, 4+ clean cold wins) and #5
RollbackConstrainedSuffix (a21_gsm7b clean win) are used by any clean win; the
rest of the IterGen-style family is zero-usage clutter on the author's menu.

User ruling (2026-06-18): prune the ~10 unused IterGen-style helpers from the
AUTHOR'S MENU. This is fair (a subtraction — gives our method nothing the
baselines lack) and board-neutral (zero recorded win used any removed helper).
Scope = MENU-ONLY: the names are stripped from `TOOL_REFERENCE` so they leave
`_ALL_HELPER_NAMES` (the bandit/mask universe). The Dafny library is NOT edited:
#2 RollbackToValidPrefix and #3 RollbackToCompletePrefix stay in the library as
hidden internals because kept helpers #5/#6 still call them; they just leave the
author-visible menu. The dead methods that no win calls remain in the .dfy
(invisible to the author) and are harmless.

10 removed from the menu: #2 RollbackToValidPrefix, #3 RollbackToCompletePrefix,
#4 RollbackConstrainedSpan, #7 RollbackAndRegenerate (already library-only),
#12 RolloutConstrainedWithPenalties, #14 RegenerateUnitOnCheckFailure.

The 2026-07-15 full helper audit restored #8 RollbackAndContinue, the paired
#10/#11 snapshot operations, and #13 SpeculativeConstrainedRollout. They are
general verified controls that do not inspect hidden answers. Snapshot values
must be redacted from trace logs before these helpers are author-visible.
4 kept (still bandit-selectable): #5 RollbackConstrainedSuffix,
#6 RollbackConstrainedToComplete, #9 DeadEndDetection,
#15 RegenerateUnitOnGroundingFailure.

Universe: 72 before; #7 was already library-only (not in the universe), so 9 of
the 10 actually leave -> 63 after.

Run:  python -m pytest tests/fairness_cloud/test_helper_menu_prune.py -q
"""

import importlib.util
import ast
import pathlib

_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[2]  # .../_focal_pull (or csd-generation on focal)
_PROMPTS_PATH = _REPO / "generate" / "prompts.py"
if not _PROMPTS_PATH.exists():  # focal layout: package root is synthesis/
    _PROMPTS_PATH = _REPO / "synthesis" / "generate" / "prompts.py"
_FEEDBACK_LOOP_PATH = _REPO / "evaluate" / "feedback_loop.py"
if not _FEEDBACK_LOOP_PATH.exists():
    _FEEDBACK_LOOP_PATH = _REPO / "synthesis" / "evaluate" / "feedback_loop.py"


def _load_prompts():
    spec = importlib.util.spec_from_file_location("_prompts_menu_prune", _PROMPTS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pipeline_helper_sets():
    tree = ast.parse(_FEEDBACK_LOOP_PATH.read_text(encoding="utf-8"))
    pipeline = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SynthesisPipeline"
    )
    values = {}
    for node in pipeline.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"NON_PRUNABLE_HELPERS", "PRUNABLE_HELPERS"}:
                values[node.targets[0].id] = ast.literal_eval(node.value)
    return values


REMOVED = (
    "RollbackToValidPrefix",
    "RollbackToCompletePrefix",
    "RollbackConstrainedSpan",
    "RollbackAndRegenerate",
    "RolloutConstrainedWithPenalties",
    "RegenerateUnitOnCheckFailure",
)
KEPT = (
    "RollbackConstrainedSuffix",
    "RollbackConstrainedToComplete",
    "DeadEndDetection",
    "RegenerateUnitOnGroundingFailure",
)
RESTORED_BY_FULL_AUDIT = (
    "RollbackAndContinue",
    "SaveLogitsSnapshot",
    "RestoreLogitsSnapshot",
    "SpeculativeConstrainedRollout",
)
# A few unrelated helpers that must survive the prune untouched (sanity guard
# against accidentally over-deleting the menu).
UNRELATED_SURVIVORS = (
    "ConstrainedStep",
    "CraneGeneration",
    "OpenConstrainedSpan",
    "CloseConstrainedSpan",
    "AdaptiveConstrainedStep",
)

# Fair prompt-grounding helpers that read ONLY prompt-visible text. Both must be
# in the menu universe. Together with the later author-visible
# DeadEndAvoidingStep, they are the reason the universe is 66
# (= blessed 63 + 2 prompt-grounding helpers + 1 dead-end step), not 63.
PROMPT_GROUNDING_HELPERS = (
    "PrefixAppearsInPrompt",
    "PrefixResemblesPromptExamples",
)

# The unfair molecule-class helper (removed 2026-07-01): it called the SMILES
# scorer's own class-membership function + our hardcoded CLASS_MOTIFS at decode
# time, leaking the answer key, and its name/description named the dataset. It was
# replaced by the fair PrefixResemblesPromptExamples (prompt-visible RDKit
# resemblance, no scorer import). It must appear nowhere in the menu.
UNFAIR_REMOVED = (
    "PrefixMatchesPromptMoleculeClass",
    "SpanMatchesPromptMoleculeClass",
)

EXPECTED_UNIVERSE_SIZE = 70


def test_removed_helpers_left_the_universe():
    p = _load_prompts()
    for name in REMOVED:
        assert name not in p._ALL_HELPER_NAMES, f"{name} should be pruned from the menu universe"


def test_kept_helpers_still_in_universe():
    p = _load_prompts()
    for name in KEPT:
        assert name in p._ALL_HELPER_NAMES, f"{name} must stay selectable"


def test_audited_helpers_are_restored_to_the_universe_and_menu():
    p = _load_prompts()
    ref = p._build_tool_reference_block(None)
    for name in RESTORED_BY_FULL_AUDIT:
        assert name in p._ALL_HELPER_NAMES, f"{name} must be selectable after the full audit"
        assert f"helpers.{name}(" in ref, f"helpers.{name}( missing from rendered menu"


def test_snapshot_pair_cannot_be_split_by_adaptive_menu_pruning():
    helper_sets = _pipeline_helper_sets()
    snapshot_pair = {"SaveLogitsSnapshot", "RestoreLogitsSnapshot"}

    assert snapshot_pair <= helper_sets["NON_PRUNABLE_HELPERS"]
    assert snapshot_pair.isdisjoint(helper_sets["PRUNABLE_HELPERS"])


def test_unrelated_helpers_untouched():
    p = _load_prompts()
    for name in UNRELATED_SURVIVORS:
        assert name in p._ALL_HELPER_NAMES, f"{name} was over-pruned"


def test_universe_size():
    p = _load_prompts()
    assert len(p._ALL_HELPER_NAMES) == EXPECTED_UNIVERSE_SIZE, (
        f"universe is {len(p._ALL_HELPER_NAMES)}, expected {EXPECTED_UNIVERSE_SIZE}"
    )


def test_prompt_grounding_helpers_present():
    p = _load_prompts()
    for name in PROMPT_GROUNDING_HELPERS:
        assert name in p._ALL_HELPER_NAMES, f"{name} must stay selectable"
    ref = p._build_tool_reference_block(None)
    for name in PROMPT_GROUNDING_HELPERS:
        assert f"helpers.{name}(" in ref, f"helpers.{name}( missing from menu"


def test_unfair_molecule_class_helper_removed():
    p = _load_prompts()
    ref = p._build_tool_reference_block(None)
    for name in UNFAIR_REMOVED:
        assert name not in p._ALL_HELPER_NAMES, f"{name} must not be in the menu universe"
        assert name not in ref, f"{name} must not appear anywhere in the rendered menu"


def test_removed_call_shapes_gone_from_rendered_menu():
    p = _load_prompts()
    ref = p._build_tool_reference_block(None)
    for name in REMOVED:
        assert f"helpers.{name}(" not in ref, f"helpers.{name}( still rendered in menu"
        assert f"CSDHelpers.{name}(" not in ref, f"CSDHelpers.{name}( still rendered in menu"
    # #14's name must not survive anywhere in the menu, incl. #15's Role prose
    # (which previously said "same control loop as RegenerateUnitOnCheckFailure").
    assert "RegenerateUnitOnCheckFailure" not in ref


def test_kept_call_shapes_present_in_rendered_menu():
    p = _load_prompts()
    ref = p._build_tool_reference_block(None)
    for name in KEPT:
        assert f"helpers.{name}(" in ref, f"helpers.{name}( unexpectedly missing from menu"


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
